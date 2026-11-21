"""Gráficas y resumen comparativo de las pruebas de carga (Semana 10).

Lee ``reports/locust/<config>/`` (generado por ``scripts/prueba_carga.py``) y escribe en
``docs/evidencia-week10-locust/``:

- ``latencia_vs_usuarios.png``: p50 y p95 de /predict según usuarios activos.
- ``rps_vs_usuarios.png``: peticiones por segundo atendidas.
- ``cpu_api_vs_usuarios.png``: CPU del contenedor de la API (100% = un núcleo).
- ``resumen-carga.json``: resúmenes de ambas configuraciones, sin rutas locales.

Uso desde la raíz: ``python scripts/graficas_carga.py``.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
# Paleta categórica validada (azul y naranja; contraste y separación CVD verificados).
SERIES = {
    "base": ("#2a78d6", "Compose base"),
    "omp1": ("#eb6834", "OMP_NUM_THREADS=1"),
}
SUPERFICIE, TINTA, TINTA_2, REJILLA = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def estilo(ax, titulo, ylabel):
    ax.set_facecolor(SUPERFICIE)
    ax.set_title(titulo, loc="left", fontsize=11, color=TINTA)
    ax.set_xlabel("Usuarios activos", color=TINTA_2)
    ax.set_ylabel(ylabel, color=TINTA_2)
    ax.grid(axis="y", color=REJILLA, linewidth=0.8)
    ax.tick_params(colors=TINTA_2, labelsize=9)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(REJILLA)
    # Margen superior para que la leyenda no tape las líneas.
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)


def historia(carpeta):
    h = pd.read_csv(carpeta / "carga_stats_history.csv")
    h = h.loc[(h.Name == "/predict") & (h["Requests/s"] > 0)].copy()
    h["t"] = h.Timestamp.astype(float)
    # Los últimos segundos corresponden al apagado de Locust, no a la carga.
    h = h.iloc[:-5]
    # Mediana móvil de 10 s para leer la tendencia sin ocultar el nivel.
    for col in ["50%", "95%", "Requests/s"]:
        h[col + "_suave"] = h[col].rolling(10, min_periods=3, center=True).median()
    return h


def por_usuarios(frame, columnas):
    """Mediana por número de usuarios: la meseta final queda como un solo punto."""
    return frame.groupby("User Count", as_index=False)[columnas].median()


def cpu_api(carpeta, h):
    d = pd.read_json(carpeta / "docker-stats.jsonl", lines=True)
    d = d.loc[d.contenedor.str.contains("api")].sort_values("t")
    d = d.loc[d.t.between(h.t.min(), h.t.max())]
    d = pd.merge_asof(d, h[["t", "User Count"]].sort_values("t"), on="t").dropna()
    d["cpu_suave"] = d.cpu_pct.rolling(5, min_periods=2, center=True).median()
    return d


def etiqueta_final(ax, x, y, texto, color):
    x, y = x[y.notna()], y[y.notna()]
    ax.annotate(
        texto,
        (x.iloc[-1], y.iloc[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=TINTA,
    )
    ax.plot(x.iloc[-1], y.iloc[-1], "o", ms=5, color=color, mec=SUPERFICIE, mew=2)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--entrada", type=Path, default=RAIZ / "reports" / "locust")
    p.add_argument("--salida", type=Path, default=RAIZ / "docs" / "evidencia-week10-locust")
    a = p.parse_args()
    a.salida.mkdir(parents=True, exist_ok=True)
    datos = {c: a.entrada / c for c in SERIES if (a.entrada / c / "resumen.json").exists()}
    if not datos:
        raise SystemExit("No hay resultados en reports/locust/; correr scripts/prueba_carga.py")
    hist = {c: historia(d) for c, d in datos.items()}
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SUPERFICIE})

    fig, ejes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for ax, col, titulo in zip(
        ejes, ["50%", "95%"], ["Latencia p50 de /predict", "Latencia p95 de /predict"]
    ):
        for c, h in hist.items():
            color, nombre = SERIES[c]
            h = por_usuarios(h, [col + "_suave"])
            ax.plot(h["User Count"], h[col + "_suave"], color=color, lw=2, label=nombre)
            etiqueta_final(ax, h["User Count"], h[col + "_suave"], nombre, color)
        estilo(ax, titulo, "Milisegundos")
    ejes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(a.salida / "latencia_vs_usuarios.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for c, h in hist.items():
        color, nombre = SERIES[c]
        h = por_usuarios(h, ["Requests/s_suave"])
        ax.plot(h["User Count"], h["Requests/s_suave"], color=color, lw=2, label=nombre)
        etiqueta_final(ax, h["User Count"], h["Requests/s_suave"], nombre, color)
    estilo(ax, "Peticiones por segundo atendidas en /predict", "Peticiones por segundo")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(a.salida / "rps_vs_usuarios.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for c, d in datos.items():
        color, nombre = SERIES[c]
        cpu = por_usuarios(cpu_api(d, hist[c]), ["cpu_suave"])
        ax.plot(cpu["User Count"], cpu["cpu_suave"], color=color, lw=2, label=nombre)
        etiqueta_final(ax, cpu["User Count"], cpu["cpu_suave"], nombre, color)
    estilo(ax, "CPU del contenedor de la API (100% = 1 núcleo)", "CPU (%)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(a.salida / "cpu_api_vs_usuarios.png", dpi=200)
    plt.close(fig)

    resumen = {}
    for c, d in datos.items():
        r = json.loads((d / "resumen.json").read_text(encoding="utf-8"))
        # Nombres de contenedor genéricos (dependen del nombre de la carpeta local).
        r["contenedores"] = {
            ("api" if "api" in k else "mlflow" if "mlflow" in k else k): v
            for k, v in r["contenedores"].items()
        }
        resumen[c] = r
    (a.salida / "resumen-carga.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Gráficas y resumen en", a.salida)


if __name__ == "__main__":
    main()
