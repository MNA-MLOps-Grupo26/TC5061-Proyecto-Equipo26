"""Prueba de carga en rampa contra el sistema de Docker Compose (Semana 10).

Levanta la API y MLflow con Docker Compose, registra ``docker stats`` mientras
Locust sube de 0 a N usuarios y deja los resultados en ``reports/locust/<config>/``:

- ``carga_stats.csv`` y ``carga_stats_history.csv``: resultados de Locust (no van a Git).
- ``reporte.html``: reporte interactivo de Locust.
- ``docker-stats.jsonl``: CPU y memoria por contenedor, con marca de tiempo.
- ``resumen.json``: parámetros, entorno y métricas agregadas (base de gráficas y FinOps).

Configuraciones:
- ``base``: además ``pruebas_carga/compose-base.yml`` (sin ``OMP_NUM_THREADS``, como
  el Compose original de la semana 10).
- ``omp1``: además ``pruebas_carga/compose-omp1.yml`` (``OMP_NUM_THREADS=1`` en la API,
  hoy también el valor por defecto del Compose).

Requisitos: Docker con Compose, el modelo exportado en ``models/p1/`` y el entorno de
``requirements-locust.txt``. Uso desde la raíz del repositorio:

    python scripts/prueba_carga.py --config base
    python scripts/prueba_carga.py --config omp1
    python scripts/graficas_carga.py
"""

import argparse
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import threading
import time
import urllib.request

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
CONFIGS = {
    "base": ["docker-compose.yml", "pruebas_carga/compose-base.yml"],
    "omp1": ["docker-compose.yml", "pruebas_carga/compose-omp1.yml"],
}
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def compose(archivos, *args):
    cmd = ["docker", "compose"]
    for archivo in archivos:
        cmd += ["-f", str(RAIZ / archivo)]
    return subprocess.run(cmd + list(args), cwd=RAIZ, check=True, capture_output=True, text=True)


def esperar_api(url, limite_s=180):
    fin = time.monotonic() + limite_s
    while time.monotonic() < fin:
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as r:
                if r.status == 200:
                    return json.loads(r.read())
        except OSError:
            pass
        time.sleep(2)
    raise RuntimeError("La API no respondió /health a tiempo")


def predecir_una_vez(url):
    cuerpo = (RAIZ / "docs/evidencia-api/peticion-predict.json").read_bytes()
    req = urllib.request.Request(
        url + "/predict", data=cuerpo, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


class MuestreoDocker:
    """Lee ``docker stats`` en flujo y guarda una línea JSON por contenedor y lectura."""

    def __init__(self, destino):
        self.destino = destino
        self.proc = None
        self.hilo = None

    def __enter__(self):
        self.proc = subprocess.Popen(
            ["docker", "stats", "--format", "{{json .}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.hilo = threading.Thread(target=self._leer, daemon=True)
        self.hilo.start()
        return self

    def _leer(self):
        with self.destino.open("w", encoding="utf-8") as f:
            for linea in self.proc.stdout:
                limpia = ANSI.sub("", linea).strip()
                if not limpia.startswith("{"):
                    continue
                d = json.loads(limpia)
                f.write(
                    json.dumps(
                        {
                            "t": time.time(),
                            "contenedor": d["Name"],
                            "cpu_pct": float(d["CPUPerc"].rstrip("%")),
                            "mem": d["MemUsage"].split("/")[0].strip(),
                        }
                    )
                    + "\n"
                )
                f.flush()

    def __exit__(self, *exc):
        self.proc.terminate()
        self.proc.wait(timeout=10)
        self.hilo.join(timeout=10)


def mem_mib(texto):
    unidades = {"B": 1 / 2**20, "KiB": 1 / 1024, "MiB": 1, "GiB": 1024}
    m = re.match(r"([\d.]+)\s*([KMG]?i?B)", texto)
    return float(m.group(1)) * unidades[m.group(2)]


def resumir(salida, config, params, entorno, prediccion):
    stats = pd.read_csv(salida / "carga_stats.csv")
    hist = pd.read_csv(salida / "carga_stats_history.csv")
    predict = stats.loc[stats.Name == "/predict"].iloc[0]
    h = hist.loc[hist.Name == "/predict"].copy()
    h = h.loc[h["Requests/s"] > 0]
    h["t"] = h.Timestamp.astype(float)
    docker = pd.read_json(salida / "docker-stats.jsonl", lines=True)
    docker["mem_mib"] = docker.mem.map(mem_mib)
    # Usuarios activos en cada lectura de docker stats (último registro de Locust previo).
    docker = pd.merge_asof(
        docker.sort_values("t"),
        h[["t", "User Count"]].sort_values("t"),
        on="t",
        direction="backward",
    ).dropna(subset=["User Count"])
    bins = list(range(0, params["usuarios"] + 25, 25))
    h["tramo"] = pd.cut(h["User Count"], bins, include_lowest=True)
    docker["tramo"] = pd.cut(docker["User Count"], bins, include_lowest=True)
    api = docker.loc[docker.contenedor.str.contains("api")]
    tramos = []
    for tramo, g in h.groupby("tramo", observed=True):
        d = api.loc[api.tramo == tramo]
        tramos.append(
            {
                "usuarios": f"{int(tramo.left) + 1}-{int(tramo.right)}",
                "segundos": len(g),
                "rps_media": round(float(g["Requests/s"].mean()), 2),
                "p50_ms_mediana": float(g["50%"].median()),
                "p95_ms_mediana": float(g["95%"].median()),
                "fallas_por_s_media": round(float(g["Failures/s"].mean()), 3),
                "api_cpu_pct_media": round(float(d.cpu_pct.mean()), 1) if len(d) else None,
            }
        )
    meseta = h.loc[h["User Count"] == h["User Count"].max()]
    contenedores = {}
    for nombre, g in docker.groupby("contenedor"):
        contenedores[nombre] = {
            "cpu_pct_media": round(float(g.cpu_pct.mean()), 1),
            "cpu_pct_max": round(float(g.cpu_pct.max()), 1),
            "mem_mib_media": round(float(g.mem_mib.mean()), 1),
            "mem_mib_max": round(float(g.mem_mib.max()), 1),
        }
    resumen = {
        "config": config,
        "parametros": params,
        "entorno": entorno,
        "prediccion_de_control": prediccion["viajes_esperados"],
        "predict_total": {
            "peticiones": int(predict["Request Count"]),
            "fallas": int(predict["Failure Count"]),
            "rps_media": round(float(predict["Requests/s"]), 2),
            "p50_ms": float(predict["50%"]),
            "p95_ms": float(predict["95%"]),
            "p99_ms": float(predict["99%"]),
            "max_ms": float(predict["Max Response Time"]),
        },
        "meseta_usuarios_maximos": {
            "usuarios": int(meseta["User Count"].max()),
            "segundos": len(meseta),
            "rps_media": round(float(meseta["Requests/s"].mean()), 2),
            "p50_ms_mediana": float(meseta["50%"].median()),
            "p95_ms_mediana": float(meseta["95%"].median()),
        },
        "por_tramo_de_usuarios": tramos,
        "contenedores": contenedores,
    }
    (salida / "resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return resumen


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", choices=CONFIGS, required=True)
    p.add_argument("--usuarios", type=int, default=200)
    p.add_argument("--rampa-s", type=int, default=240, help="Segundos para llegar al máximo")
    p.add_argument("--duracion-s", type=int, default=300, help="Duración total")
    p.add_argument("--host", default="http://localhost:8000")
    p.add_argument("--out", type=Path, default=RAIZ / "reports" / "locust")
    p.add_argument("--mantener", action="store_true", help="No apagar Compose al terminar")
    a = p.parse_args()
    if not (RAIZ / "models/p1/registry.json").exists():
        sys.exit("Falta el modelo exportado en models/p1/ (ver README)")
    salida = a.out / a.config
    salida.mkdir(parents=True, exist_ok=True)
    archivos = CONFIGS[a.config]
    print(f"Levantando Compose ({a.config})", flush=True)
    compose(archivos, "up", "-d", "--build", "--force-recreate")
    try:
        salud = esperar_api(a.host)
        prediccion = predecir_una_vez(a.host)
        params = {
            "usuarios": a.usuarios,
            "tasa_alta_usuarios_por_s": round(a.usuarios / a.rampa_s, 4),
            "rampa_s": a.rampa_s,
            "duracion_s": a.duracion_s,
            "espera_entre_peticiones_s": [0.5, 1.5],
            "mezcla": "/predict 10 : /health 1",
            "compose": archivos,
        }
        entorno = {
            "sistema": platform.platform(),
            "cpu_logicas_host": os.cpu_count(),
            "docker": subprocess.run(
                ["docker", "version", "--format", "{{.Server.Version}}"],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "locust": subprocess.run(
                [sys.executable, "-m", "locust", "--version"],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.split()[1],
            "modelo": salud.get("modelo", {}),
        }
        print("Iniciando Locust", flush=True)
        with MuestreoDocker(salida / "docker-stats.jsonl"):
            # Locust devuelve 1 si hubo fallas; se registran en el resumen, no se ocultan.
            codigo = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "locust",
                    "-f",
                    str(RAIZ / "pruebas_carga/locustfile.py"),
                    "--host",
                    a.host,
                    "--headless",
                    "-u",
                    str(a.usuarios),
                    "-r",
                    str(params["tasa_alta_usuarios_por_s"]),
                    "-t",
                    f"{a.duracion_s}s",
                    "--csv",
                    str(salida / "carga"),
                    "--csv-full-history",
                    "--html",
                    str(salida / "reporte.html"),
                    "--only-summary",
                ],
                cwd=RAIZ,
                check=False,
            ).returncode
            params["codigo_salida_locust"] = codigo
    finally:
        if not a.mantener:
            compose(archivos, "down")
    resumen = resumir(salida, a.config, params, entorno, prediccion)
    print(json.dumps(resumen["predict_total"], indent=2), flush=True)


if __name__ == "__main__":
    main()
