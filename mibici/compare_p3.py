"""Comparacion exploratoria P3, sin seleccionar modelo final ni usar holdouts."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

from mibici.config import (
    P3_K_EXPLORACION,
    P3_N_INIT,
    RANDOM_SEED,
    TRAIN_START,
    VALIDATION_START,
)
from mibici.features_p3 import columnas_perfil_p3

K_VALUES = P3_K_EXPLORACION
SEED = RANDOM_SEED
N_INIT = P3_N_INIT


def matriz_forma(perfiles):
    """Distancia euclidea sobre sqrt(probabilidad)/sqrt(2), sin estandarizar bins."""
    x = perfiles[columnas_perfil_p3()].to_numpy(dtype=float)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Perfiles requieren probabilidades finitas no negativas")
    if not np.allclose(x.reshape(-1, 2, 168).sum(axis=2), 1):
        raise ValueError("Cada direccion debe sumar uno")
    return np.sqrt(x) / np.sqrt(2)


def comparar(perfiles):
    """Diez configuraciones predeclaradas; etiquetas no implican ranking final."""
    if len(perfiles) <= max(K_VALUES):
        raise ValueError("Se necesitan mas de seis estaciones")
    x = matriz_forma(perfiles)
    filas, etiquetas, representantes = [], {}, []
    for metodo in ("kmeans", "ward"):
        for k in K_VALUES:
            modelo = (
                KMeans(
                    n_clusters=k, random_state=SEED, n_init=N_INIT, max_iter=300, algorithm="lloyd"
                )
                if metodo == "kmeans"
                else AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")
            )
            labels = modelo.fit_predict(x)
            sizes = np.bincount(labels, minlength=k)
            if len(np.unique(labels)) != k:
                raise ValueError("Configuracion degenerada: menos grupos que k")
            nombre = f"{metodo}_k{k}"
            etiquetas[nombre] = labels
            filas.append(
                {
                    "configuracion": nombre,
                    "metodo": metodo,
                    "k": k,
                    "silhouette": silhouette_score(x, labels, metric="euclidean"),
                    "tamanos": " / ".join(map(str, sizes)),
                    "minimo": int(sizes.min()),
                    "maximo": int(sizes.max()),
                    "fraccion_mayor": float(sizes.max() / len(x)),
                    "cv_tamano": float(sizes.std() / sizes.mean()),
                    "grupos_unitarios": int((sizes == 1).sum()),
                }
            )
            for grupo in range(k):
                miembros = np.flatnonzero(labels == grupo)
                centro = x[miembros].mean(axis=0)
                dist = np.linalg.norm(x[miembros] - centro, axis=1)
                for rank, pos in enumerate(np.argsort(dist, kind="stable")[:3], 1):
                    row = perfiles.iloc[miembros[pos]]
                    representantes.append(
                        {
                            "configuracion": nombre,
                            "grupo": grupo,
                            "n_estaciones": int(sizes[grupo]),
                            "rango": rank,
                            "estacion_id": row.estacion_id,
                            "nombre": row.get("nombre", ""),
                            "distancia_centro": float(dist[pos]),
                            "movimientos_contexto": row.get("total_movimientos", np.nan),
                        }
                    )
    return pd.DataFrame(filas), etiquetas, pd.DataFrame(representantes)


def guardar_grafica(perfiles, labels, nombre, destino, contexto="solo entrenamiento"):
    raw = perfiles[columnas_perfil_p3()].to_numpy().reshape(-1, 2, 7, 24)
    k = len(np.unique(labels))
    fig, axes = plt.subplots(k, 2, figsize=(13, 2.45 * k), squeeze=False)
    medias = []
    for grupo in range(k):
        mean = raw[labels == grupo].mean(axis=0)
        medias.append(mean.reshape(-1))
        n = int((labels == grupo).sum())
        for direction, color, label in ((0, "#2166ac", "Salidas"), (1, "#d6604d", "Llegadas")):
            axes[grupo, 0].plot(
                range(24), mean[direction].sum(axis=0) * 100, color=color, label=label
            )
            axes[grupo, 1].plot(
                range(7), mean[direction].sum(axis=1) * 100, marker="o", color=color
            )
        axes[grupo, 0].set(
            title=f"Grupo {grupo} · {n} estaciones",
            ylabel="% del perfil direccional",
            xticks=range(0, 24, 3),
        )
        axes[grupo, 1].set(
            xticks=range(7), xticklabels=["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]
        )
        for ax in axes[grupo]:
            ax.grid(alpha=0.2)
            ax.set_ylim(bottom=0)
        axes[grupo, 0].legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("Hora local · suma sobre weekdays")
    axes[-1, 1].set_xlabel("Weekday · suma sobre horas")
    fig.suptitle(
        f"P3 exploratorio · {nombre} · {contexto}\nPromedios de perfiles normalizados; ceros no confirman demanda cero",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(destino / f"perfiles_{nombre}.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(medias, columns=columnas_perfil_p3()).rename_axis("grupo").to_csv(
        destino / f"perfiles_medios_{nombre}.csv"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entrada", type=Path)
    parser.add_argument("salida", type=Path)
    args = parser.parse_args()
    if args.salida.exists():
        raise FileExistsError("La salida debe ser nueva")
    p = pd.read_parquet(args.entrada)
    if not (
        p.conjunto.eq("train").all()
        and p.periodo_inicio.eq(TRAIN_START.isoformat()).all()
        and p.periodo_fin_exclusivo.eq(VALIDATION_START.isoformat()).all()
    ):
        raise ValueError("Solo se admite la preparacion del periodo de entrenamiento")
    p = p.loc[p.elegible].sort_values("estacion_id").reset_index(drop=True)
    if p.estacion_id.duplicated().any():
        raise ValueError("Estaciones duplicadas")
    tabla, etiquetas, reps = comparar(p)
    args.salida.mkdir(parents=True)
    tabla.to_csv(args.salida / "comparacion.csv", index=False)
    reps.to_csv(args.salida / "representantes.csv", index=False)
    asignaciones = p[["estacion_id", "nombre"]].copy()
    for nombre, labels in etiquetas.items():
        asignaciones[nombre] = labels
        guardar_grafica(p, labels, nombre, args.salida)
    asignaciones.to_csv(args.salida / "asignaciones.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for metodo in ("kmeans", "ward"):
        t = tabla[tabla.metodo.eq(metodo)]
        axes[0].plot(t.k, t.silhouette, "o-", label=metodo)
        axes[1].plot(t.k, t.fraccion_mayor, "o-", label=metodo)
    axes[0].set(ylabel="Silhouette euclidea (336 variables transformadas)")
    axes[1].set(ylabel="Fraccion de estaciones en el grupo mayor", ylim=(0, 1))
    for ax in axes:
        ax.set(xlabel="Numero de grupos", xticks=K_VALUES)
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle(f"P3 · comparacion exploratoria · {len(p)} perfiles de entrenamiento")
    fig.tight_layout()
    fig.savefig(args.salida / "comparacion.png", dpi=160)
    plt.close(fig)
    with args.entrada.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    resumen = {
        "estaciones": len(p),
        "variables": 336,
        "k": list(K_VALUES),
        "semilla": SEED,
        "n_init": N_INIT,
        "max_iter": 300,
        "entrada": str(args.entrada.resolve()),
        "sha256": sha,
        "distancia": "euclidea sobre sqrt(p)/sqrt(2); cada bloque direccional suma p=1",
        "escalado": "sin StandardScaler; cada bloque aporta norma cuadrada 1/2",
        "seleccion_final": None,
        "holdouts_leidos": False,
        "mlflow_registrado": False,
    }
    (args.salida / "protocolo.json").write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    print(tabla.to_string(index=False))


if __name__ == "__main__":
    main()
