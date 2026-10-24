"""Gráficas y mapas reutilizables.

Contiene las figuras de P3: perfiles medios por grupo y mapa de estaciones.
Las gráficas de P1 y P2 se generan dentro de sus módulos en ``modeling``.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mibici.config import P3_K_EXPLORACION
from mibici.features import columnas_perfil_p3

COLORES = ("#2166ac", "#d6604d", "#1b7837", "#762a83", "#e08214", "#4d4d4d")


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


def guardar_mapa(train, labels, estaciones, destino):
    """Mapa de puntos sin basemap; coordenadas del catalogo septiembre2026."""
    datos = (
        train[["estacion_id", "nombre"]]
        .assign(grupo_train=labels)
        .merge(
            estaciones[["estacion_id", "latitud", "longitud"]],
            on="estacion_id",
            how="left",
            validate="one_to_one",
        )
    )
    valid = datos.latitud.between(-90, 90) & datos.longitud.between(-180, 180)
    datos["ubicacion_valida"] = valid
    datos.to_csv(destino / "estaciones_mapa.csv", index=False)
    datos.loc[~valid].to_csv(destino / "estaciones_sin_ubicacion.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    for g in range(int(np.max(labels)) + 1):
        z = datos.loc[valid & datos.grupo_train.eq(g)]
        ax.scatter(
            z.longitud,
            z.latitud,
            s=25,
            c=COLORES[g],
            label=f"Grupo {g}: {len(z)}",
            alpha=0.8,
            edgecolors="white",
            linewidths=0.3,
        )
    for station in (200, 11, 196):
        z = datos.loc[valid & datos.estacion_id.eq(station)]
        if len(z):
            ax.annotate(
                str(station),
                (z.longitud.iloc[0], z.latitud.iloc[0]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )
    if valid.any():
        ax.set_aspect(1 / np.cos(np.deg2rad(datos.loc[valid, "latitud"].mean())))
    ax.set(
        title=f"P3 · grupos de entrenamiento (k={int(np.max(labels)) + 1})\nCatalogo septiembre2026 · {int(valid.sum())}/{len(datos)} con ubicacion",
        xlabel="Longitud (grados)",
        ylabel="Latitud (grados)",
    )
    ax.text(0.98, 0.98, "N ↑", transform=ax.transAxes, ha="right", va="top")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.text(
        0.5,
        0.015,
        "Mapa de puntos sin cartografia base. Coordenadas actuales no prueban ubicacion historica ni disponibilidad.\nUbicacion y volumen NO usados para agrupar; etiquetas descriptivas con solapamiento.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(destino / "mapa_grupos_train.png", dpi=180)
    plt.close(fig)
    return int(valid.sum()), int((~valid).sum())


def graficar_comparacion_k(tabla, n_perfiles, destino):
    """Silhouette y tamaño del grupo mayor por número de grupos (comparación exploratoria de P3)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for metodo in ("kmeans", "ward"):
        t = tabla[tabla.metodo.eq(metodo)]
        axes[0].plot(t.k, t.silhouette, "o-", label=metodo)
        axes[1].plot(t.k, t.fraccion_mayor, "o-", label=metodo)
    axes[0].set(ylabel="Silhouette euclidea (336 variables transformadas)")
    axes[1].set(ylabel="Fraccion de estaciones en el grupo mayor", ylim=(0, 1))
    for ax in axes:
        ax.set(xlabel="Numero de grupos", xticks=P3_K_EXPLORACION)
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle(f"P3 · comparacion exploratoria · {n_perfiles} perfiles de entrenamiento")
    fig.tight_layout()
    fig.savefig(destino / "comparacion.png", dpi=160)
    plt.close(fig)
