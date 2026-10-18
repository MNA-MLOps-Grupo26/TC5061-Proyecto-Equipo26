"""P2: clasificación del riesgo aproximado de quedarse sin bicicletas o sin anclajes.

Dos runs en MLflow:

- ``p2-baseline-semana-anterior``: repite la categoría del mismo día de la
  semana anterior (sin entrenamiento).
- ``p2-logreg-multiclase``: regresión logística multiclase con pesos
  balanceados, porque los días con riesgo son pocos.

Métricas en validación y prueba: F1 macro (mismo peso a las tres clases),
recall por clase de riesgo y Recall@K (de las estaciones con riesgo en un día,
cuántas aparecen entre las K que el equipo de rebalanceo revisaría primero).
La selección de configuraciones se hace con validación; prueba solo se reporta.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mibici.config import (
    FIGURES_DIR,
    P2_CLASES,
    P2_HORA_INICIO_DIA,
    P2_OCUPACION_INICIAL,
    P2_TOP_K,
    RANDOM_SEED,
)
from mibici.features import VARIABLES_P2, dataset_p2, umbrales_riesgo
from mibici.tracking import iniciar_run, registrar_artefacto, registrar_metricas

SIN_RIESGO, SIN_BICIS, SIN_ANCLAJES = P2_CLASES
CONJUNTOS_EVALUACION = {"val": "validation", "test": "test"}


def recall_at_k(datos: pd.DataFrame, puntaje: pd.Series, k: int = P2_TOP_K) -> float:
    """Promedio diario de la fracción de estaciones con riesgo que quedan en el top K.

    Solo se promedian días con al menos una estación en riesgo.
    """
    tabla = pd.DataFrame(
        {"fecha": datos["fecha"], "riesgo": datos["categoria"] != SIN_RIESGO, "puntaje": puntaje}
    )
    tabla["rango"] = tabla.groupby("fecha")["puntaje"].rank(method="first", ascending=False)
    por_dia = (
        tabla.loc[tabla["riesgo"]]
        .groupby("fecha")["rango"]
        .apply(lambda r: float((r <= k).mean()))
    )
    return float(por_dia.mean()) if len(por_dia) else float("nan")


def calcular_metricas(datos: pd.DataFrame, prediccion: pd.Series, puntaje: pd.Series) -> dict:
    """Métricas de clasificación coherentes con la decisión de rebalanceo."""
    real = datos["categoria"]
    recall_clases = recall_score(
        real, prediccion, labels=[SIN_BICIS, SIN_ANCLAJES], average=None, zero_division=0
    )
    return {
        "f1_macro": f1_score(real, prediccion, labels=list(P2_CLASES), average="macro"),
        "balanced_accuracy": balanced_accuracy_score(real, prediccion),
        "recall_sin_bicis": recall_clases[0],
        "recall_sin_anclajes": recall_clases[1],
        f"recall_at_{P2_TOP_K}": recall_at_k(datos, puntaje),
    }


def guardar_matriz_confusion(datos: pd.DataFrame, prediccion: pd.Series, nombre: str):
    """Guarda la matriz de confusión normalizada por clase real."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figura, eje = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        datos["categoria"],
        prediccion,
        labels=list(P2_CLASES),
        normalize="true",
        values_format=".2f",
        cmap="Blues",
        ax=eje,
    )
    eje.set_title(nombre)
    figura.tight_layout()
    ruta = FIGURES_DIR / f"{nombre}.png"
    figura.savefig(ruta, dpi=150)
    plt.close(figura)
    return ruta


def guardar_distribucion(datos: pd.DataFrame):
    """Guarda la distribución agregada de clases por conjunto (sin datos personales)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tabla = (
        datos.groupby("conjunto")["categoria"].value_counts().unstack(fill_value=0).reset_index()
    )
    ruta = FIGURES_DIR / "p2_distribucion_clases.csv"
    tabla.to_csv(ruta, index=False)
    return ruta


def parametros_p2(datos: pd.DataFrame) -> dict:
    """Decisiones de la etiqueta y tamaño de los conjuntos."""
    umbral_bicis, umbral_anclajes = umbrales_riesgo()
    tamanos = datos["conjunto"].value_counts()
    return {
        "dia_operativo_inicio_hora": P2_HORA_INICIO_DIA,
        "ocupacion_inicial": P2_OCUPACION_INICIAL,
        "umbral_sin_bicis": umbral_bicis,
        "umbral_sin_anclajes": umbral_anclajes,
        "top_k": P2_TOP_K,
        "granularidad": "estacion-dia",
        "horizonte": "dia_siguiente",
        "n_train": int(tamanos.get("train", 0)),
        "n_validation": int(tamanos.get("validation", 0)),
        "n_test": int(tamanos.get("test", 0)),
        "n_estaciones": int(datos["estacion_id"].nunique()),
    }


def run_baseline(datos: pd.DataFrame) -> None:
    """Baseline: la categoría del mismo día de la semana anterior."""
    nombre = "p2-baseline-semana-anterior"
    with iniciar_run(
        "P2",
        nombre,
        modelo="BaselineSemanaAnterior",
        parametros={**parametros_p2(datos), "regla": "categoria_lag7; sin historia = sin_riesgo"},
    ):
        for prefijo, conjunto in CONJUNTOS_EVALUACION.items():
            subset = datos.loc[datos["conjunto"] == conjunto]
            prediccion = subset["categoria_lag7"].fillna(SIN_RIESGO)
            # Orden del top K: primero las que la regla marca en riesgo; empate por
            # frecuencia de riesgo en los 28 días previos.
            puntaje = (prediccion != SIN_RIESGO).astype(float) + 0.01 * (
                subset["frac_sin_bicis_28"].fillna(0) + subset["frac_sin_anclajes_28"].fillna(0)
            )
            registrar_metricas(prefijo, calcular_metricas(subset, prediccion, puntaje))
            if prefijo == "val":
                registrar_artefacto(
                    guardar_matriz_confusion(subset, prediccion, f"{nombre}-confusion-val"),
                    "figuras",
                )
        registrar_artefacto(guardar_distribucion(datos), "tablas")


def run_logreg(datos: pd.DataFrame) -> None:
    """Candidato: regresión logística multiclase con imputación y escalamiento."""
    nombre = "p2-logreg-multiclase"
    hiper = {"C": 1.0, "class_weight": "balanced", "max_iter": 2000, "solver": "lbfgs"}
    modelo = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(random_state=RANDOM_SEED, **hiper),
    )

    train = datos.loc[datos["conjunto"] == "train"]
    with iniciar_run(
        "P2",
        nombre,
        modelo="LogisticRegression",
        parametros={
            **parametros_p2(datos),
            **hiper,
            "imputacion": "mediana",
            "escalamiento": "StandardScaler",
            "n_variables": len(VARIABLES_P2),
        },
    ):
        modelo.fit(train[VARIABLES_P2], train["categoria"])
        clases = list(modelo.classes_)
        registrar_metricas(
            "train",
            {
                "f1_macro": f1_score(
                    train["categoria"],
                    modelo.predict(train[VARIABLES_P2]),
                    labels=list(P2_CLASES),
                    average="macro",
                )
            },
        )

        for prefijo, conjunto in CONJUNTOS_EVALUACION.items():
            subset = datos.loc[datos["conjunto"] == conjunto]
            probabilidades = modelo.predict_proba(subset[VARIABLES_P2])
            prediccion = pd.Series(
                np.asarray(clases)[probabilidades.argmax(axis=1)], index=subset.index
            )
            puntaje = pd.Series(
                1 - probabilidades[:, clases.index(SIN_RIESGO)], index=subset.index
            )
            registrar_metricas(prefijo, calcular_metricas(subset, prediccion, puntaje))
            if prefijo == "val":
                registrar_artefacto(
                    guardar_matriz_confusion(subset, prediccion, f"{nombre}-confusion-val"),
                    "figuras",
                )

        coeficientes = pd.DataFrame(modelo[-1].coef_.T, index=VARIABLES_P2, columns=clases).round(
            4
        )
        ruta = FIGURES_DIR / f"{nombre}-coeficientes.csv"
        coeficientes.to_csv(ruta)
        registrar_artefacto(ruta, "tablas")


def ejecutar(viajes: pd.DataFrame, estaciones: pd.DataFrame) -> None:
    """Construye el conjunto de P2 y registra baseline y candidato."""
    datos = dataset_p2(viajes, estaciones)
    run_baseline(datos)
    run_logreg(datos)
