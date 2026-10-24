"""P1: predecir las salidas diarias por estacion y registrar resultados en MLflow."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mibici.config import FIGURES_DIR, RANDOM_SEED, SEMANAS_BASELINE_P1,P1_HIPERPARAMETROS
from mibici.features import VARIABLES_P1, dataset_p1
from mibici.tracking import iniciar_run, registrar_artefacto, registrar_metricas

# Candidato: pérdida Poisson porque el objetivo es un conteo no negativo con
# muchos días en cero; acepta valores faltantes (estaciones sin capacidad).
HIPERPARAMETROS_CANDIDATO = P1_HIPERPARAMETROS


def calcular_metricas(datos, prediccion):
    """MAE y RMSE en viajes por estación-día; WAPE es el error relativo al volumen total."""
    reales = datos["salidas"]
    errores = np.abs(np.asarray(reales, dtype=float) - np.asarray(prediccion, dtype=float))
    total = float(np.sum(reales))
    return {
        "mae": float(mean_absolute_error(reales, prediccion)),
        "rmse": float(np.sqrt(mean_squared_error(reales, prediccion))),
        "wape": float(errores.sum() / total) if total else float("nan"),
    }


def predecir_baseline(datos):
    columna = f"salidas_mismo_dia_media{SEMANAS_BASELINE_P1}"
    return datos[columna].fillna(datos["salidas_media7"]).fillna(0)


def evaluar_conjunto(datos, conjunto):
    subconjunto = datos.loc[datos["conjunto"] == conjunto]
    if subconjunto.empty:
        raise ValueError(f"No hay datos para {conjunto}")
    prediccion = predecir_baseline(subconjunto)
    guardar_predicciones(subconjunto, prediccion, f"p1-baseline-{conjunto}")
    return calcular_metricas(subconjunto, prediccion)
def ejecutar(viajes, estaciones, modelo="default", learning_rate=None, max_iter=None):
    datos = dataset_p1(viajes, estaciones)

    if modelo == "default":
        run_baseline(datos)
        run_candidato(datos)
    elif modelo == "hgb":
        hiperparametros = P1_HIPERPARAMETROS.copy()

        if learning_rate is not None:
            hiperparametros["learning_rate"] = learning_rate

        if max_iter is not None:
            hiperparametros["max_iter"] = max_iter

        run_candidato(datos, hiperparametros)
    else:
        raise ValueError(f"Modelo no disponible para P1: {modelo}")


def run_baseline(datos):
    """Baseline: promedio del mismo día de la semana en las semanas previas."""
    with iniciar_run(
        "P1",
        "p1-baseline-promedio-semanal",
        modelo="BaselinePromedioSemanal",
        parametros={"semanas": SEMANAS_BASELINE_P1, "horizonte_dias": 1},
    ):
        for etiqueta, conjunto in [("val", "validation"), ("test", "test")]:
            metricas = evaluar_conjunto(datos, conjunto)
            registrar_metricas(etiqueta, metricas)
            print(etiqueta, metricas)


def guardar_predicciones(datos, prediccion, nombre):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tabla = datos[["estacion_id", "fecha", "salidas"]].copy()
    tabla["prediccion"] = prediccion
    ruta = FIGURES_DIR / f"{nombre}-predicciones.csv"
    tabla.to_csv(ruta, index=False)
    registrar_artefacto(ruta, "tablas")
    diario = tabla.groupby("fecha")[["salidas", "prediccion"]].sum()
    eje = diario.plot(figsize=(10, 4), title=f"{nombre}: estaciones evaluadas")
    eje.set_ylabel("Salidas diarias")
    figura = eje.get_figure()
    figura.tight_layout()
    ruta_grafica = FIGURES_DIR / f"{nombre}-grafica.png"
    figura.savefig(ruta_grafica, dpi=150)
    plt.close(figura)
    registrar_artefacto(ruta_grafica, "figuras")


def run_candidato(datos,hiperparametros=None):
    """Candidato: gradient boosting con pérdida Poisson sobre las variables de P1.

    Se ajusta solo con entrenamiento; validación y prueba se evalúan sin reajustar.
    """
    if hiperparametros is None:
        hiperparametros = P1_HIPERPARAMETROS.copy()
    modelo = HistGradientBoostingRegressor(random_state=RANDOM_SEED, **hiperparametros)
    train = datos.loc[datos["conjunto"] == "train"]
    with iniciar_run(
        "P1",
        "p1-hgb-poisson",
        modelo="HistGradientBoostingRegressor",
        parametros={
            **hiperparametros,
            "horizonte_dias": 1,
            "n_variables": len(VARIABLES_P1),
            "fit": "train_only",
        },
    ):
        modelo.fit(train[VARIABLES_P1], train["salidas"])
        registrar_metricas("train", calcular_metricas(train, modelo.predict(train[VARIABLES_P1])))
        for etiqueta, conjunto in [("val", "validation"), ("test", "test")]:
            subconjunto = datos.loc[datos["conjunto"] == conjunto]
            prediccion = modelo.predict(subconjunto[VARIABLES_P1])
            guardar_predicciones(subconjunto, prediccion, f"p1-hgb-{conjunto}")
            registrar_metricas(etiqueta, calcular_metricas(subconjunto, prediccion))
