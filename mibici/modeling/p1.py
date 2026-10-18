"""P1: predecir las salidas diarias por estacion y registrar resultados en MLflow."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mibici.config import SEMANAS_BASELINE_P1
from mibici.features import dataset_p1
from mibici.tracking import iniciar_run, registrar_metricas


def calcular_metricas(datos, prediccion):
    reales = datos["salidas"]
    return {
        "mae": float(mean_absolute_error(reales, prediccion)),
        "rmse": float(np.sqrt(mean_squared_error(reales, prediccion))),
    }


def predecir_baseline(datos):
    columna = f"salidas_mismo_dia_media{SEMANAS_BASELINE_P1}"
    return datos[columna].fillna(datos["salidas_media7"]).fillna(0)


def evaluar_conjunto(datos, conjunto):
    subconjunto = datos.loc[datos["conjunto"] == conjunto]
    if subconjunto.empty:
        raise ValueError(f"No hay datos para {conjunto}")
    prediccion = predecir_baseline(subconjunto)
    return calcular_metricas(subconjunto, prediccion)


def ejecutar(viajes, estaciones):
    datos = dataset_p1(viajes, estaciones)
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
