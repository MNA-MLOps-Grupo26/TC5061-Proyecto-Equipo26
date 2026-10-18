"""P1: predecir las salidas diarias por estacion y registrar resultados en MLflow."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mibici.config import FIGURES_DIR, SEMANAS_BASELINE_P1
from mibici.features import dataset_p1
from mibici.tracking import iniciar_run, registrar_artefacto, registrar_metricas


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
