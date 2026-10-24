"""P1: predecir las salidas diarias por estación y registrar resultados en MLflow.

Modelos (``mlflow run . -e train -P question=P1 -P model=...``):

- ``baseline``: promedio del mismo día de la semana en las semanas previas.
- ``hgb``: ``HistGradientBoostingRegressor`` con pérdida Poisson (referencia).
- ``glm``: ``PoissonRegressor`` con imputación y escalamiento (otra familia).
- ``default``: baseline y hgb, como en la entrega anterior.

La búsqueda de hiperparámetros (``mlflow run . -e search -P question=P1``) usa
``buscar``. Todos los candidatos se ajustan solo con entrenamiento; validación
elige y prueba solo se reporta.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mibici.config import (
    BUSQUEDAS,
    FIGURES_DIR,
    P1_GLM_HIPERPARAMETROS,
    P1_HIPERPARAMETROS,
    RANDOM_SEED,
    SEMANAS_BASELINE_P1,
)
from mibici.features import VARIABLES_P1, dataset_p1
from mibici.tracking import (
    iniciar_run,
    medir_tiempo,
    registrar_artefacto,
    registrar_candidato,
    registrar_metricas,
)

# Se conserva por compatibilidad con la entrega anterior.
HIPERPARAMETROS_CANDIDATO = P1_HIPERPARAMETROS
CONJUNTOS = [("val", "validation"), ("test", "test")]
MODELOS = ("default", "baseline", "hgb", "glm")


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


def construir_modelo(modelo: str, hiper: dict):
    """Estimador sklearn de un candidato de P1."""
    if modelo == "hgb":
        return HistGradientBoostingRegressor(random_state=RANDOM_SEED, **hiper)
    if modelo == "glm":
        return make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), PoissonRegressor(**hiper)
        )
    raise ValueError(f"Modelo no disponible para P1: {modelo}")


def hiperparametros(modelo: str, cambios: dict | None = None) -> dict:
    """Valores por defecto de config.py con los cambios pedidos."""
    base = {"hgb": P1_HIPERPARAMETROS, "glm": P1_GLM_HIPERPARAMETROS}[modelo]
    return {**base, **(cambios or {})}


def ejecutar(viajes, estaciones, modelo="default", learning_rate=None, max_iter=None):
    datos = dataset_p1(viajes, estaciones)
    cambios = {
        clave: valor
        for clave, valor in {"learning_rate": learning_rate, "max_iter": max_iter}.items()
        if valor is not None
    }

    if modelo == "default":
        run_baseline(datos)
        run_candidato(datos)
    elif modelo == "baseline":
        run_baseline(datos)
    elif modelo in ("hgb", "glm"):
        run_candidato(datos, modelo=modelo, hiper=hiperparametros(modelo, cambios))
    else:
        raise ValueError(f"Modelo no disponible para P1: {modelo}")


def run_baseline(datos):
    """Baseline: promedio del mismo día de la semana en las semanas previas."""
    with iniciar_run(
        "P1",
        "p1-baseline-promedio-semanal",
        modelo="BaselinePromedioSemanal",
        parametros={"semanas": SEMANAS_BASELINE_P1, "horizonte_dias": 1},
        etapa="baseline",
    ):
        for etiqueta, conjunto in CONJUNTOS:
            metricas = evaluar_conjunto(datos, conjunto)
            registrar_metricas(etiqueta, metricas)
            print(etiqueta, metricas)


def guardar_predicciones(datos, prediccion, nombre):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tabla = datos[["estacion_id", "fecha", "salidas"]].copy()
    tabla["prediccion"] = prediccion
    ruta = FIGURES_DIR / f"{nombre}-predicciones.csv"
    tabla.to_csv(ruta, index=False)
    registrar_artefacto(ruta, "tables")
    diario = tabla.groupby("fecha")[["salidas", "prediccion"]].sum()
    eje = diario.plot(figsize=(10, 4), title=f"{nombre}: estaciones evaluadas")
    eje.set_ylabel("Salidas diarias")
    figura = eje.get_figure()
    figura.tight_layout()
    ruta_grafica = FIGURES_DIR / f"{nombre}-grafica.png"
    figura.savefig(ruta_grafica, dpi=150)
    plt.close(figura)
    registrar_artefacto(ruta_grafica, "plots")


def guardar_real_vs_predicho(datos, prediccion, nombre):
    """Real contra predicho (hexbin) y distribución de residuales en una figura."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    real = np.asarray(datos["salidas"], dtype=float)
    pred = np.asarray(prediccion, dtype=float)
    residual = real - pred
    figura, (izq, der) = plt.subplots(1, 2, figsize=(11, 4.5))
    hb = izq.hexbin(real, pred, gridsize=40, bins="log", cmap="Blues", mincnt=1)
    tope = float(max(real.max(), pred.max(), 1))
    izq.plot([0, tope], [0, tope], color="#d6604d", lw=1, label="y = x")
    izq.set(xlabel="Salidas reales", ylabel="Salidas predichas", title="Real vs. predicho")
    izq.legend(loc="upper left")
    figura.colorbar(hb, ax=izq, label="log10(estación-días)")
    der.hist(residual, bins=60, color="#2166ac", alpha=0.85)
    der.axvline(0, color="#d6604d", lw=1)
    der.set(xlabel="Residual (real - predicho)", ylabel="Estación-días", title="Residuales")
    figura.suptitle(nombre)
    figura.tight_layout()
    ruta = FIGURES_DIR / f"{nombre}-real-vs-predicho.png"
    figura.savefig(ruta, dpi=150)
    plt.close(figura)
    registrar_artefacto(ruta, "plots")


def nombre_candidato(modelo: str, hiper: dict) -> str:
    if modelo == "hgb":
        return f"p1-hgb-lr{hiper['learning_rate']}-it{hiper['max_iter']}"
    return f"p1-glm-alpha{hiper['alpha']}"


def run_candidato(datos, hiperparametros_run=None, *, modelo="hgb", hiper=None, etapa="candidate"):
    """Entrena un candidato de P1 y registra métricas, gráficas y artefactos.

    Se ajusta solo con entrenamiento; validación y prueba se evalúan sin reajustar.
    ``hiperparametros_run`` se conserva por compatibilidad (equivale a ``hiper``).
    """
    hiper = hiper or hiperparametros_run or hiperparametros(modelo)
    estimador = construir_modelo(modelo, hiper)
    nombre = nombre_candidato(modelo, hiper)
    train = datos.loc[datos["conjunto"] == "train"]
    clase = type(estimador[-1] if hasattr(estimador, "steps") else estimador).__name__

    with iniciar_run(
        "P1",
        nombre,
        modelo=clase,
        parametros={
            **hiper,
            "horizonte_dias": 1,
            "n_variables": len(VARIABLES_P1),
            "fit": "train_only",
            "selection_metric": "val_wape",
        },
        etapa=etapa,
    ):
        with medir_tiempo("fit_time_s"):
            estimador.fit(train[VARIABLES_P1], train["salidas"])
        registrar_metricas(
            "train", calcular_metricas(train, estimador.predict(train[VARIABLES_P1]))
        )
        for etiqueta, conjunto in CONJUNTOS:
            subconjunto = datos.loc[datos["conjunto"] == conjunto]
            prediccion = estimador.predict(subconjunto[VARIABLES_P1])
            guardar_predicciones(subconjunto, prediccion, f"{nombre}-{conjunto}")
            guardar_real_vs_predicho(subconjunto, prediccion, f"{nombre}-{conjunto}")
            registrar_metricas(etiqueta, calcular_metricas(subconjunto, prediccion))
        registrar_candidato(
            estimador,
            train[VARIABLES_P1],
            hiper,
            nombre_datos="p1-train-estacion-dia",
            y_train=train["salidas"],
            extra_config={"pregunta": "P1", "modelo": modelo, "objetivo": "salidas"},
        )


def buscar(viajes, estaciones, modelo="hgb"):
    """Búsqueda con nested runs: un padre y un hijo por combinación de BUSQUEDAS."""
    combinaciones = BUSQUEDAS[("P1", modelo)]
    datos = dataset_p1(viajes, estaciones)
    with iniciar_run(
        "P1",
        f"p1-busqueda-{modelo}",
        modelo=type(construir_modelo(modelo, hiperparametros(modelo))).__name__,
        parametros={"search_method": "grid", "n_combinations": len(combinaciones)},
        etapa="search",
    ):
        for cambios in combinaciones:
            run_candidato(
                datos, modelo=modelo, hiper=hiperparametros(modelo, cambios), etapa="search"
            )
