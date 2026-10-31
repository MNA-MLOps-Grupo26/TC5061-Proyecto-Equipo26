"""Reproducibilidad del entrenamiento de P1 y P2 con datos sintéticos.

Construye viajes falsos, arma los paneles con el mismo código de features.py y
entrena cada candidato dos veces con la configuración de config.py. Con la
misma semilla, predicciones y métricas deben ser idénticas. No descarga datos
ni registra runs en MLflow.
"""

import numpy as np
import pandas as pd
import pytest

from mibici.config import RANDOM_SEED, TEST_START, VALIDATION_START
from mibici.features import VARIABLES_P1, VARIABLES_P2, dataset_p1, dataset_p2
from mibici.modeling import p1, p2

# Iteraciones reducidas para que la suite siga siendo rápida.
CAMBIOS_RAPIDOS = {"hgb": {"max_iter": 40}, "glm": {}, "logreg": {}}


def _viajes_sinteticos(n_estaciones: int = 6, semilla: int = 0) -> pd.DataFrame:
    """Viajes entre estaciones de junio a septiembre de 2026 (train, val y test)."""
    rng = np.random.default_rng(semilla)
    dias = pd.date_range(pd.Timestamp(VALIDATION_START) - pd.Timedelta(days=60), "2026-09-30")
    filas = []
    viaje_id = 0
    for dia in dias:
        for origen in range(1, n_estaciones + 1):
            for _ in range(rng.poisson(8 + 2 * origen + 4 * (dia.dayofweek < 5))):
                inicio = dia + pd.Timedelta(minutes=int(rng.integers(5 * 60, 23 * 60)))
                filas.append(
                    {
                        "viaje_id": viaje_id,
                        "inicio_viaje": inicio,
                        "fin_viaje": inicio + pd.Timedelta(minutes=int(rng.integers(3, 45))),
                        "origen_id": origen,
                        "destino_id": int(rng.integers(1, n_estaciones + 1)),
                    }
                )
                viaje_id += 1
    return pd.DataFrame(filas)


def _estaciones_sinteticas(n_estaciones: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "estacion_id": range(1, n_estaciones + 1),
            "capacidad": 10 + 2 * np.arange(n_estaciones),
            "latitud": np.linspace(20.65, 20.70, n_estaciones),
            "longitud": np.linspace(-103.40, -103.35, n_estaciones),
        }
    )


@pytest.fixture(scope="module")
def paneles():
    viajes = _viajes_sinteticos()
    estaciones = _estaciones_sinteticas()
    return {"P1": dataset_p1(viajes, estaciones), "P2": dataset_p2(viajes, estaciones)}


def _predicciones_p1(modelo, datos):
    hiper = p1.hiperparametros(modelo, CAMBIOS_RAPIDOS[modelo])
    estimador = p1.construir_modelo(modelo, hiper)
    train = datos.loc[datos["conjunto"] == "train"]
    val = datos.loc[datos["conjunto"] == "validation"]
    estimador.fit(train[VARIABLES_P1], train["salidas"])
    return val, estimador.predict(val[VARIABLES_P1])


def test_paneles_sinteticos_cubren_los_tres_conjuntos(paneles):
    for panel in paneles.values():
        assert set(panel["conjunto"]) == {"train", "validation", "test"}
    assert paneles["P2"]["categoria"].nunique() >= 2


def test_construccion_de_paneles_es_determinista():
    viajes, estaciones = _viajes_sinteticos(), _estaciones_sinteticas()
    pd.testing.assert_frame_equal(dataset_p1(viajes, estaciones), dataset_p1(viajes, estaciones))
    pd.testing.assert_frame_equal(dataset_p2(viajes, estaciones), dataset_p2(viajes, estaciones))


@pytest.mark.parametrize("modelo", ["hgb", "glm"])
def test_p1_misma_semilla_mismas_predicciones_y_metricas(paneles, modelo):
    datos = paneles["P1"]
    val, pred_a = _predicciones_p1(modelo, datos)
    # Se ensucia el estado global de numpy: el resultado no debe depender de él.
    np.random.seed(RANDOM_SEED + 1)
    np.random.rand(100)
    _, pred_b = _predicciones_p1(modelo, datos)

    np.testing.assert_array_equal(pred_a, pred_b)
    assert p1.calcular_metricas(val, pred_a) == p1.calcular_metricas(val, pred_b)


def _probabilidades_p2(modelo, datos):
    """Entrena P2 y devuelve probabilidades y métricas como en p2.run_candidato."""
    hiper = p2.hiperparametros(modelo, CAMBIOS_RAPIDOS[modelo])
    estimador = p2.construir_modelo(modelo, hiper)
    train = datos.loc[datos["conjunto"] == "train"]
    val = datos.loc[datos["conjunto"] == "validation"]
    estimador.fit(train[VARIABLES_P2], train["categoria"])
    clases = list(estimador.classes_)
    probabilidades = estimador.predict_proba(val[VARIABLES_P2])
    prediccion = pd.Series(np.asarray(clases)[probabilidades.argmax(axis=1)], index=val.index)
    puntaje = pd.Series(1 - probabilidades[:, clases.index(p2.SIN_RIESGO)], index=val.index)
    return probabilidades, p2.calcular_metricas(val, prediccion, puntaje)


@pytest.mark.parametrize("modelo", ["logreg", "hgb"])
def test_p2_misma_semilla_mismas_probabilidades_y_metricas(paneles, modelo):
    datos = paneles["P2"]
    proba_a, metricas_a = _probabilidades_p2(modelo, datos)
    np.random.seed(RANDOM_SEED + 1)
    np.random.rand(100)
    proba_b, metricas_b = _probabilidades_p2(modelo, datos)

    np.testing.assert_array_equal(proba_a, proba_b)
    assert metricas_a == metricas_b


@pytest.mark.parametrize("modulo, modelo", [(p1, "hgb"), (p2, "logreg"), (p2, "hgb")])
def test_estimadores_fijan_la_semilla_del_proyecto(modulo, modelo):
    estimador = modulo.construir_modelo(modelo, modulo.hiperparametros(modelo))
    parametros = estimador.get_params()
    semillas = {v for k, v in parametros.items() if k.endswith("random_state")}
    assert semillas == {RANDOM_SEED}


def test_test_nunca_se_usa_para_entrenar(paneles):
    for panel in paneles.values():
        train = panel.loc[panel["conjunto"] == "train", "fecha"]
        val = panel.loc[panel["conjunto"] == "validation", "fecha"]
        test = panel.loc[panel["conjunto"] == "test", "fecha"]
        assert train.max() < val.min()
        assert val.max() < test.min()
        assert val.min() >= pd.Timestamp(VALIDATION_START)
        assert test.min() >= pd.Timestamp(TEST_START)
