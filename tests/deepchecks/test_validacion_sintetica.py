"""Pruebas de los chequeos de DeepChecks con datos sintéticos.

Corren en el entorno aislado de validacion_datos/ (``make validar-datos``).
En el entorno del proyecto se omiten porque deepchecks no está instalado.
"""

import numpy as np
import pandas as pd
import pytest

try:
    from validacion_datos import checks
except Exception:  # deepchecks ausente o incompatible con el entorno del proyecto
    pytest.skip("Correr con `make test-deepchecks`", allow_module_level=True)

VARIABLES = [
    "salidas_lag1",
    "salidas_media7",
    "dias_historia",
    "dia_semana",
    "es_fin_semana",
    "mes",
    "dia_mes",
    "capacidad",
]
ESQUEMA = {
    "p1": {
        "features": VARIABLES,
        "label": "salidas",
        "cat_features": ["dia_semana", "es_fin_semana"],
        "datetime": "fecha",
        "llaves": ["estacion_id", "fecha"],
    },
    "viajes": {
        "features": ["origen_id", "destino_id", "duracion_min", "hora", "dia_semana"],
        "cat_features": ["origen_id", "destino_id", "dia_semana"],
        "datetime": "inicio_viaje",
        "llaves": ["viaje_id"],
    },
}


def _panel(inicio, dias, semilla=0, n_estaciones=20):
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range(inicio, periods=dias)
    filas = []
    for estacion in range(1, n_estaciones + 1):
        base = 5 + estacion
        for i, fecha in enumerate(fechas):
            salidas = rng.poisson(base)
            filas.append(
                {
                    "estacion_id": estacion,
                    "fecha": fecha,
                    "salidas_lag1": float(rng.poisson(base)),
                    "salidas_media7": float(base + rng.normal(0, 1)),
                    "dias_historia": 300 + i,
                    "dia_semana": fecha.dayofweek,
                    "es_fin_semana": int(fecha.dayofweek >= 5),
                    "mes": fecha.month,
                    "dia_mes": fecha.day,
                    "capacidad": 10 + estacion,
                    "salidas": salidas,
                }
            )
    return pd.DataFrame(filas)


def _viajes(inicio, n, semilla=0, desplazamiento=0):
    rng = np.random.default_rng(semilla)
    inicios = pd.Timestamp(inicio) + pd.to_timedelta(rng.integers(0, 28 * 24 * 60, n), "min")
    duracion = rng.gamma(2, 7, n) + desplazamiento
    return pd.DataFrame(
        {
            "viaje_id": np.arange(n) + semilla * 1_000_000,
            "inicio_viaje": inicios,
            "origen_id": rng.integers(1, 30, n),
            "destino_id": rng.integers(1, 30, n),
            "duracion_min": duracion,
            "hora": inicios.hour,
            "dia_semana": inicios.dayofweek,
        }
    )


@pytest.fixture(scope="module")
def train():
    return _panel("2026-06-01", 60)


@pytest.fixture(scope="module")
def prueba():
    return _panel("2026-09-01", 30, semilla=1)


def _por_chequeo(resultados, texto):
    return [r for r in resultados if texto in r.chequeo or texto in r.datos]


def test_panel_limpio_pasa_bloqueantes(train, prueba):
    resultados = checks.integridad_panel(train, ESQUEMA, "train") + checks.train_test_panel(
        train, prueba, ESQUEMA, "train vs test"
    )
    assert checks.bloqueantes_fallidos(resultados) == []


def test_detecta_estacion_dia_duplicado(train):
    sucio = pd.concat([train, train.head(5)], ignore_index=True)
    resultados = checks.integridad_panel(sucio, ESQUEMA, "train")
    fallidos = checks.bloqueantes_fallidos(resultados)
    assert any("Duplicates" in r.chequeo for r in fallidos)


def test_detecta_objetivo_nulo(train):
    sucio = train.copy()
    sucio.loc[:9, "salidas"] = np.nan
    fallidos = checks.bloqueantes_fallidos(checks.integridad_panel(sucio, ESQUEMA, "train"))
    assert any("Nulls" in r.chequeo and "objetivo" in r.datos for r in fallidos)


def test_rezagos_nulos_al_inicio_no_bloquean(train):
    con_huecos = train.copy()
    primeras = con_huecos.groupby("estacion_id").head(1).index
    con_huecos.loc[primeras, ["salidas_lag1", "salidas_media7"]] = np.nan
    resultados = checks.integridad_panel(con_huecos, ESQUEMA, "train")
    assert checks.bloqueantes_fallidos(resultados) == []


def test_detecta_fechas_de_prueba_dentro_de_entrenamiento(train):
    traslape = _panel("2026-07-15", 30, semilla=2)
    fallidos = checks.bloqueantes_fallidos(
        checks.train_test_panel(train, traslape, ESQUEMA, "train vs test")
    )
    assert any("Leakage" in r.chequeo for r in fallidos)


def test_detecta_drift_en_variable(train, prueba):
    cambiada = prueba.copy()
    cambiada["salidas_media7"] = cambiada["salidas_media7"] * 3
    resultados = checks.train_test_panel(train, cambiada, ESQUEMA, "train vs test")
    drift = _por_chequeo(resultados, "Feature Drift")[0]
    assert not drift.paso and not drift.bloqueante


def test_drift_estructural_del_calendario_no_se_reporta(train, prueba):
    resultados = checks.train_test_panel(train, prueba, ESQUEMA, "train vs test")
    drift = _por_chequeo(resultados, "Feature Drift")[0]
    assert drift.paso


def test_viajes_limpios_pasan_y_duplicados_fallan():
    viajes = _viajes("2026-06-01", 3000)
    assert checks.bloqueantes_fallidos(checks.integridad_viajes(viajes, ESQUEMA, "train")) == []
    sucio = pd.concat([viajes, viajes.head(3)], ignore_index=True)
    fallidos = checks.bloqueantes_fallidos(checks.integridad_viajes(sucio, ESQUEMA, "train"))
    assert any("Duplicates" in r.chequeo for r in fallidos)


def test_viajes_detectan_drift_de_duracion():
    train = _viajes("2026-06-01", 3000)
    prueba = _viajes("2026-09-01", 3000, semilla=1, desplazamiento=25)
    resultados = checks.train_test_viajes(train, prueba, ESQUEMA, "train vs test")
    assert checks.bloqueantes_fallidos(resultados) == []
    assert not _por_chequeo(resultados, "Feature Drift")[0].paso
