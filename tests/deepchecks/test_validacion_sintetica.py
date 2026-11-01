"""Pruebas de los chequeos de DeepChecks con datos sintéticos.

Corren en el entorno aislado de validacion_datos/ (``make validar-datos``).
En el entorno del proyecto se omiten porque deepchecks no está instalado.
"""

import os

import numpy as np
import pandas as pd
import pytest

# En el entorno del proyecto se omite porque falta deepchecks. `make
# test-deepchecks` define MIBICI_DEEPCHECKS=1: ahí un entorno roto (p. ej. sin
# pkg_resources) o un error en checks.py debe fallar, no omitirse.
if os.environ.get("MIBICI_DEEPCHECKS") != "1":
    pytest.importorskip("deepchecks", reason="Correr con `make test-deepchecks`")
from validacion_datos import checks

# Copia reducida de mibici.validacion.esquema(): las pruebas no importan mibici
# porque corren en el entorno de DeepChecks. tests/test_validacion.py comprueba
# que el esquema real coincide con las variables del proyecto.
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


@pytest.mark.parametrize("columna", ["estacion_id", "fecha"])
def test_detecta_llave_o_fecha_nula_en_panel(train, columna):
    sucio = train.copy()
    sucio.loc[0, columna] = pd.NaT if columna == "fecha" else np.nan
    fallidos = checks.bloqueantes_fallidos(checks.integridad_panel(sucio, ESQUEMA, "train"))
    assert any("Nulls" in r.chequeo and "llaves" in r.datos for r in fallidos)


@pytest.mark.parametrize("columna", ["viaje_id", "inicio_viaje"])
def test_detecta_llave_o_fecha_nula_en_viajes(columna):
    sucio = _viajes("2026-06-01", 3000)
    sucio.loc[0, columna] = pd.NaT if columna == "inicio_viaje" else np.nan
    fallidos = checks.bloqueantes_fallidos(checks.integridad_viajes(sucio, ESQUEMA, "train"))
    assert any("Nulls" in r.chequeo for r in fallidos)


def test_nulos_fuera_de_la_muestra_de_deepchecks_se_detectan():
    # PercentOfNulls muestrea 100 000 filas con random_state=42 por defecto. El
    # nulo se pone justo en la fila que esa muestra deja fuera.
    viajes = _viajes("2026-06-01", 100_001)
    muestra = viajes.sample(100_000, random_state=42).index
    fuera = viajes.index.difference(muestra)[0]
    viajes.loc[fuera, "duracion_min"] = np.nan
    fallidos = checks.bloqueantes_fallidos(checks.integridad_viajes(viajes, ESQUEMA, "train"))
    assert any("Nulls" in r.chequeo for r in fallidos)


def _todos(viajes):
    return viajes[["viaje_id", "inicio_viaje"]]


def test_viajes_sin_particionar_limpios_pasan():
    todos = _todos(
        pd.concat([_viajes("2026-06-01", 3000), _viajes("2026-09-01", 3000, 1)], ignore_index=True)
    )
    resultados = checks.integridad_viajes_todos(todos, ESQUEMA, "todos")
    assert checks.bloqueantes_fallidos(resultados) == []


def test_detecta_inicio_nulo_antes_del_filtro_de_periodo():
    # Un NaT no cumple ``inicio >= TRAIN_START``: el filtro lo descarta y las
    # particiones nunca lo ven. Solo la validación sin particionar lo detecta.
    todos = _todos(_viajes("2026-06-01", 3000))
    todos.loc[0, "inicio_viaje"] = pd.NaT
    fallidos = checks.bloqueantes_fallidos(checks.integridad_viajes_todos(todos, ESQUEMA, "t"))
    assert any("Nulls" in r.chequeo for r in fallidos)


def test_detecta_viaje_id_repetido_entre_particiones():
    train = _viajes("2026-06-01", 3000)
    prueba = _viajes("2026-09-01", 3000, semilla=1)
    prueba.loc[0, "viaje_id"] = train.loc[0, "viaje_id"]
    for parte in (train, prueba):  # cada partición por separado no tiene duplicados
        fallidos = checks.bloqueantes_fallidos(checks.integridad_viajes(parte, ESQUEMA, "p"))
        assert fallidos == []
    todos = _todos(pd.concat([train, prueba], ignore_index=True))
    fallidos = checks.bloqueantes_fallidos(checks.integridad_viajes_todos(todos, ESQUEMA, "t"))
    assert any("Duplicates" in r.chequeo for r in fallidos)


def test_detecta_pocos_textos_en_columna_numerica():
    viajes = _viajes("2026-06-01", 3000)
    viajes["origen_id"] = viajes["origen_id"].astype(object)
    viajes.loc[0, "origen_id"] = "estación 1"  # 1 de 3 000: MixedDataTypes lo deja pasar
    resultados = checks.integridad_viajes(viajes, ESQUEMA, "train")
    fallidos = checks.bloqueantes_fallidos(resultados)
    assert [r.chequeo for r in fallidos] == ["Expected Types"]
    assert "origen_id" in fallidos[0].detalle
    assert _por_chequeo(resultados, "Mixed Data Types")[0].paso


def test_detecta_columna_completa_de_texto_y_fecha_como_texto(train):
    sucio = train.copy()
    sucio["capacidad"] = sucio["capacidad"].astype(str)
    sucio["fecha"] = sucio["fecha"].dt.strftime("%Y-%m-%d")
    tipos = checks.integridad_panel(sucio, ESQUEMA, "train")[0]
    assert tipos.chequeo == "Expected Types" and not tipos.paso and tipos.bloqueante
    assert "capacidad" in tipos.detalle and "fecha" in tipos.detalle
