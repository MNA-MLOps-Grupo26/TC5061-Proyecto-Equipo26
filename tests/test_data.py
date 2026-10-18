import pandas as pd

from mibici import config
from mibici.features import particionar_dataframe


def test_particion_temporal_en_orden():
    """La partición no debe mezclar periodos: entrenamiento < validación < prueba."""
    assert config.TRAIN_START < config.VALIDATION_START < config.TEST_START
    assert config.TEST_START < config.DATA_END_EXCLUSIVE


def test_meses_de_descarga_cubren_la_particion():
    assert config.DATA_START_MONTH == config.TRAIN_START.strftime("%Y-%m")
    assert config.DATA_END_MONTH == "2026-09"


def test_nombre_del_experimento_incluye_equipo():
    assert config.EXPERIMENT_NAME == "mibici-equipo26"

def test_particion_temporal_sin_traslapes():
    """Cada viaje debe pertenecer a un solo conjunto temporal."""
    df = pd.DataFrame({
        "viaje_id": [1, 2, 3, 4, 5],
        "inicio_viaje": pd.to_datetime([
            "2025-07-31 23:59:59",
            "2025-08-01 00:00:00",
            "2026-07-31 23:59:59",
            "2026-08-01 00:00:00",
            "2026-09-01 00:00:00",
        ]),
    })

    particiones = particionar_dataframe(df)

    assert particiones["train"]["viaje_id"].tolist() == [2, 3]
    assert particiones["validation"]["viaje_id"].tolist() == [4]
    assert particiones["test"]["viaje_id"].tolist() == [5]

    asignados = sum(len(p) for p in particiones.values())
    assert asignados == 4


def test_particion_excluye_fecha_final():
    """La fecha final es exclusiva y no debe entrar en ningún conjunto."""
    df = pd.DataFrame({
        "viaje_id": [1],
        "inicio_viaje": pd.to_datetime(["2026-10-01 00:00:00"]),
    })

    particiones = particionar_dataframe(df)

    assert sum(len(p) for p in particiones.values()) == 0
