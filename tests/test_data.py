import pandas as pd
import pytest

from mibici import config
from mibici.features import (
    agregar_variables_temporales,
    particionar_dataframe,
    validar_columnas_viajes,
)


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
    df = pd.DataFrame(
        {
            "viaje_id": [1, 2, 3, 4, 5],
            "inicio_viaje": pd.to_datetime(
                [
                    "2025-07-31 23:59:59",
                    "2025-08-01 00:00:00",
                    "2026-07-31 23:59:59",
                    "2026-08-01 00:00:00",
                    "2026-09-01 00:00:00",
                ]
            ),
        }
    )

    particiones = particionar_dataframe(df)

    assert particiones["train"]["viaje_id"].tolist() == [2, 3]
    assert particiones["validation"]["viaje_id"].tolist() == [4]
    assert particiones["test"]["viaje_id"].tolist() == [5]

    asignados = sum(len(p) for p in particiones.values())
    assert asignados == 4


def test_particion_excluye_fecha_final():
    """La fecha final es exclusiva y no debe entrar en ningún conjunto."""
    df = pd.DataFrame(
        {
            "viaje_id": [1],
            "inicio_viaje": pd.to_datetime(["2026-10-01 00:00:00"]),
        }
    )

    particiones = particionar_dataframe(df)

    assert sum(len(p) for p in particiones.values()) == 0


def test_validar_columnas_viajes():
    """Acepta las columnas requeridas y rechaza las incompletas."""
    df = pd.DataFrame(
        columns=[
            "viaje_id",
            "inicio_viaje",
            "fin_viaje",
            "origen_id",
            "destino_id",
        ]
    )

    validar_columnas_viajes(df)

    with pytest.raises(ValueError, match="Faltan columnas"):
        validar_columnas_viajes(pd.DataFrame(columns=["viaje_id"]))


def test_agregar_variables_temporales():
    """Genera correctamente las variables de calendario."""
    df = pd.DataFrame(
        {
            "inicio_viaje": pd.to_datetime(
                [
                    "2026-08-03 09:30:00",  # Lunes
                    "2026-08-08 18:15:00",  # Sábado
                ]
            )
        }
    )

    resultado = agregar_variables_temporales(
        df,
        "inicio_viaje",
    )

    assert resultado["hora"].tolist() == [9, 18]
    assert resultado["dia_semana"].tolist() == [0, 5]
    assert resultado["mes"].tolist() == [8, 8]
    assert resultado["anio"].tolist() == [2026, 2026]
    assert resultado["es_fin_semana"].tolist() == [0, 1]

    # La función debe conservar el DataFrame original.
    assert "hora" not in df.columns


def test_agregar_variables_temporales_rechaza_columna_inexistente():
    """Informa claramente si la columna temporal no existe."""
    df = pd.DataFrame({"viaje_id": [1]})

    with pytest.raises(ValueError, match="No existe la columna"):
        agregar_variables_temporales(df, "fecha_inexistente")
