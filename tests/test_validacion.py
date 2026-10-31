"""Exportación para DeepChecks: el esquema debe seguir a las variables del proyecto."""

import pandas as pd

from mibici import dataset
from mibici.features import VARIABLES_P1
from mibici.validacion import columnas_viajes, columnas_viajes_todos, esquema, preparar_viajes


def test_esquema_p1_usa_las_variables_del_proyecto():
    p1 = esquema()["p1"]
    assert p1["features"] == VARIABLES_P1
    assert set(p1["cat_features"]) <= set(p1["features"])
    assert p1["label"] not in p1["features"]
    assert p1["datetime"] in p1["llaves"]


def test_esquema_viajes_es_consistente():
    v = esquema()["viajes"]
    assert set(v["cat_features"]) <= set(v["features"])
    assert v["datetime"] not in v["features"]
    assert not set(v["llaves"]) & set(v["features"])


def test_preparar_viajes_exporta_llaves_fecha_y_variables():
    viajes = pd.DataFrame(
        {
            "viaje_id": [1, 2],
            "inicio_viaje": pd.to_datetime(["2026-08-01 07:00", "2026-08-02 23:30"]),
            "fin_viaje": pd.to_datetime(["2026-08-01 07:12", "2026-08-03 00:00"]),
            "origen_id": [10, 11],
            "destino_id": [11, 10],
        }
    )
    original = viajes.copy(deep=True)
    resultado = preparar_viajes(viajes)[columnas_viajes()]

    assert list(resultado.columns) == columnas_viajes()
    assert resultado["duracion_min"].tolist() == [12, 30]
    assert resultado["hora"].tolist() == [7, 23]
    assert resultado["dia_semana"].tolist() == [5, 6]  # sábado y domingo
    pd.testing.assert_frame_equal(viajes, original)


def test_cargar_viajes_sin_filtro_conserva_inicios_nulos_y_fuera_de_periodo(monkeypatch, tmp_path):
    viajes = pd.DataFrame(
        {
            "viaje_id": [1, 2, 3],
            "inicio_viaje": pd.to_datetime(["2026-08-01 07:00", None, "2030-01-01 07:00"]),
        }
    )
    viajes.to_parquet(tmp_path / "datos_abiertos_2026_08.parquet", index=False)
    monkeypatch.setattr(dataset, "PROCESSED_DATA_DIR", tmp_path)

    assert dataset.cargar_viajes()["viaje_id"].tolist() == [1]
    todos = dataset.cargar_viajes(columnas_viajes_todos(), filtrar_periodo=False)
    assert todos["viaje_id"].tolist() == [1, 2, 3]
    assert todos["inicio_viaje"].isna().sum() == 1
    assert list(todos.columns) == ["viaje_id", "inicio_viaje"]
