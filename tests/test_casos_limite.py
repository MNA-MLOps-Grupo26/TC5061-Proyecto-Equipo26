"""Casos de borde con datos sinteticos; no descargan datos ni registran runs."""

import numpy as np
import pandas as pd
import pytest

from mibici.config import SEMANAS_BASELINE_P1
from mibici.features import (
    agregar_rezagos,
    calcular_indicador_riesgo,
    particionar_dataframe,
)
from mibici.modeling.p1 import calcular_metricas, predecir_baseline
from mibici.modeling.p2 import metricas_top_k


def test_baseline_respeta_prioridad_y_ceros_sin_cambiar_entrada():
    datos = pd.DataFrame(
        {
            f"salidas_mismo_dia_media{SEMANAS_BASELINE_P1}": [8, np.nan, np.nan, 0],
            "salidas_media7": [99, 3, np.nan, 99],
        },
        index=[10, 20, 30, 40],
    )
    original = datos.copy(deep=True)

    resultado = predecir_baseline(datos)

    assert resultado.tolist() == [8, 3, 0, 0]
    assert resultado.index.equals(datos.index)
    pd.testing.assert_frame_equal(datos, original)


@pytest.mark.parametrize("predicciones, error", [([0, 0], 0), ([2, 2], 2)])
def test_wape_sin_volumen_es_indefinido(predicciones, error):
    resultado = calcular_metricas(pd.DataFrame({"salidas": [0, 0]}), predicciones)
    assert np.isnan(resultado["wape"])
    assert resultado["mae"] == error
    assert resultado["rmse"] == error


def test_particion_vacia_conserva_columnas():
    datos = pd.DataFrame({"viaje_id": [], "inicio_viaje": pd.to_datetime([])})
    particiones = particionar_dataframe(datos)
    assert set(particiones) == {"train", "validation", "test"}
    for parte in particiones.values():
        assert parte.empty
        assert parte.columns.equals(datos.columns)


def test_particion_rechaza_columna_ausente():
    with pytest.raises(ValueError, match="inicio_viaje"):
        particionar_dataframe(pd.DataFrame({"viaje_id": [1]}))


def test_particion_rechaza_fecha_invalida():
    with pytest.raises(ValueError):
        particionar_dataframe(pd.DataFrame({"inicio_viaje": ["fecha invalida"]}))


def test_rezagos_aislan_estaciones_y_no_usan_presente_ni_futuro():
    # Se intercalan estaciones para detectar mezclas accidentales entre grupos.
    panel = pd.DataFrame(
        {
            "estacion_id": [1, 2] * 9,
            "fecha": np.repeat(pd.date_range("2026-07-28", periods=9), 2),
            "salidas": [2, 100] * 9,
            "llegadas": [1, 50] * 9,
        }
    )
    original = panel.copy(deep=True)
    resultado = agregar_rezagos(panel)
    for estacion, salidas in [(1, 2), (2, 100)]:
        grupo = resultado.loc[resultado.estacion_id.eq(estacion)]
        assert np.isnan(grupo.iloc[0].salidas_lag1)
        assert np.isnan(grupo.iloc[0].salidas_media7)
        assert grupo.iloc[7].salidas_lag7 == salidas
        assert grupo.iloc[7].salidas_media7 == salidas
        assert grupo.iloc[7].salidas_mismo_dia_media4 == salidas

    corte = pd.Timestamp("2026-08-01")
    cambiado = panel.copy(deep=True)
    cambiado.loc[cambiado.fecha.ge(corte), ["salidas", "llegadas"]] = 9999
    nuevo = agregar_rezagos(cambiado)
    columnas = [c for c in resultado if "lag" in c or "media" in c]
    columnas += ["dias_desde_ultimo_viaje"]
    pd.testing.assert_frame_equal(
        resultado.loc[resultado.fecha.le(corte), columnas],
        nuevo.loc[nuevo.fecha.le(corte), columnas],
    )
    pd.testing.assert_frame_equal(panel, original)


def test_top_k_promedia_por_dia_y_respeta_capacidad():
    datos = pd.DataFrame(
        {
            "fecha": ["2026-08-01"] * 3 + ["2026-08-02"],
            "categoria": ["sin_bicis", "sin_anclajes", "sin_riesgo", "sin_riesgo"],
        }
    )
    resultado = metricas_top_k(datos, pd.Series([0.9, 0.8, 0.1, 0.5]), k=1)
    assert resultado["recall_at_1"] == 0.5
    assert resultado["precision_at_1"] == 0.5
    assert resultado["recall_at_1_max_posible"] == 0.5
    assert resultado["estaciones_riesgo_por_dia"] == 1


def test_top_k_empates_siguen_orden_de_entrada():
    datos = pd.DataFrame(
        {
            "fecha": ["2026-08-01"] * 2,
            "categoria": ["sin_riesgo", "sin_bicis"],
        },
        index=[20, 10],
    )
    puntaje = pd.Series([0.5, 0.5], index=datos.index)
    assert metricas_top_k(datos, puntaje, k=1)["recall_at_1"] == 0
    assert metricas_top_k(datos.iloc[::-1], puntaje, k=1)["recall_at_1"] == 1


def test_top_k_mayor_que_estaciones_revisa_solo_las_disponibles():
    datos = pd.DataFrame(
        {
            "fecha": ["2026-08-01"] * 2,
            "categoria": ["sin_bicis", "sin_riesgo"],
        }
    )
    resultado = metricas_top_k(datos, pd.Series([0.9, 0.1]), k=30)
    assert resultado["recall_at_30"] == 1
    assert resultado["precision_at_30"] == 0.5
    assert resultado["recall_at_30_max_posible"] == 1


def test_top_k_sin_riesgo_no_inventa_recall():
    datos = pd.DataFrame({"fecha": ["2026-08-01"], "categoria": ["sin_riesgo"]})
    resultado = metricas_top_k(datos, pd.Series([0.2]), k=1)
    assert np.isnan(resultado["recall_at_1"])
    assert np.isnan(resultado["recall_at_1_max_posible"])
    assert resultado["precision_at_1"] == 0


@pytest.mark.parametrize(
    "cantidad, categoria_origen, categoria_destino",
    [(1, "sin_riesgo", "sin_riesgo"), (2, "sin_bicis", "sin_anclajes")],
)
def test_riesgo_debajo_y_exactamente_en_umbral(cantidad, categoria_origen, categoria_destino):
    viajes = pd.DataFrame(
        {
            "viaje_id": range(cantidad),
            "inicio_viaje": pd.date_range("2026-08-01 07:00", periods=cantidad, freq="h"),
            "fin_viaje": pd.date_range("2026-08-01 07:10", periods=cantidad, freq="h"),
            "origen_id": [1] * cantidad,
            "destino_id": [2] * cantidad,
        }
    )
    estaciones = pd.DataFrame({"estacion_id": [1, 2], "capacidad": [4, 4]})
    resultado = calcular_indicador_riesgo(viajes, estaciones).set_index("estacion_id")
    assert resultado.loc[1, "categoria"] == categoria_origen
    assert resultado.loc[2, "categoria"] == categoria_destino
    assert resultado.loc[1, "indicador_max"] == cantidad / 4
    assert resultado.loc[2, "indicador_min"] == -cantidad / 4


def test_riesgo_excluye_estacion_sin_capacidad_conocida():
    viajes = pd.DataFrame(
        {
            "viaje_id": [1],
            "inicio_viaje": pd.to_datetime(["2026-08-01 07:00"]),
            "fin_viaje": pd.to_datetime(["2026-08-01 07:10"]),
            "origen_id": [1],
            "destino_id": [2],
        }
    )
    estaciones = pd.DataFrame({"estacion_id": [1, 2], "capacidad": [4, np.nan]})
    resultado = calcular_indicador_riesgo(viajes, estaciones)
    assert resultado.estacion_id.tolist() == [1]


def test_constructores_p1_p2_separan_objetivo_de_historia():
    from mibici.features import VARIABLES_P1, VARIABLES_P2, dataset_p1, dataset_p2

    estaciones = pd.DataFrame(
        {
            "estacion_id": [1, 2],
            "capacidad": [4, 4],
            "latitud": [20.6, 20.7],
            "longitud": [-103.3, -103.4],
        }
    )
    viajes = pd.DataFrame(
        {
            "viaje_id": [1, 2],
            "inicio_viaje": pd.to_datetime(["2026-07-31 07:00", "2026-08-01 07:00"]),
            "fin_viaje": pd.to_datetime(["2026-07-31 07:10", "2026-08-01 07:10"]),
            "origen_id": [1, 1],
            "destino_id": [2, 2],
        }
    )
    # El viaje extra cambia el objetivo del 1 de agosto, pero aun no es historia.
    extra = viajes.iloc[[1]].copy().assign(viaje_id=3)
    mas_viajes = pd.concat([viajes, extra], ignore_index=True)
    for construir, variables in [(dataset_p1, VARIABLES_P1), (dataset_p2, VARIABLES_P2)]:
        antes = construir(viajes, estaciones)
        despues = construir(mas_viajes, estaciones)
        assert antes.fecha.min() == pd.Timestamp("2026-08-01")
        primero = antes.loc[antes.fecha.eq("2026-08-01")]
        assert primero.conjunto.eq("validation").all()
        pd.testing.assert_frame_equal(
            primero[variables],
            despues.loc[despues.fecha.eq("2026-08-01"), variables],
        )
        assert antes.loc[antes.fecha.eq("2026-09-01"), "conjunto"].eq("test").all()
        if construir is dataset_p2:
            assert primero.categoria.eq("sin_riesgo").all()
            assert set(despues.loc[despues.fecha.eq("2026-08-01"), "categoria"]) == {
                "sin_bicis",
                "sin_anclajes",
            }


def test_comparacion_p3_rechaza_muestra_insuficiente():
    from mibici.evaluation import comparar

    # El contrato exige mas estaciones que el mayor k (6).
    with pytest.raises(ValueError, match="mas de seis"):
        comparar(pd.DataFrame(index=range(6)))


def test_metricas_p2_predicciones_perfectas_en_tres_clases():
    from mibici.modeling.p2 import calcular_metricas as metricas_p2

    datos = pd.DataFrame(
        {
            "fecha": ["2026-08-01"] * 3,
            "categoria": ["sin_riesgo", "sin_bicis", "sin_anclajes"],
        }
    )
    resultado = metricas_p2(datos, datos.categoria.copy(), pd.Series([0.1, 0.9, 0.8]))
    for clave in ["f1_macro", "balanced_accuracy", "recall_sin_bicis", "recall_sin_anclajes"]:
        assert resultado[clave] == 1
    assert resultado["recall_at_30"] == 1
    assert resultado["precision_at_30"] == pytest.approx(2 / 3)


def test_riesgo_conserva_primer_cruce_aunque_despues_cruce_el_opuesto():
    # Estacion 1 primero pierde dos bicicletas y despues recibe cuatro.
    viajes = pd.DataFrame(
        {
            "viaje_id": range(6),
            "inicio_viaje": pd.date_range("2026-08-01 07:00", periods=6, freq="h"),
            "fin_viaje": pd.date_range("2026-08-01 07:10", periods=6, freq="h"),
            "origen_id": [1, 1, 2, 2, 2, 2],
            "destino_id": [2, 2, 1, 1, 1, 1],
        }
    )
    estaciones = pd.DataFrame({"estacion_id": [1, 2], "capacidad": [4, 4]})
    resultado = calcular_indicador_riesgo(viajes, estaciones).set_index("estacion_id")
    assert resultado.loc[1, "indicador_max"] == 0.5
    assert resultado.loc[1, "indicador_min"] == -0.5
    assert resultado.loc[1, "categoria"] == "sin_bicis"
    assert resultado.loc[2, "categoria"] == "sin_anclajes"
