"""Pruebas del panel estación-día y del indicador de riesgo de P2."""

import pandas as pd

from mibici.features import (
    agregar_rezagos,
    asignar_dia,
    calcular_indicador_riesgo,
    construir_panel,
    contar_por_estacion_dia,
)


def _viajes(filas):
    df = pd.DataFrame(filas, columns=["inicio_viaje", "fin_viaje", "origen_id", "destino_id"])
    df["inicio_viaje"] = pd.to_datetime(df["inicio_viaje"])
    df["fin_viaje"] = pd.to_datetime(df["fin_viaje"])
    df["viaje_id"] = range(len(df))
    return df


def test_dia_operativo_empieza_a_las_cinco():
    fechas = pd.Series(pd.to_datetime(["2025-08-02 04:59", "2025-08-02 05:00"]))
    dias = asignar_dia(fechas, hora_inicio=5)
    assert dias.dt.day.tolist() == [1, 2]


def test_panel_rellena_ceros_y_rezagos_no_ven_el_futuro():
    viajes = _viajes(
        [
            ("2025-08-01 08:00", "2025-08-01 08:10", 1, 2),
            ("2025-08-03 08:00", "2025-08-03 08:10", 1, 2),
            ("2025-08-03 09:00", "2025-08-03 09:10", 1, 2),
        ]
    )
    panel = agregar_rezagos(construir_panel(contar_por_estacion_dia(viajes)))
    estacion = panel.loc[panel["estacion_id"] == 1].set_index("fecha")

    # 2 de agosto no tuvo viajes pero la estación ya existía: cero real
    assert estacion.loc["2025-08-02", "salidas"] == 0
    # El rezago del 3 de agosto es el 2 de agosto, no el propio 3
    assert estacion.loc["2025-08-03", "salidas_lag1"] == 0
    assert estacion.loc["2025-08-04", "salidas_lag1"] == 2
    assert estacion.loc["2025-08-03", "dias_desde_ultimo_viaje"] == 2


def test_indicador_toma_el_primer_umbral_alcanzado():
    estaciones = pd.DataFrame({"estacion_id": [1, 2], "capacidad": [4, 4]})
    # Estación 1: dos salidas netas (2/4 = 0.5) antes de cualquier llegada -> sin_bicis
    viajes = _viajes(
        [
            ("2025-08-01 07:00", "2025-08-01 07:10", 1, 2),
            ("2025-08-01 07:30", "2025-08-01 07:40", 1, 2),
        ]
    )
    resultado = calcular_indicador_riesgo(viajes, estaciones).set_index("estacion_id")
    assert resultado.loc[1, "categoria"] == "sin_bicis"
    # Estación 2 recibe dos llegadas netas (-0.5) -> sin_anclajes
    assert resultado.loc[2, "categoria"] == "sin_anclajes"
