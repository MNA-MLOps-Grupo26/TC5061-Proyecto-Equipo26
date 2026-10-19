"""Pruebas de cortes, cobertura y normalizacion de perfiles P3."""

import numpy as np
import pandas as pd
import pytest

from mibici.features import columnas_perfil_p3, preparar_perfiles_p3


def viajes(filas):
    result = pd.DataFrame(filas, columns=["inicio_viaje", "fin_viaje", "origen_id", "destino_id"])
    result["viaje_id"] = range(len(result))
    return result


def catalogo():
    return pd.DataFrame(
        {
            "estacion_id": [1, 2, 3],
            "nombre": ["A", "B", "C"],
            "latitud": [20.6, np.nan, 20.7],
            "longitud": [-103.3, np.nan, -103.4],
        }
    )


def test_corte_independiente_y_eventos_futuros_no_cambian_perfil():
    base = viajes(
        [
            ("2025-08-01 08:00", "2025-08-01 09:00", 1, 2),
            ("2025-08-02 08:00", "2025-08-02 09:00", 1, 2),
            ("2026-07-31 23:50", "2026-08-01 00:10", 1, 2),
            ("2025-07-31 23:50", "2025-08-01 00:10", 2, 1),
        ]
    )
    r = preparar_perfiles_p3(base, catalogo()).set_index("estacion_id")
    assert r.loc[1, "total_salidas"] == 2
    assert r.loc[2, "total_llegadas"] == 1
    future = viajes([("2026-09-01 08:00", "2026-09-01 09:00", 3, 3)])
    pd.testing.assert_frame_equal(
        preparar_perfiles_p3(base, catalogo()),
        preparar_perfiles_p3(pd.concat([base, future]), catalogo()),
    )


def test_criterios_historia_normalizacion_y_metadatos():
    dates = pd.date_range("2025-08-01", periods=36)
    rows = [
        (d + pd.Timedelta(hours=8), d + pd.Timedelta(hours=18), 1, 1)
        for d in dates
        for _ in range(3)
    ]
    r = preparar_perfiles_p3(viajes(rows), catalogo()).set_index("estacion_id")
    assert r.loc[1, "elegible"]
    assert r.loc[1, "dias_con_eventos"] == 35
    assert r.loc[1, "total_movimientos"] == 210
    for tipo in ("salidas", "llegadas"):
        cols = [c for c in columnas_perfil_p3() if c.startswith(tipo)]
        assert r.loc[1, cols].astype(float).sum() == pytest.approx(1)
    assert r.loc[2, "cobertura_fuente_diaria"] == "desconocida"
    assert not r.loc[2, "elegible"] and not r.loc[2, "mapeable"]
    assert "sin_eventos_entrenamiento" in r.loc[2, "motivo_exclusion"]
    assert r.loc[2, columnas_perfil_p3()].isna().all()


def test_intervalo_largo_no_equivale_dias_observados():
    rows = [
        ("2025-08-01 08:00", "2025-08-01 09:00", 1, 1),
        ("2026-07-30 08:00", "2026-07-30 09:00", 1, 1),
    ]
    r = preparar_perfiles_p3(viajes(rows), catalogo()).iloc[0]
    assert r["dias_denominador_proxy"] > 300
    assert r["dias_con_eventos"] == 1
    assert not r["elegible"]
    assert "weekday_sin_eventos" in r["motivo_exclusion"]


def test_solo_una_direccion_y_catalogo_incompleto():
    dates = pd.date_range("2025-08-01", periods=36)
    rows = [
        (d + pd.Timedelta(hours=8), d + pd.Timedelta(hours=9), 1, 9)
        for d in dates
        for _ in range(3)
    ]
    r = preparar_perfiles_p3(viajes(rows), catalogo()).set_index("estacion_id")
    assert "sin_llegadas" in r.loc[1, "motivo_exclusion"]
    assert r.loc[9, "sin_catalogo"] and not r.loc[9, "mapeable"]
    assert "sin_salidas" in r.loc[9, "motivo_exclusion"]


def test_faltan_weekdays_aunque_haya_volumen():
    dates = pd.date_range("2025-08-01", periods=100)
    rows = [
        (d + pd.Timedelta(hours=8), d + pd.Timedelta(hours=9), 1, 1)
        for d in dates
        if d.dayofweek < 5
        for _ in range(2)
    ]
    r = preparar_perfiles_p3(viajes(rows), catalogo()).iloc[0]
    assert r["dias_con_eventos"] >= 28 and r["total_movimientos"] >= 100
    assert not r["elegible"] and "weekday_sin_eventos" in r["motivo_exclusion"]


def test_rechaza_periodo_de_prueba_y_catalogo_duplicado():
    empty = viajes([])
    with pytest.raises(ValueError, match="entrenamiento"):
        preparar_perfiles_p3(empty, catalogo(), fin="2026-10-01")
    with pytest.raises(ValueError, match="duplicado"):
        preparar_perfiles_p3(empty, pd.concat([catalogo(), catalogo()]))


def test_entrada_no_se_modifica_y_vacio_es_valido():
    empty = viajes([])
    original = empty.copy(deep=True)
    r = preparar_perfiles_p3(empty, catalogo())
    pd.testing.assert_frame_equal(empty, original)
    assert len(r) == 3 and not r.elegible.any()
