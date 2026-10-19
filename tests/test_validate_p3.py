import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

from mibici.features import columnas_perfil_p3, matriz_forma, perfiles_agosto
from mibici.evaluation import asignar_fijo


def estaciones():
    return pd.DataFrame(
        {
            "estacion_id": [1, 2],
            "nombre": ["a", "b"],
            "primer_evento": pd.to_datetime(["2025-08-01", "2025-08-01"]),
        }
    )


def viajes():
    dates = pd.date_range("2026-08-01", periods=31, freq="D")
    rows = []
    for date in dates:
        for _ in range(4):
            rows.append(
                {
                    "viaje_id": len(rows),
                    "inicio_viaje": date + pd.Timedelta(hours=8),
                    "fin_viaje": date + pd.Timedelta(hours=9),
                    "origen_id": 1,
                    "destino_id": 1,
                }
            )
    return pd.DataFrame(rows)


def test_agosto_preserva_dia1_y_normaliza_por_calendario():
    p = perfiles_agosto(viajes(), estaciones())
    a = p.iloc[0]
    assert a.evaluable
    assert a.dias_con_eventos == 31
    assert a.dias_denominador_proxy == 31
    assert a.eventos_agosto_1 == 8
    assert a.total_movimientos == 248
    for d in range(7):
        assert a[f"salidas_d{d}_h08"] == pytest.approx(1 / 7)
        assert a[f"llegadas_d{d}_h09"] == pytest.approx(1 / 7)
    assert np.isclose(p.iloc[0][columnas_perfil_p3()[:168]].astype(float).sum(), 1)
    assert not p.iloc[1].evaluable
    assert "menos_14_dias" in p.iloc[1].motivo_exclusion


def test_eventos_limites_son_independientes_y_sin_septiembre():
    v = viajes()
    extra = pd.DataFrame(
        [
            {
                "viaje_id": 999,
                "inicio_viaje": pd.Timestamp("2026-07-31 23:55"),
                "fin_viaje": pd.Timestamp("2026-08-01 00:05"),
                "origen_id": 2,
                "destino_id": 1,
            },
            {
                "viaje_id": 1000,
                "inicio_viaje": pd.Timestamp("2026-08-31 23:55"),
                "fin_viaje": pd.Timestamp("2026-09-01 00:05"),
                "origen_id": 1,
                "destino_id": 2,
            },
        ]
    )
    p = perfiles_agosto(pd.concat([v, extra]), estaciones())
    assert p.iloc[0].total_salidas == 125
    assert p.iloc[0].total_llegadas == 125
    assert p.iloc[1].total_movimientos == 0
    assert p.iloc[0].eventos_agosto_1 == 9


def test_historia_previa_requerida():
    e = estaciones()
    e.loc[0, "primer_evento"] = pd.Timestamp("2026-08-01")
    with pytest.raises(ValueError):
        perfiles_agosto(viajes(), e)


def test_asignacion_no_refit_y_excluye_metadatos():
    rng = np.random.default_rng(12)
    x = rng.dirichlet(np.ones(168), size=(12, 2)).reshape(12, 336)
    p = pd.DataFrame(x, columns=columnas_perfil_p3())
    model = KMeans(n_clusters=3, n_init=2, random_state=42).fit(matriz_forma(p))
    before = model.cluster_centers_.copy()
    p["volumen"] = 1e15
    _, labels, dist = asignar_fijo(model, p)
    assert np.array_equal(before, model.cluster_centers_)
    assert np.array_equal(labels, model.labels_)
    assert np.isfinite(dist).all()
