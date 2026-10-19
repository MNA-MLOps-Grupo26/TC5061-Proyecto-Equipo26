from datetime import date

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from mibici import evaluation, plots
from mibici.dataset import sha256
from mibici.features import columnas_perfil_p3, matriz_forma, perfiles_agosto
from mibici.modeling import p3, train


def test_septiembre_mismos_umbrales_y_primer_dia_completo():
    e = pd.DataFrame(
        {"estacion_id": [1], "nombre": ["a"], "primer_evento": pd.to_datetime(["2025-08-01"])}
    )
    rows = []
    for d in pd.date_range("2026-09-01", periods=30):
        for _ in range(4):
            rows.append(
                {
                    "viaje_id": len(rows),
                    "origen_id": 1,
                    "destino_id": 1,
                    "inicio_viaje": d + pd.Timedelta(hours=8),
                    "fin_viaje": d + pd.Timedelta(hours=9),
                }
            )
    rows.append(
        {
            "viaje_id": 999,
            "origen_id": 2,
            "destino_id": 1,
            "inicio_viaje": pd.Timestamp("2026-08-31 23:59"),
            "fin_viaje": pd.Timestamp("2026-09-01 00:01"),
        }
    )
    rows.append(
        {
            "viaje_id": 1000,
            "origen_id": 1,
            "destino_id": 2,
            "inicio_viaje": pd.Timestamp("2026-09-30 23:59"),
            "fin_viaje": pd.Timestamp("2026-10-01 00:01"),
        }
    )
    q = perfiles_agosto(pd.DataFrame(rows), e, inicio=date(2026, 9, 1), fin=date(2026, 10, 1))
    assert q.iloc[0].evaluable
    assert q.iloc[0].conjunto == "test"
    assert q.iloc[0].dias_con_eventos == 30
    assert q.iloc[0].dias_denominador_proxy == 30
    assert q.iloc[0].total_salidas == 121
    assert q.iloc[0].total_llegadas == 121
    assert q.iloc[0].eventos_agosto_1 == 9


def test_metricas_fijas_no_ajustan_centroides_y_reportan_exclusiones():
    rng = np.random.default_rng(77)
    x = rng.dirichlet(np.ones(168), size=(18, 2)).reshape(18, 336)
    t = pd.DataFrame(x, columns=columnas_perfil_p3())
    t["estacion_id"] = range(18)
    t["nombre"] = "a"
    t["dias_con_eventos"] = 300
    t["total_movimientos"] = 500
    model = KMeans(n_clusters=3, n_init=2, random_state=42).fit(matriz_forma(t))
    before = model.cluster_centers_.copy()
    v = t.copy().assign(evaluable=True)
    v.loc[0, "evaluable"] = False
    q, a, m, g = evaluation.resumir_fijo(model, t, v)
    assert len(q) == 17
    assert m["n_excluidas"] == 1
    assert m["fraccion_estable"] == 1
    assert m["ari_vs_train"] == 1
    assert m["cambio_perfil_mediana"] == 0
    assert len(g) == 3
    assert np.array_equal(before, model.cluster_centers_)
    assert a.estable.all()


def test_entrada_p3_no_carga_viajes_globales_ni_ejecuta_p1_p2(monkeypatch):
    called = []

    def fail():
        raise AssertionError("No debe cargar todos los viajes")

    monkeypatch.setattr(train, "cargar_viajes", fail)
    monkeypatch.setattr(train, "cargar_estaciones", lambda: pd.DataFrame({"estacion_id": [1]}))
    monkeypatch.setattr(train, "EJECUTORES", {"P3": lambda v, e: called.append((v, len(e)))})
    train.main(pregunta="P3")
    assert called == [(None, 1)]


def test_mapa_identifica_estaciones_sin_coordenadas(tmp_path):
    t = pd.DataFrame({"estacion_id": [1, 2], "nombre": ["a", "b"]})
    e = pd.DataFrame(
        {"estacion_id": [1, 2], "latitud": [20.7, np.nan], "longitud": [-103.3, np.nan]}
    )
    assert plots.guardar_mapa(t, np.array([0, 1]), e, tmp_path) == (1, 1)
    missing = pd.read_csv(tmp_path / "estaciones_sin_ubicacion.csv")
    assert missing.estacion_id.tolist() == [2]


def test_reproduccion_rechaza_metricas_distintas_sin_tuning(tmp_path):
    import pytest

    summary = {s: {"n": 10, "silhouette": 0.1} for s in ("train", "val", "test")}
    summary["centroides_sin_refit"] = True
    path = tmp_path / "asignaciones.csv"
    path.write_text("estacion_id,grupo\n1,0\n", encoding="utf-8")
    spec = {
        "historical_run_id": "historico",
        "expected_results": {s: dict(summary[s]) for s in ("train", "val", "test")},
        "assignment_file_sha256": {path.name: sha256(path)},
    }
    evaluation.verificar_reproduccion(summary, tmp_path, spec)
    assert (tmp_path / "verificacion_reproduccion.json").exists()
    summary["test"]["silhouette"] = 0.2
    with pytest.raises(ValueError, match="Reproduccion difiere"):
        evaluation.verificar_reproduccion(summary, tmp_path, spec)


def test_reproduccion_rechaza_asignaciones_distintas(tmp_path):
    import pytest

    summary = {s: {"n": 10} for s in ("train", "val", "test")}
    summary["centroides_sin_refit"] = True
    path = tmp_path / "asignaciones.csv"
    path.write_text("estacion_id,grupo\n1,0\n", encoding="utf-8")
    spec = {
        "historical_run_id": "historico",
        "expected_results": summary,
        "assignment_file_sha256": {path.name: sha256(path)},
    }
    path.write_text("estacion_id,grupo\n1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Asignaciones diferentes"):
        evaluation.verificar_reproduccion(summary, tmp_path, spec)
