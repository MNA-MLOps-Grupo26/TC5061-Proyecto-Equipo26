import json

import pandas as pd
import pytest

from mibici.config import PROJ_ROOT
from mibici.modeling import p3_base, train


def test_train_usa_p3_autocontenido():
    assert train.EJECUTORES["P3"] is p3_base.ejecutar


def test_llegadas_desde_conserva_viajes_que_cruzan_el_corte():
    viajes = pd.DataFrame(
        {
            "inicio_viaje": ["2026-07-31 23:50", "2026-07-31 10:00"],
            "fin_viaje": ["2026-08-01 00:10", "2026-07-31 10:20"],
        }
    )
    cruzan = p3_base.llegadas_desde(viajes, "2026-08-01")
    assert len(cruzan) == 1
    assert cruzan.fin_viaje.iloc[0] == "2026-08-01 00:10"


def test_comparar_con_congelado_detecta_diferencias():
    spec = json.loads((PROJ_ROOT / "references/p3_frozen.json").read_text(encoding="utf-8"))
    esperado = {k: dict(spec["expected_results"][k]) for k in ("train", "val", "test")}
    assert p3_base.comparar_con_congelado(esperado)
    esperado["test"]["ari_vs_train"] += 0.01
    assert not p3_base.comparar_con_congelado(esperado)


def _catalogo(tmp_path):
    ruta = tmp_path / "estaciones.parquet"
    pd.DataFrame({"estacion_id": [1, 2], "nombre": ["A", "B"]}).to_parquet(ruta, index=False)
    return ruta, {"sha256": p3_base.sha256(ruta), "records": 2}


def test_catalogo_verificado_se_lee(tmp_path):
    ruta, registro = _catalogo(tmp_path)
    assert len(p3_base.leer_catalogo_verificado(ruta, registro)) == 2


def test_catalogo_alterado_se_rechaza(tmp_path):
    ruta, registro = _catalogo(tmp_path)
    pd.DataFrame({"estacion_id": [1, 3], "nombre": ["A", "C"]}).to_parquet(ruta, index=False)
    with pytest.raises(ValueError, match="SHA-256"):
        p3_base.leer_catalogo_verificado(ruta, registro)


def test_catalogo_con_conteo_incorrecto_se_rechaza(tmp_path):
    ruta, registro = _catalogo(tmp_path)
    with pytest.raises(ValueError, match="registros"):
        p3_base.leer_catalogo_verificado(ruta, {**registro, "records": 3})
