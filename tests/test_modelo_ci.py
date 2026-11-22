"""El modelo sintético del smoke test de Compose pasa la verificación de la API."""

import json

import pandas as pd
import pytest

from api.esquemas import VARIABLES_P1
from api.modelo import cargar_modelo
from scripts.modelo_ci import generar


def test_modelo_ci_pasa_verificacion_y_predice(tmp_path):
    destino = tmp_path / "ci-smoke"
    registro = generar(destino)

    modelo = cargar_modelo(destino)
    assert modelo.info == {"nombre": "mibici-ci-smoke", "version": "0", "run_id": "ci-smoke"}
    assert modelo.archivos_verificados == len(registro["archivos"])

    fila = pd.DataFrame([[1.0] * len(VARIABLES_P1)], columns=VARIABLES_P1)
    (valor,) = modelo.predictor.predict(fila)
    assert valor >= 0


def test_modelo_ci_no_sobrescribe(tmp_path):
    destino = tmp_path / "ocupado"
    destino.mkdir()
    (destino / "algo.txt").write_text("x")
    with pytest.raises(SystemExit):
        generar(destino)


def test_registry_marca_el_modelo_como_sintetico(tmp_path):
    destino = tmp_path / "ci-smoke"
    generar(destino)
    registro = json.loads((destino / "registry.json").read_text(encoding="utf-8"))
    assert "smoke test" in registro["proposito"]
