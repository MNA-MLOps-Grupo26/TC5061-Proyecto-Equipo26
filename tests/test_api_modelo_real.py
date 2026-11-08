"""Integración con el modelo real exportado (models/p1).

Se omite si no está definida MIBICI_MODELO_EXPORTADO (ruta al modelo exportado):

    MIBICI_MODELO_EXPORTADO=models/p1 python -m pytest tests/test_api_modelo_real.py
"""

import json
import math
import os
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
EVID = REPO / "docs" / "evidencia-api"
RUTA = os.environ.get("MIBICI_MODELO_EXPORTADO")

pytestmark = pytest.mark.skipif(not RUTA, reason="defina MIBICI_MODELO_EXPORTADO")


@pytest.fixture(scope="module")
def cliente():
    from fastapi.testclient import TestClient

    from api.main import crear_app
    from api.modelo import cargar_modelo

    with TestClient(crear_app(lambda: cargar_modelo(Path(RUTA)))) as c:
        yield c


@pytest.fixture(scope="module")
def pyfunc():
    import mlflow.pyfunc

    return mlflow.pyfunc.load_model(RUTA)


def _directo(pyfunc, registros):
    from api.esquemas import VARIABLES_P1, RegistroP1

    filas = [RegistroP1.model_validate(r).a_fila() for r in registros]
    return list(pyfunc.predict(pd.DataFrame(filas, columns=VARIABLES_P1).astype("float64")))


def test_health_real(cliente):
    cuerpo = cliente.get("/health").json()
    assert cuerpo["estado"] == "ok"
    assert cuerpo["modelo"]["nombre"] == "mibici-equipo26-p1-demand"
    assert cuerpo["verificacion"]["archivos_verificados"] == 8


def test_api_igual_a_pyfunc_y_referencia(cliente, pyfunc):
    lote = json.loads((EVID / "peticion-batch.json").read_text())
    ref = json.loads((EVID / "prediccion-referencia.json").read_text())
    api = [
        p["viajes_esperados"]
        for p in cliente.post("/predict/batch", json=lote).json()["predicciones"]
    ]
    uno = cliente.post("/predict", json=lote["registros"][0]).json()["viajes_esperados"]

    # Idéntica al modelo cargado directo en la misma máquina (incluye la fila con nulos).
    assert api == _directo(pyfunc, lote["registros"])
    assert uno == api[0]
    # Contra la referencia (calculada en arm64): HistGradientBoosting puede variar en el
    # último bit entre arquitecturas, así que se compara con tolerancia de 1e-12.
    esperado = ref["peticion_batch"]["viajes_esperados_en_orden"]
    assert all(math.isclose(a, e, rel_tol=1e-12) for a, e in zip(api, esperado, strict=True))
