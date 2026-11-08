"""Arranque, verificación del modelo, /health y errores 500 controlados de la API P1."""

import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest

from api.esquemas import VARIABLES_P1
from api.main import crear_app
from api.modelo import ErrorModelo, ModeloServido, cargar_modelo

REPO = Path(__file__).resolve().parents[1]
VALIDO = json.loads((REPO / "docs/evidencia-api/peticion-predict.json").read_text())
LINAJE = {
    "nombre": "mibici-equipo26-p1-demand",
    "version": "1",
    "alias_consultado": "champion",
    "run_id": "run",
    "logged_model_id": "m-1",
    "source_sha": "sha",
    "data_version": "dv",
}


class ModeloFijo:
    def __init__(self, salida):
        self.salida = salida

    def predict(self, datos):
        return self.salida if self.salida is not None else np.zeros(len(datos))


def cliente_con(salida=None):
    modelo = ModeloServido(ModeloFijo(salida), LINAJE, 8)
    return TestClient(crear_app(lambda: modelo))


def test_health_con_linaje():
    with cliente_con() as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "estado": "ok",
        "modelo": LINAJE,
        "verificacion": {"hashes": "ok", "firma": "ok", "archivos_verificados": 8},
    }


def test_docs_disponible():
    with cliente_con() as c:
        assert c.get("/docs").status_code == 200
        esquema = c.get("/openapi.json").json()["components"]["schemas"]["RegistroP1"]
    assert esquema["examples"][0]["estacion_id"] == 21


@pytest.mark.parametrize(
    "salida",
    [np.array([-1.0]), np.array([np.nan]), np.array([np.inf]), np.array([1.0, 2.0])],
    ids=["negativa", "nan", "infinita", "longitud_distinta"],
)
def test_salida_invalida_da_500_controlado(salida):
    with cliente_con(salida) as c:
        r = c.post("/predict", json=VALIDO)
    assert r.status_code == 500
    assert r.json()["detail"].startswith("Salida inválida del modelo")


# --- Verificación del modelo exportado -----------------------------------------


def _sha(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


@pytest.fixture
def exportado(tmp_path):
    archivos = {"MLmodel": b"flavors: {}", "model.skops": b"binario"}
    for nombre, datos in archivos.items():
        (tmp_path / nombre).write_bytes(datos)
    registry = {"modelo": LINAJE, "archivos": {n: _sha(d) for n, d in archivos.items()}}
    (tmp_path / "registry.json").write_text(json.dumps(registry))
    return tmp_path


def carga_falsa(nombres=VARIABLES_P1):
    return lambda _: (ModeloFijo(None), list(nombres))


def test_carga_correcta(exportado):
    modelo = cargar_modelo(exportado, carga_falsa())
    assert modelo.archivos_verificados == 2
    assert modelo.info == {"nombre": LINAJE["nombre"], "version": "1", "run_id": "run"}


def test_hash_alterado_impide_arrancar(exportado):
    (exportado / "model.skops").write_bytes(b"alterado")
    with pytest.raises(ErrorModelo, match="SHA-256"):
        cargar_modelo(exportado, carga_falsa())


def test_archivo_sobrante_impide_arrancar(exportado):
    (exportado / "extra.txt").write_text("x")
    with pytest.raises(ErrorModelo, match="sobran"):
        cargar_modelo(exportado, carga_falsa())


def test_archivo_faltante_impide_arrancar(exportado):
    (exportado / "MLmodel").unlink()
    with pytest.raises(ErrorModelo, match="faltan"):
        cargar_modelo(exportado, carga_falsa())


@pytest.mark.parametrize(
    "nombres",
    [VARIABLES_P1[:-1], list(reversed(VARIABLES_P1)), []],
    ids=["corta", "orden", "vacia"],
)
def test_firma_incompatible_impide_arrancar(exportado, nombres):
    with pytest.raises(ErrorModelo, match="firma"):
        cargar_modelo(exportado, carga_falsa(nombres))


def test_app_no_arranca_si_falla_la_verificacion(exportado):
    (exportado / "model.skops").write_bytes(b"alterado")
    app = crear_app(lambda: cargar_modelo(exportado, carga_falsa()))
    with pytest.raises(ErrorModelo), TestClient(app):
        pass


def test_sin_model_path_no_arranca(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    with pytest.raises(RuntimeError, match="MODEL_PATH"), TestClient(crear_app()):
        pass
