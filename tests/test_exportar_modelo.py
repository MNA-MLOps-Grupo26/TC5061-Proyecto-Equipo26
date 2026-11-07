"""Exportación con un cliente MLflow doble y artefactos en directorios temporales."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from mibici.features import VARIABLES_P1
from scripts import exportar_modelo as exp


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    args = argparse.Namespace(
        tracking_uri="https://tracking.invalid",
        registry_uri="https://registry.invalid",
        destino=str(tmp_path / "exportado"),
        nombre=exp.NOMBRE,
        alias=exp.ALIAS,
        esperar_version="1",
        esperar_run_id="run-prueba",
        esperar_logged_model_id="m-prueba",
        esperar_experimento=exp.EXPERIMENTO,
        recibo_almacen=str(tmp_path / "recibo.json"),
    )
    identidad = {"id": "almacen-prueba"}
    Path(args.recibo_almacen).write_text(json.dumps({"almacen": identidad}), encoding="utf-8")
    version = SimpleNamespace(
        status="READY",
        version="1",
        run_id="run-prueba",
        model_id="m-prueba",
        source="models:/m-prueba",
        tags={"source_sha": "commit-prueba", "reproduction_of_run_id": "run-anterior"},
    )
    cliente = Mock(
        spec=["get_registered_model", "get_model_version_by_alias", "get_run", "get_experiment"]
    )
    cliente.get_registered_model.return_value = SimpleNamespace(name=args.nombre)
    cliente.get_model_version_by_alias.return_value = version
    cliente.get_run.return_value = SimpleNamespace(
        info=SimpleNamespace(status="FINISHED", run_id="run-prueba", experiment_id="7"),
        data=SimpleNamespace(tags={"git_dirty": "false"}, params={"data_version": "datos-prueba"}),
    )
    cliente.get_experiment.return_value = SimpleNamespace(
        name=args.esperar_experimento, experiment_id="7"
    )
    mlmodel = {"model_id": "m-prueba", "run_id": "run-prueba", "mlflow_version": "prueba"}
    archivos = {"MLmodel": json.dumps(mlmodel).encode(), "data/pesos.bin": b"pesos de prueba"}

    def descargar(uri, *, dst_path):
        carpeta = Path(dst_path) / "modelo"
        for nombre, contenido in archivos.items():
            ruta = carpeta / nombre
            ruta.parent.mkdir(parents=True, exist_ok=True)
            ruta.write_bytes(contenido)
        return str(carpeta)

    modelo = Mock()
    modelo.metadata.get_input_schema.return_value.input_names.return_value = list(VARIABLES_P1)
    modelo.input_example = [{nombre: 1 for nombre in VARIABLES_P1}]
    modelo.predict.return_value = [12.5]
    mlflow = ModuleType("mlflow")
    mlflow.__version__ = "prueba"
    mlflow.MlflowClient = Mock(return_value=cliente)
    mlflow.set_tracking_uri = Mock()
    mlflow.set_registry_uri = Mock()
    mlflow.artifacts = SimpleNamespace(download_artifacts=Mock(side_effect=descargar))
    mlflow.pyfunc = SimpleNamespace(load_model=Mock(return_value=modelo))
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    # También los temporales internos quedan dentro del directorio de esta prueba.
    monkeypatch.setattr(exp.tempfile, "tempdir", str(tmp_path))
    return SimpleNamespace(
        args=args,
        version=version,
        cliente=cliente,
        mlflow=mlflow,
        modelo=modelo,
        archivos=archivos,
        mlmodel=mlmodel,
        identidad=identidad,
    )


@pytest.mark.parametrize("registry_separado", [False, True])
def test_exportacion_correcta_preserva_artefactos_y_linaje(entorno, registry_separado):
    e = entorno
    if not registry_separado:
        e.args.registry_uri = None
    registro = exp.exportar(e.args)
    destino = Path(e.args.destino)
    esperado_registry = e.args.registry_uri or e.args.tracking_uri
    e.mlflow.MlflowClient.assert_called_once_with(
        tracking_uri=e.args.tracking_uri, registry_uri=esperado_registry
    )
    e.cliente.get_model_version_by_alias.assert_called_once_with(e.args.nombre, e.args.alias)
    e.mlflow.artifacts.download_artifacts.assert_called_once()
    assert e.mlflow.artifacts.download_artifacts.call_args.args == (f"models:/{e.args.nombre}/1",)
    e.mlflow.set_tracking_uri.assert_called_once_with(e.args.tracking_uri)
    e.mlflow.set_registry_uri.assert_called_once_with(esperado_registry)
    assert json.loads((destino / "registry.json").read_text()) == registro
    assert registro["archivos"] == {
        nombre: hashlib.sha256(contenido).hexdigest() for nombre, contenido in e.archivos.items()
    }
    for nombre, contenido in e.archivos.items():
        assert (destino / nombre).read_bytes() == contenido
    assert registro["almacen"]["identidad_declarada"] == e.identidad
    assert registro["almacen"]["registry_uri"] == esperado_registry
    assert registro["modelo"]["run_id"] == "run-prueba"
    assert registro["modelo"]["logged_model_id"] == "m-prueba"
    assert registro["modelo"]["source_sha"] == "commit-prueba"
    assert registro["modelo"]["data_version"] == "datos-prueba"
    assert registro["historico"] == {"reproduction_of_run_id": "run-anterior"}
    e.mlflow.pyfunc.load_model.assert_called_once_with(str(destino))
    e.modelo.predict.assert_called_once()
    entrada = e.modelo.predict.call_args.args[0]
    assert list(entrada.columns) == list(VARIABLES_P1)
    assert all(str(tipo) == "float64" for tipo in entrada.dtypes)


def test_destino_existente_no_se_sobrescribe_ni_consulta_registry(entorno):
    e = entorno
    destino = Path(e.args.destino)
    destino.mkdir()
    testigo = destino / "conservar.txt"
    testigo.write_bytes(b"original")
    with pytest.raises(exp.ErrorExportacion, match="ya existe"):
        exp.exportar(e.args)
    assert list(destino.iterdir()) == [testigo]
    assert testigo.read_bytes() == b"original"
    e.mlflow.MlflowClient.assert_not_called()
    e.mlflow.artifacts.download_artifacts.assert_not_called()


@pytest.mark.parametrize(
    ("campo", "valor", "mensaje"),
    [
        ("version", "2", "versión 2"),
        ("run_id", "otro-run", "run_id otro-run"),
        ("model_id", "m-otro", "logged model m-otro"),
    ],
)
def test_identidad_inesperada_no_descarga_ni_crea_destino(entorno, campo, valor, mensaje):
    e = entorno
    setattr(e.version, campo, valor)
    with pytest.raises(exp.ErrorExportacion, match=mensaje):
        exp.exportar(e.args)
    assert not Path(e.args.destino).exists()
    e.cliente.get_model_version_by_alias.assert_called_once_with(e.args.nombre, e.args.alias)
    e.mlflow.artifacts.download_artifacts.assert_not_called()
    e.mlflow.pyfunc.load_model.assert_not_called()


@pytest.mark.parametrize(
    ("campo", "valor", "mensaje"),
    [("run_id", "otro-run", "MLmodel.run_id"), ("model_id", "m-otro", "MLmodel.model_id")],
)
def test_artefacto_con_identidad_distinta_no_se_exporta(entorno, campo, valor, mensaje):
    e = entorno
    e.mlmodel[campo] = valor
    e.archivos["MLmodel"] = json.dumps(e.mlmodel).encode()
    with pytest.raises(exp.ErrorExportacion, match=mensaje):
        exp.exportar(e.args)
    e.mlflow.artifacts.download_artifacts.assert_called_once()
    assert not Path(e.args.destino).exists()
    e.mlflow.pyfunc.load_model.assert_not_called()
