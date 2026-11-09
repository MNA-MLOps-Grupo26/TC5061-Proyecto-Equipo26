"""scripts/registrar_p1.py sobre almacenes temporales y desechables.

Cada prueba crea su propio SQLite en tmp_path con un run y un modelo dobles que
imitan los parámetros y métricas publicados. Nunca toca el almacén entregado.
"""

import pytest

pytest.importorskip("mlflow")

import mlflow
from mlflow import MlflowClient
from sklearn.dummy import DummyRegressor

from scripts import registrar_p1 as reg


@pytest.fixture
def almacen(tmp_path, monkeypatch):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    exp_id = mlflow.create_experiment(
        reg.EXPERIMENTO, artifact_location=(tmp_path / "art").as_uri()
    )
    yield uri, exp_id
    mlflow.set_tracking_uri(None)


def _run(exp_id, metricas=None, **cambios_param):
    import numpy as np

    with mlflow.start_run(experiment_id=exp_id) as run:
        mlflow.log_params({**reg.PARAMETROS_ESPERADOS, **cambios_param})
        mlflow.log_metrics({**reg.METRICAS_ESPERADAS, **(metricas or {})})
        mlflow.set_tags({"git_commit": "8c7c709", "git_dirty": "false"})
        modelo = DummyRegressor().fit(np.zeros((3, 1)), np.ones(3))
        mlflow.sklearn.log_model(modelo, name="model")
    return run.info.run_id


def _main(uri, run_id, *extra):
    return reg.main(["--tracking-uri", uri, "--run-id", run_id, *extra])


def test_revision_no_cambia_nada(almacen):
    uri, exp_id = almacen
    run_id = _run(exp_id)
    assert _main(uri, run_id) == 0
    assert MlflowClient(uri).search_registered_models() == []


def test_aplicar_registra_y_fija_alias_y_luego_reutiliza(almacen):
    uri, exp_id = almacen
    run_id = _run(exp_id)
    assert _main(uri, run_id, "--aplicar") == 0
    cliente = MlflowClient(uri)
    version = cliente.get_model_version_by_alias(reg.NOMBRE, reg.ALIAS)
    assert str(version.version) == "1" and version.run_id == run_id
    assert version.tags["reproduction_of_run_id"] == reg.HISTORICO["reproduction_of_run_id"]
    # Segunda ejecución: reutiliza, no crea otra versión ni cambia el alias.
    assert _main(uri, run_id, "--aplicar") == 0
    assert len(cliente.search_model_versions(f"name='{reg.NOMBRE}'")) == 1


def test_run_con_parametros_distintos_se_detiene(almacen):
    uri, exp_id = almacen
    run_id = _run(exp_id, learning_rate="0.1")
    assert _main(uri, run_id, "--aplicar") == 2
    assert MlflowClient(uri).search_registered_models() == []


def test_otra_version_existente_es_conflicto(almacen):
    uri, exp_id = almacen
    primero = _run(exp_id)
    assert _main(uri, primero, "--aplicar") == 0
    segundo = _run(exp_id)
    assert _main(uri, segundo, "--aplicar") == 2
    cliente = MlflowClient(uri)
    assert len(cliente.search_model_versions(f"name='{reg.NOMBRE}'")) == 1
    assert cliente.get_model_version_by_alias(reg.NOMBRE, reg.ALIAS).run_id == primero


def test_alias_en_otra_version_no_se_reemplaza(almacen):
    uri, exp_id = almacen
    run_id = _run(exp_id)
    cliente = MlflowClient(uri)
    cliente.create_registered_model(reg.NOMBRE)
    otro = _run(exp_id)
    modelos = cliente.search_logged_models(experiment_ids=[exp_id])
    del_otro = next(m for m in modelos if m.source_run_id == otro)
    del_run = next(m for m in modelos if m.source_run_id == run_id)
    v1 = mlflow.register_model(f"models:/{del_otro.model_id}", reg.NOMBRE)
    mlflow.register_model(f"models:/{del_run.model_id}", reg.NOMBRE)
    cliente.set_registered_model_alias(reg.NOMBRE, reg.ALIAS, v1.version)
    assert _main(uri, run_id, "--aplicar") == 2
    assert str(cliente.get_model_version_by_alias(reg.NOMBRE, reg.ALIAS).version) == str(
        v1.version
    )


@pytest.mark.parametrize("valor", [float("nan"), float("inf")])
def test_metrica_no_finita_se_detiene(almacen, valor):
    uri, exp_id = almacen
    run_id = _run(exp_id, metricas={"val_mae": valor})
    assert _main(uri, run_id, "--aplicar") == 2
    assert MlflowClient(uri).search_registered_models() == []


def test_error_de_servidor_al_leer_el_registry_no_escribe(almacen, monkeypatch, capsys):
    """Un fallo que no es "no existe" no se interpreta como alias ausente."""
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import INTERNAL_ERROR

    uri, exp_id = almacen
    run_id = _run(exp_id)
    assert _main(uri, run_id, "--aplicar") == 0
    cliente = MlflowClient(uri)
    cliente.delete_registered_model_alias(reg.NOMBRE, reg.ALIAS)  # alias ausente a propósito
    escrituras = []
    monkeypatch.setattr(
        MlflowClient,
        "get_registered_model",
        lambda self, nombre: (_ for _ in ()).throw(
            MlflowException("fallo simulado del servidor", error_code=INTERNAL_ERROR)
        ),
    )
    monkeypatch.setattr(
        MlflowClient,
        "set_registered_model_alias",
        lambda self, *args: escrituras.append(args),
    )
    assert _main(uri, run_id, "--aplicar") == 3
    assert escrituras == []
    assert "fallo simulado" in capsys.readouterr().err
    monkeypatch.undo()
    assert MlflowClient(uri).get_registered_model(reg.NOMBRE).aliases == {}


def test_modelo_inexistente_si_se_interpreta_como_ausencia(almacen):
    uri, exp_id = almacen
    run_id = _run(exp_id)
    plan = reg.planear(
        MlflowClient(uri),
        reg.NOMBRE,
        reg.ALIAS,
        *reg.verificar_run(MlflowClient(uri), run_id, reg.EXPERIMENTO),
    )
    assert [accion for accion, _ in plan] == ["crear_modelo", "crear_version", "fijar_alias"]
