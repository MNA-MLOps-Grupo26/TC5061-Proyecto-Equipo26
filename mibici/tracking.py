"""Registro homogéneo de experimentos en MLflow para P1, P2 y P3.

Todos los runs del equipo pasan por ``iniciar_run`` para que registren lo mismo:

- Parámetros de reproducción: rango de fechas, partición, data_version y semilla.
- Etiquetas: equipo, pregunta (P1, P2 o P3), modelo, commit de Git y si había
  cambios sin commit.
- Métricas con prefijo del conjunto (``val_mae``, ``test_f1_macro``).

Nunca se registran credenciales ni columnas de usuario.
"""

import os
import subprocess

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import subprocess

from loguru import logger
import mlflow

from mibici.config import (
    DATA_END_EXCLUSIVE,
    EXPERIMENT_NAME,
    MLFLOW_ARTIFACTS_DIR,
    MLFLOW_TRACKING_URI,
    PROJ_ROOT,
    RANDOM_SEED,
    TEAM_NUMBER,
    TEST_START,
    TRAIN_START,
    VALIDATION_START,
)
from mibici.dataset import obtener_data_version

PREGUNTAS = {"P1", "P2", "P3"}
CONJUNTOS = {"train", "val", "test"}


def _git(*args: str) -> str:
    """Ejecuta un comando de Git en la raíz del proyecto."""
    return subprocess.run(
        ["git", *args], cwd=PROJ_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def obtener_commit() -> tuple[str, bool]:
    """Devuelve el commit actual y si hay cambios de código sin commit.

    Solo se revisan archivos versionados (``-uno``): datos, mlflow.db y
    figuras están en .gitignore y no cuentan como cambios de código.
    """
    try:
        commit = _git("rev-parse", "HEAD")
        sucio = bool(_git("status", "--porcelain", "-uno"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("No se pudo leer el commit de Git; se registra como 'desconocido'.")
        return "desconocido", True
    if sucio:
        logger.warning("Hay cambios sin commit: el run no será reproducible con ese commit.")
    return commit, sucio


def configurar_mlflow() -> str:
    """Apunta a la base local y crea el experimento único del equipo si no existe."""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI))
    experimento = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experimento is None:
        MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        experiment_id = mlflow.create_experiment(
            EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACTS_DIR.as_uri()
        )
    else:
        experiment_id = experimento.experiment_id
    mlflow.set_experiment(EXPERIMENT_NAME)
    return experiment_id


def parametros_comunes() -> dict:
    """Parámetros que todo run registra para poder reproducirse."""
    return {
        "train_start": TRAIN_START.isoformat(),
        "validation_start": VALIDATION_START.isoformat(),
        "test_start": TEST_START.isoformat(),
        "data_end_exclusive": DATA_END_EXCLUSIVE.isoformat(),
        "data_version": obtener_data_version(),
        "random_seed": RANDOM_SEED,
    }


@contextmanager
def iniciar_run(
    pregunta: str, nombre: str, modelo: str, parametros: dict | None = None
) -> Iterator[mlflow.ActiveRun]:
    """Abre un run con nombre descriptivo, parámetros comunes y etiquetas del equipo.

    Args:
        pregunta: "P1", "P2" o "P3".
        nombre: nombre descriptivo del run, por ejemplo "p2-baseline-semana-anterior".
        modelo: nombre corto del modelo, por ejemplo "LogisticRegression".
        parametros: hiperparámetros y decisiones propias de la pregunta.
    """
    if pregunta not in PREGUNTAS:
        raise ValueError(f"Pregunta inválida: {pregunta}. Usen P1, P2 o P3.")

    configurar_mlflow()
    commit, sucio = obtener_commit()
    etiquetas = {
        "team": f"equipo{TEAM_NUMBER:02d}",
        "question": pregunta,
        "model": modelo,
        "model_family": modelo,
        "git_commit": commit,
        "git_dirty": str(sucio).lower(),
        "mlflow.source.git.commit": commit,
    }
    # Conservar el identificador del proyecto durante toda la ejecución.
    parent_run_id = os.environ.get("MLFLOW_PROJECT_PARENT_RUN_ID")
    if not parent_run_id:
        parent_run_id = os.environ.pop("MLFLOW_RUN_ID", None)
        if parent_run_id:
            os.environ["MLFLOW_PROJECT_PARENT_RUN_ID"] = parent_run_id

    with mlflow.start_run(
        run_name=nombre,
        tags=etiquetas,
        nested=bool(parent_run_id or mlflow.active_run()),
        parent_run_id=parent_run_id,
    ) as run:
        parametros_run = {**parametros_comunes(), **(parametros or {})}
        mlflow.log_params(parametros_run)

        logger.info(f"Run iniciado: {nombre} ({run.info.run_id})")
        yield run
        logger.success(f"Run terminado: {nombre}")


def registrar_metricas(conjunto: str, metricas: dict[str, float]) -> None:
    """Registra métricas con el conjunto como prefijo, por ejemplo ``val_mae``."""
    if conjunto not in CONJUNTOS:
        raise ValueError(f"Conjunto inválido: {conjunto}. Usen train, val o test.")
    mlflow.log_metrics(
        {f"{conjunto}_{nombre}": float(valor) for nombre, valor in metricas.items()}
    )


def registrar_artefacto(ruta: Path, carpeta: str | None = None) -> None:
    """Registra un archivo (gráfica, tabla agregada) como artefacto del run."""
    mlflow.log_artifact(str(ruta), artifact_path=carpeta)
