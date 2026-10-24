"""Registro homogéneo de experimentos en MLflow para P1, P2 y P3.

Todos los runs del equipo pasan por ``iniciar_run`` para que registren lo mismo:

- Parámetros de reproducción: rango de fechas, partición, data_version y semilla.
- Etiquetas: equipo, pregunta (P1, P2 o P3), modelo, familia del modelo, etapa
  del trabajo (``baseline``, ``search``, ``candidate``), commit de Git y si había
  cambios sin commit.
- Métricas con prefijo del conjunto (``val_mae``, ``test_f1_macro``).

Las corridas candidatas registran además, con ``registrar_candidato``:

- ``model``: el modelo con firma y ejemplo de entrada (``log_model``).
- ``data/``: los manifiestos de la ingesta y el conjunto de entrenamiento como
  input del run (``log_input``, solo esquema, perfil y digest).
- ``config/``: los hiperparámetros efectivos y el ``config.py`` con que corrió.
- ``plots/`` y ``tables/``: gráficas y tablas agregadas de cada pregunta.

Nunca se registran credenciales ni columnas de usuario.
"""

from collections.abc import Iterator
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import time

from loguru import logger
import mlflow
from mlflow.models import infer_signature
import pandas as pd

from mibici.config import (
    DATA_END_EXCLUSIVE,
    DOWNLOADS_MANIFEST,
    EXPERIMENT_NAME,
    MANIFESTS_DIR,
    MLFLOW_ARTIFACTS_DIR,
    MLFLOW_TRACKING_URI,
    PROCESSED_MANIFEST,
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
ETAPAS = {"baseline", "search", "candidate"}
# MLflow 3 serializa los modelos sklearn con skops, que solo carga tipos auditados.
# Estos son los únicos tipos adicionales que usan nuestros modelos (revisados con
# skops.io.get_untrusted_types); los modelos los genera el propio proyecto.
TIPOS_SKOPS_CONFIABLES = [
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "numpy.dtype",
]
# Familia de cada modelo para la etiqueta model_family; si falta, se usa el nombre.
FAMILIAS_MODELO = {
    "BaselinePromedioSemanal": "baseline",
    "BaselineSemanaAnterior": "baseline",
    "HistGradientBoostingRegressor": "gradient_boosting",
    "HistGradientBoostingClassifier": "gradient_boosting",
    "PoissonRegressor": "linear",
    "LogisticRegression": "linear",
    "KMeans": "clustering",
}


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
    pregunta: str,
    nombre: str,
    modelo: str,
    parametros: dict | None = None,
    etapa: str = "candidate",
) -> Iterator[mlflow.ActiveRun]:
    """Abre un run con nombre descriptivo, parámetros comunes y etiquetas del equipo.

    Si ya hay un run activo (el padre de una búsqueda), el nuevo run se anida bajo
    él. Si no, se anida bajo el run que creó ``mlflow run`` (``MLFLOW_RUN_ID``).

    Args:
        pregunta: "P1", "P2" o "P3".
        nombre: nombre descriptivo del run, por ejemplo "p2-baseline-semana-anterior".
        modelo: nombre corto del modelo, por ejemplo "LogisticRegression".
        parametros: hiperparámetros y decisiones propias de la pregunta.
        etapa: "baseline", "search" o "candidate" (etiqueta stage_of_work).
    """
    if pregunta not in PREGUNTAS:
        raise ValueError(f"Pregunta inválida: {pregunta}. Usen P1, P2 o P3.")
    if etapa not in ETAPAS:
        raise ValueError(f"Etapa inválida: {etapa}. Usen {', '.join(sorted(ETAPAS))}.")

    configurar_mlflow()
    commit, sucio = obtener_commit()
    etiquetas = {
        "team": f"equipo{TEAM_NUMBER:02d}",
        "question": pregunta,
        "model": modelo,
        "model_family": FAMILIAS_MODELO.get(modelo, modelo),
        "stage_of_work": etapa,
        "git_commit": commit,
        "git_dirty": str(sucio).lower(),
        "mlflow.source.git.commit": commit,
    }
    # Conservar el identificador del proyecto durante toda la ejecución.
    project_run_id = os.environ.get("MLFLOW_PROJECT_PARENT_RUN_ID")
    if not project_run_id:
        project_run_id = os.environ.pop("MLFLOW_RUN_ID", None)
        if project_run_id:
            os.environ["MLFLOW_PROJECT_PARENT_RUN_ID"] = project_run_id
            # El run de `mlflow run` también se etiqueta para poder filtrarlo.
            cliente = mlflow.MlflowClient()
            for clave, valor in {
                "team": etiquetas["team"],
                "question": pregunta,
                "stage_of_work": "project",
            }.items():
                cliente.set_tag(project_run_id, clave, valor)
    # Un run activo (padre de búsqueda) tiene prioridad sobre el run del proyecto.
    activo = mlflow.active_run()
    parent_run_id = None if activo else project_run_id

    with mlflow.start_run(
        run_name=nombre,
        tags=etiquetas,
        nested=bool(activo or parent_run_id),
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


@contextmanager
def medir_tiempo(metrica: str) -> Iterator[None]:
    """Registra la duración del bloque en segundos, por ejemplo ``fit_time_s``."""
    inicio = time.perf_counter()
    yield
    mlflow.log_metric(metrica, time.perf_counter() - inicio)


def registrar_candidato(
    modelo,
    x_train: pd.DataFrame,
    hiperparametros: dict,
    *,
    nombre_datos: str,
    y_train: pd.Series | None = None,
    extra_config: dict | None = None,
) -> None:
    """Registra los artefactos que respaldan una corrida candidata.

    - ``model``: modelo sklearn con firma y ejemplo de entrada (5 filas, sin
      columnas de usuario).
    - ``data/``: manifiestos de la ingesta y ``log_input`` del entrenamiento.
    - ``config/``: hiperparámetros efectivos, parámetros comunes y ``config.py``.
    """
    # Columnas como float64: en inferencia pueden llegar con valores faltantes.
    ejemplo = x_train.head(5).astype("float64")
    # KMeans se ajusta con la matriz sin nombres de columna; se predice igual.
    entrada = ejemplo if hasattr(modelo, "feature_names_in_") else ejemplo.to_numpy()
    firma = infer_signature(ejemplo, modelo.predict(entrada))
    mlflow.sklearn.log_model(
        modelo,
        name="model",
        signature=firma,
        input_example=ejemplo,
        skops_trusted_types=TIPOS_SKOPS_CONFIABLES,
    )

    for manifiesto in (DOWNLOADS_MANIFEST, PROCESSED_MANIFEST):
        mlflow.log_artifact(str(MANIFESTS_DIR / manifiesto), artifact_path="data")
    entrenamiento = x_train if y_train is None else x_train.assign(**{y_train.name: y_train})
    dataset = mlflow.data.from_pandas(
        entrenamiento,
        name=nombre_datos,
        targets=None if y_train is None else y_train.name,
    )
    mlflow.log_input(dataset, context="training")

    mlflow.log_dict(
        {
            "hiperparametros": hiperparametros,
            "parametros_comunes": parametros_comunes(),
            "variables": list(x_train.columns),
            **(extra_config or {}),
        },
        "config/run_config.json",
    )
    mlflow.log_artifact(str(PROJ_ROOT / "mibici" / "config.py"), artifact_path="config")
