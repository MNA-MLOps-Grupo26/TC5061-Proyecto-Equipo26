"""Configuración compartida del proyecto MiBici (equipo 26).

Rutas, periodo de datos, partición temporal, semilla y nombres de MLflow viven
solo aquí. Los demás módulos los importan desde este archivo; no repitan rutas,
fechas ni semillas en otros lugares.
"""

from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

REFERENCES_DIR = PROJ_ROOT / "references"
MANIFESTS_DIR = PROJ_ROOT / "manifests"

# Fuente de datos
MIBICI_PORTAL = "https://mibici.net/es/datos-abiertos/"
LOCAL_TIMEZONE = "America/Mexico_City"

# Periodo y partición temporal.
# Intervalos cerrados por la izquierda y abiertos por la derecha:
# entrenamiento [TRAIN_START, VALIDATION_START), validación
# [VALIDATION_START, TEST_START) y prueba [TEST_START, DATA_END_EXCLUSIVE).
TRAIN_START = date(2025, 8, 1)
VALIDATION_START = date(2026, 8, 1)
TEST_START = date(2026, 9, 1)
DATA_END_EXCLUSIVE = date(2026, 10, 1)
DATA_START_MONTH = TRAIN_START.strftime("%Y-%m")
DATA_END_MONTH = (DATA_END_EXCLUSIVE - timedelta(days=1)).strftime("%Y-%m")

# Reproducibilidad
RANDOM_SEED = 42

# MLflow
TEAM_NUMBER = 26
EXPERIMENT_NAME = f"mibici-equipo{TEAM_NUMBER:02d}"
MLFLOW_DB_PATH = PROJ_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
MLFLOW_ARTIFACTS_DIR = PROJ_ROOT / "mlartifacts"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
