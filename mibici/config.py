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
# Catálogo de estaciones (id, nombre, coordenadas y capacidad). Se usa la
# versión más reciente publicada; aplicar su capacidad a meses previos es un
# supuesto documentado en el ML Canvas.
NOMENCLATURA_MES = "2026_09"
NOMENCLATURA_FILENAME = f"nomenclatura_{NOMENCLATURA_MES}.csv"

# Archivos procesados y manifiestos
VIAJES_PARQUET_GLOB = "datos_abiertos_*.parquet"
ESTACIONES_PARQUET = "estaciones.parquet"
DOWNLOADS_MANIFEST = "mibici_downloads.json"
PROCESSED_MANIFEST = "mibici_processed.json"

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

# Variables de P1 y P2 (panel estación-día)
# Rezagos y ventanas en días previos al día pronosticado.
VENTANAS_MEDIA = (7, 28)
SEMANAS_BASELINE_P1 = 4  # promedio del mismo día de la semana en 4 semanas previas
# P1: hiperparámetros predeterminados del candidato
P1_HIPERPARAMETROS = {
    "loss": "poisson",
    "learning_rate": 0.05,
    "max_iter": 300,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 50,
    "early_stopping": False,
}
P2_HIPERPARAMETROS = {
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 2000,
    "solver": "lbfgs",
}




# P2: indicador de riesgo aproximado
# Día operativo de 05:00 a 05:00 del día siguiente (el sistema reinicia a las 05:00).
P2_HORA_INICIO_DIA = 5
# Ocupación inicial supuesta. Con 0.5, el umbral de "sin bicicletas" es +0.5 y
# el de "sin anclajes" es -0.5 (saldo neto acumulado / capacidad).
P2_OCUPACION_INICIAL = 0.5
P2_CLASES = ("sin_riesgo", "sin_bicis", "sin_anclajes")
P2_TOP_K = 30  # estaciones que el equipo de rebalanceo podría revisar por día

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

# P3: preparacion provisional de perfiles; no presupone disponibilidad diaria.
P3_MIN_DIAS_ACTIVOS = 28
P3_MIN_MOVIMIENTOS = 100
# P3: KMeans fijado en references/p3_frozen.json y umbrales de evaluación mensual.
P3_K = 3
P3_K_EXPLORACION = (2, 3, 4, 5, 6)  # solo para la comparación exploratoria
P3_N_INIT = 20
P3_MAX_ITER = 300
P3_MIN_DIAS_TRAIN = 180  # cohorte con historia suficiente en entrenamiento
P3_COHORTES = (180, 300)  # umbrales de historia predeclarados para la sensibilidad
P3_MES_MIN_DIAS = 14  # días con eventos para evaluar una estación en un mes
P3_MES_MIN_MOVIMIENTOS = 100
# P3: evidencias históricas de la selección y de la evaluación congelada (no se
# versionan; viven en la máquina donde se generaron). Solo se leen o se crean nuevas.
P3_PREPARACION_PARQUET = REPORTS_DIR / "p3-preparacion-20261018/perfiles_entrenamiento.parquet"
P3_VALIDACION_DIR = REPORTS_DIR / "p3-validacion-agosto-tentativa-20261018"
P3_SENSIBILIDAD_DIR = REPORTS_DIR / "p3-sensibilidad-historia-20261018"
P3_EVALUACION_CONGELADA_DIR = REPORTS_DIR / "p3-evaluacion-congelada-20261018"
P3_REPRODUCCION_DIR = REPORTS_DIR / "p3-reproduccion-limpia-20261018"
