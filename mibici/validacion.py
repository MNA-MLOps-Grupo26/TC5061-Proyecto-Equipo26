"""Exporta los datos que valida DeepChecks.

Corre en el entorno del proyecto (``python -m mibici.validacion``). Escribe en
``data/interim/deepchecks/``:

- ``viajes_{train,validation,test}.parquet``: muestra de viajes limpios por
  partición, con duración, hora y día de la semana.
- ``p1_{train,validation,test}.parquet``: panel estación-día de P1 completo.
- ``esquema.json``: variables, objetivo y columnas categóricas, para que el
  entorno de DeepChecks no tenga que importar ``mibici``.

DeepChecks corre en un entorno aparte (ver ``validacion_datos/``).
"""

import json

from loguru import logger
import pandas as pd

from mibici.config import INTERIM_DATA_DIR, RANDOM_SEED
from mibici.dataset import cargar_estaciones, cargar_viajes
from mibici.features import VARIABLES_P1, dataset_p1, particionar_dataframe

DESTINO = INTERIM_DATA_DIR / "deepchecks"
MUESTRA_VIAJES = 300_000  # por partición; DeepChecks no necesita los 5 M de viajes
CONJUNTOS = ("train", "validation", "test")


def preparar_viajes(viajes: pd.DataFrame) -> pd.DataFrame:
    """Agrega duración, hora y día de la semana a los viajes limpios."""
    resultado = viajes.copy()
    resultado["duracion_min"] = (
        resultado["fin_viaje"] - resultado["inicio_viaje"]
    ).dt.total_seconds() / 60
    resultado["hora"] = resultado["inicio_viaje"].dt.hour
    resultado["dia_semana"] = resultado["inicio_viaje"].dt.dayofweek
    return resultado


def esquema() -> dict:
    return {
        "p1": {
            "features": VARIABLES_P1,
            "label": "salidas",
            "cat_features": ["dia_semana", "es_fin_semana"],
            "datetime": "fecha",
            "llaves": ["estacion_id", "fecha"],
        },
        "viajes": {
            "features": ["origen_id", "destino_id", "duracion_min", "hora", "dia_semana"],
            "cat_features": ["origen_id", "destino_id", "dia_semana"],
            "datetime": "inicio_viaje",
            "llaves": ["viaje_id"],
        },
    }


def exportar(destino=DESTINO, muestra=MUESTRA_VIAJES) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    viajes = cargar_viajes()
    estaciones = cargar_estaciones()

    for nombre, parte in particionar_dataframe(preparar_viajes(viajes)).items():
        if len(parte) > muestra:
            parte = parte.sample(muestra, random_state=RANDOM_SEED)
        parte.to_parquet(destino / f"viajes_{nombre}.parquet", index=False)
        logger.info(f"viajes_{nombre}: {len(parte):,} filas")

    panel = dataset_p1(viajes, estaciones)
    columnas = ["estacion_id", "fecha", *VARIABLES_P1, "salidas"]
    for nombre in CONJUNTOS:
        parte = panel.loc[panel["conjunto"] == nombre, columnas]
        parte.to_parquet(destino / f"p1_{nombre}.parquet", index=False)
        logger.info(f"p1_{nombre}: {len(parte):,} filas")

    (destino / "esquema.json").write_text(json.dumps(esquema(), indent=2), encoding="utf-8")
    logger.success(f"Datos para DeepChecks en {destino}")


if __name__ == "__main__":
    exportar()
