"""Chequeos de DeepChecks para los viajes y el panel de P1.

No importa ``mibici``: corre en el entorno aislado de ``requirements.txt`` y
recibe las columnas desde ``esquema.json``.

Cada chequeo es bloqueante o informativo:

- Bloqueante: si falla, los datos no deben usarse para entrenar (duplicados,
  nulos donde no puede haberlos, tipos mezclados, fechas de prueba dentro de
  entrenamiento).
- Informativo: se reporta para revisarlo, pero no detiene el pipeline. El drift
  entre entrenamiento (ago 2025 - jul 2026) y prueba (sep 2026) es esperable por
  estacionalidad; lo que importa es documentarlo.
"""

from dataclasses import dataclass, field
import warnings

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import (
    DataDuplicates,
    DateTrainTestLeakageOverlap,
    FeatureDrift,
    FeatureLabelCorrelation,
    IsSingleValue,
    LabelDrift,
    MixedDataTypes,
    MixedNulls,
    NewCategoryTrainTest,
    PercentOfNulls,
)
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# Variables que cambian con el calendario por construcción: train y test nunca
# comparten mes ni antigüedad de la estación. Se excluyen del drift para que el
# reporte muestre cambios de comportamiento, no del calendario.
DRIFT_ESTRUCTURAL = ["mes", "dia_mes", "dias_historia"]
UMBRAL_DRIFT = 0.2
UMBRAL_PPS = 0.9  # PPS mayor sugiere fuga del objetivo hacia una variable
UMBRAL_NULOS_REZAGOS = 0.10  # rezagos y medias móviles pueden faltar al inicio
# Sin nulos permitidos: objetivo y calendario siempre se conocen. El resto
# (rezagos, catálogo de estaciones) puede faltar y se reporta como aviso.
OBLIGATORIAS_P1 = ["dia_semana", "es_fin_semana", "mes", "dia_mes", "dias_historia"]


@dataclass
class Resultado:
    chequeo: str
    datos: str
    bloqueante: bool
    paso: bool
    detalle: str
    resultado: object = field(default=None, repr=False)


def _evaluar(check, datos, bloqueante, *datasets) -> Resultado:
    resultado = check.run(*datasets)
    condiciones = resultado.conditions_results
    paso = all(c.is_pass() for c in condiciones)
    detalle = " | ".join(" ".join(c.details.split()) for c in condiciones if c.details)
    return Resultado(resultado.get_header(), datos, bloqueante, paso, detalle, resultado)


def dataset_panel(df: pd.DataFrame, esquema: dict) -> Dataset:
    p = esquema["p1"]
    return Dataset(
        df,
        label=p["label"],
        features=p["features"],
        cat_features=p["cat_features"],
        datetime_name=p["datetime"],
        label_type="regression",
    )


def dataset_viajes(df: pd.DataFrame, esquema: dict) -> Dataset:
    v = esquema["viajes"]
    return Dataset(
        df,
        features=v["features"],
        cat_features=v["cat_features"],
        datetime_name=v["datetime"],
    )


def integridad_panel(df: pd.DataFrame, esquema: dict, nombre: str) -> list[Resultado]:
    p = esquema["p1"]
    ds = dataset_panel(df, esquema)
    sin_nulos = [c for c in p["features"] if c in OBLIGATORIAS_P1] + [p["label"]]
    rezagos = [c for c in p["features"] if c not in sin_nulos]
    llaves = Dataset(df[p["llaves"]], cat_features=[])
    # DeepChecks no revisa nulos en el objetivo; se arma un Dataset sin label.
    obligatorias = Dataset(df[sin_nulos], cat_features=[])
    return [
        _evaluar(
            DataDuplicates().add_condition_ratio_less_or_equal(0),
            f"{nombre} (estacion_id, fecha)",
            True,
            llaves,
        ),
        _evaluar(
            PercentOfNulls().add_condition_percent_of_nulls_not_greater_than(0),
            f"{nombre} objetivo y calendario",
            True,
            obligatorias,
        ),
        _evaluar(
            PercentOfNulls(columns=rezagos).add_condition_percent_of_nulls_not_greater_than(
                UMBRAL_NULOS_REZAGOS
            ),
            f"{nombre} rezagos y catálogo",
            False,
            ds,
        ),
        _evaluar(MixedNulls().add_condition_different_nulls_less_equal_to(1), nombre, True, ds),
        _evaluar(MixedDataTypes().add_condition_rare_type_ratio_not_in_range(), nombre, True, ds),
        _evaluar(
            IsSingleValue(ignore_columns=DRIFT_ESTRUCTURAL).add_condition_not_single_value(),
            nombre,
            False,
            ds,
        ),
        _evaluar(
            FeatureLabelCorrelation().add_condition_feature_pps_less_than(UMBRAL_PPS),
            nombre,
            False,
            ds,
        ),
    ]


def integridad_viajes(df: pd.DataFrame, esquema: dict, nombre: str) -> list[Resultado]:
    v = esquema["viajes"]
    ds = dataset_viajes(df, esquema)
    llaves = Dataset(df[v["llaves"]], cat_features=[])
    return [
        _evaluar(
            DataDuplicates().add_condition_ratio_less_or_equal(0),
            f"{nombre} (viaje_id)",
            True,
            llaves,
        ),
        _evaluar(
            PercentOfNulls().add_condition_percent_of_nulls_not_greater_than(0), nombre, True, ds
        ),
        _evaluar(MixedDataTypes().add_condition_rare_type_ratio_not_in_range(), nombre, True, ds),
    ]


def train_test_panel(
    train: pd.DataFrame, prueba: pd.DataFrame, esquema: dict, nombre: str
) -> list[Resultado]:
    ds_train, ds_prueba = dataset_panel(train, esquema), dataset_panel(prueba, esquema)
    return [
        _evaluar(
            DateTrainTestLeakageOverlap().add_condition_leakage_ratio_less_or_equal(0),
            nombre,
            True,
            ds_train,
            ds_prueba,
        ),
        _evaluar(
            FeatureDrift(ignore_columns=DRIFT_ESTRUCTURAL).add_condition_drift_score_less_than(
                UMBRAL_DRIFT, UMBRAL_DRIFT
            ),
            nombre,
            False,
            ds_train,
            ds_prueba,
        ),
        _evaluar(
            LabelDrift().add_condition_drift_score_less_than(UMBRAL_DRIFT),
            nombre,
            False,
            ds_train,
            ds_prueba,
        ),
    ]


def train_test_viajes(
    train: pd.DataFrame, prueba: pd.DataFrame, esquema: dict, nombre: str
) -> list[Resultado]:
    ds_train, ds_prueba = dataset_viajes(train, esquema), dataset_viajes(prueba, esquema)
    return [
        _evaluar(
            DateTrainTestLeakageOverlap().add_condition_leakage_ratio_less_or_equal(0),
            nombre,
            True,
            ds_train,
            ds_prueba,
        ),
        _evaluar(
            FeatureDrift().add_condition_drift_score_less_than(UMBRAL_DRIFT, UMBRAL_DRIFT),
            nombre,
            False,
            ds_train,
            ds_prueba,
        ),
        # Estaciones de origen que aparecen en prueba y no existían en entrenamiento.
        _evaluar(
            NewCategoryTrainTest(columns=["origen_id"]).add_condition_new_category_ratio_less_or_equal(
                0.05
            ),
            nombre,
            False,
            ds_train,
            ds_prueba,
        ),
    ]


def bloqueantes_fallidos(resultados: list[Resultado]) -> list[Resultado]:
    return [r for r in resultados if r.bloqueante and not r.paso]
