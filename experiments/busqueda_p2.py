"""Búsqueda de hiperparámetros para P2 con runs anidados en MLflow."""

import os
import subprocess

import mlflow
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mibici.config import (
    DATA_END_EXCLUSIVE,
    P2_HIPERPARAMETROS,
    P2_HORA_INICIO_DIA,
    RANDOM_SEED,
)
from mibici.dataset import cargar_estaciones, cargar_viajes
from mibici.features import VARIABLES_P2, dataset_p2
from mibici.modeling.p2 import SIN_RIESGO, calcular_metricas

EXPERIMENTO = "mibici-equipo26"

COMBINACIONES = [
    {"C": 0.1},
    {"C": 0.5},
    {"C": 5.0},
]


def obtener_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(EXPERIMENTO)

    datos = dataset_p2(cargar_viajes(), cargar_estaciones())
    fin_dia = datos["fecha"] + pd.Timedelta(days=1, hours=P2_HORA_INICIO_DIA)
    datos = datos.loc[fin_dia <= pd.Timestamp(DATA_END_EXCLUSIVE)].copy()

    train = datos.loc[datos["conjunto"] == "train"]
    val = datos.loc[datos["conjunto"] == "validation"]

    if train.empty or val.empty:
        raise ValueError("Entrenamiento o validación están vacíos.")

    etiquetas = {
        "team": "equipo26",
        "question": "P2",
        "model_family": "LogisticRegression",
        "stage_of_work": "search",
        "git_commit": obtener_commit(),
    }

    resultados = []

    with mlflow.start_run(
        run_name="p2-busqueda-hiperparametros",
        tags=etiquetas,
    ):
        mlflow.log_param("search_method", "manual_grid")
        mlflow.log_param("n_combinations", len(COMBINACIONES))
        mlflow.log_param("selection_metric", "val_f1_macro")

        for indice, combinacion in enumerate(COMBINACIONES, start=1):
            parametros = {**P2_HIPERPARAMETROS, **combinacion}

            with mlflow.start_run(
                run_name=f"p2-search-{indice}",
                nested=True,
                tags=etiquetas,
            ):
                mlflow.log_params(parametros)
                mlflow.log_param("n_variables", len(VARIABLES_P2))
                mlflow.log_param("fit", "train_only")

                modelo = make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    LogisticRegression(
                        random_state=RANDOM_SEED,
                        **parametros,
                    ),
                )

                modelo.fit(
                    train[VARIABLES_P2],
                    train["categoria"],
                )

                clases = list(modelo.classes_)

                for nombre, conjunto in [
                    ("train", train),
                    ("val", val),
                ]:
                    probabilidades = modelo.predict_proba(conjunto[VARIABLES_P2])

                    prediccion = pd.Series(
                        modelo.classes_[probabilidades.argmax(axis=1)],
                        index=conjunto.index,
                    )

                    puntaje = pd.Series(
                        1 - probabilidades[:, clases.index(SIN_RIESGO)],
                        index=conjunto.index,
                    )

                    metricas = calcular_metricas(
                        conjunto,
                        prediccion,
                        puntaje,
                    )

                    mlflow.log_metrics(
                        {f"{nombre}_{clave}": valor for clave, valor in metricas.items()}
                    )

                    if nombre == "val":
                        resultados.append(
                            {
                                **combinacion,
                                **metricas,
                            }
                        )

        mejor = max(
            resultados,
            key=lambda resultado: resultado["f1_macro"],
        )

        mlflow.log_metric(
            "best_val_f1_macro",
            mejor["f1_macro"],
        )
        mlflow.log_param("best_C", mejor["C"])

    print("\nBúsqueda P2 finalizada.")
    print("Mejor combinación:", mejor)


if __name__ == "__main__":
    main()
