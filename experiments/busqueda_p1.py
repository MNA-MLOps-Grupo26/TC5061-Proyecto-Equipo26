"""Búsqueda organizada de hiperparámetros para P1 con nested runs."""

import os
import subprocess

import mlflow
from sklearn.ensemble import HistGradientBoostingRegressor

from mibici.config import P1_HIPERPARAMETROS, RANDOM_SEED
from mibici.dataset import cargar_estaciones, cargar_viajes
from mibici.features import VARIABLES_P1, dataset_p1
from mibici.modeling.p1 import calcular_metricas

EXPERIMENTO = "mibici-equipo26"

COMBINACIONES = [
    {"learning_rate": 0.03, "max_iter": 400},
    {"learning_rate": 0.10, "max_iter": 200},
    {"learning_rate": 0.07, "max_iter": 350},
]


def obtener_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(EXPERIMENTO)

    datos = dataset_p1(cargar_viajes(), cargar_estaciones())
    train = datos.loc[datos["conjunto"] == "train"]
    val = datos.loc[datos["conjunto"] == "validation"]

    if train.empty or val.empty:
        raise ValueError("Las particiones de entrenamiento o validación están vacías.")

    etiquetas = {
        "team": "equipo26",
        "question": "P1",
        "model_family": "HistGradientBoostingRegressor",
        "stage_of_work": "search",
        "git_commit": obtener_commit(),
    }

    resultados = []

    with mlflow.start_run(
        run_name="p1-busqueda-hiperparametros",
        tags=etiquetas,
    ):
        mlflow.log_param("search_method", "manual_grid")
        mlflow.log_param("n_combinations", len(COMBINACIONES))
        mlflow.log_param("selection_metric", "val_mae")

        for indice, combinacion in enumerate(COMBINACIONES, start=1):
            parametros = {**P1_HIPERPARAMETROS, **combinacion}

            with mlflow.start_run(
                run_name=f"p1-search-{indice}",
                nested=True,
                tags=etiquetas,
            ):
                mlflow.log_params(parametros)
                mlflow.log_param("n_variables", len(VARIABLES_P1))
                mlflow.log_param("fit", "train_only")

                modelo = HistGradientBoostingRegressor(
                    random_state=RANDOM_SEED,
                    **parametros,
                )
                modelo.fit(train[VARIABLES_P1], train["salidas"])

                for nombre, conjunto in [("train", train), ("val", val)]:
                    prediccion = modelo.predict(conjunto[VARIABLES_P1])
                    metricas = calcular_metricas(conjunto, prediccion)
                    mlflow.log_metrics({f"{nombre}_{k}": v for k, v in metricas.items()})

                    if nombre == "val":
                        resultados.append(
                            {
                                **combinacion,
                                **metricas,
                            }
                        )

        mejor = min(resultados, key=lambda r: r["mae"])
        mlflow.log_metric("best_val_mae", mejor["mae"])
        mlflow.log_param("best_learning_rate", mejor["learning_rate"])
        mlflow.log_param("best_max_iter", mejor["max_iter"])

    print("\nBúsqueda finalizada.")
    print("Mejor combinación:", mejor)


if __name__ == "__main__":
    main()
