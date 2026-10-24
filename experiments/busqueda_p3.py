"""Busqueda exploratoria P3: tres candidatos KMeans con runs anidados."""

import json
import os

from experiments.evaluacion_p3 import evaluar_validacion
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from mibici.config import (
    MANIFESTS_DIR,
    P3_MAX_ITER,
    P3_MIN_DIAS_TRAIN,
    P3_N_INIT,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TEST_START,
    TRAIN_START,
    VALIDATION_START,
)
from mibici.dataset import leer_catalogo_verificado
from mibici.features import cohorte_historia, matriz_forma, perfiles_agosto, preparar_perfiles_p3
from mibici.modeling.p3 import cargar_periodo, llegadas_desde


def main():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment("mibici-equipo26")

    manifest = json.loads((MANIFESTS_DIR / "mibici_processed.json").read_text(encoding="utf-8"))
    records = {r["filename"]: r for r in manifest["files"]}

    estaciones = leer_catalogo_verificado(
        PROCESSED_DATA_DIR / manifest["stations"]["filename"],
        manifest["stations"],
    )

    viajes = cargar_periodo(records, TRAIN_START, VALIDATION_START)
    perfiles = preparar_perfiles_p3(viajes, estaciones)
    train = cohorte_historia(perfiles, P3_MIN_DIAS_TRAIN)
    X = matriz_forma(train)

    agosto = cargar_periodo(records, VALIDATION_START, TEST_START)
    viajes_validacion = pd.concat(
        [llegadas_desde(viajes, VALIDATION_START), agosto],
        ignore_index=True,
    )
    val = perfiles_agosto(viajes_validacion, train)

    print(f"Cohorte: {len(train)} estaciones, {X.shape[1]} variables")

    with mlflow.start_run(run_name="P3-busqueda-kmeans") as padre:
        mlflow.set_tags(
            {
                "question": "P3",
                "search_type": "nested_hyperparameter_search",
                "model_family": "KMeans",
                "evaluation_role": "train_and_validation",
                "stage_of_work": "search",
            }
        )
        mlflow.log_param("candidatos_k", "2,4,5")
        mlflow.log_param("cohorte_min_dias_train", P3_MIN_DIAS_TRAIN)
        mlflow.log_param("random_seed", RANDOM_SEED)

        print("Run padre:", padre.info.run_id)

        for k in [2, 4, 5]:
            with mlflow.start_run(
                run_name=f"P3-kmeans-k{k}",
                nested=True,
            ):
                modelo = KMeans(
                    n_clusters=k,
                    random_state=RANDOM_SEED,
                    n_init=P3_N_INIT,
                    max_iter=P3_MAX_ITER,
                    algorithm="lloyd",
                )
                labels = modelo.fit_predict(X)

                silueta = float(silhouette_score(X, labels))
                tamanos = np.bincount(labels, minlength=k)
                metricas_val = evaluar_validacion(modelo, train, val)

                mlflow.set_tags(
                    {
                        "question": "P3",
                        "model_family": "KMeans",
                        "evaluation_role": "train_and_validation",
                        "stage_of_work": "search",
                    }
                )
                mlflow.log_params(
                    {
                        "k": k,
                        "n_init": P3_N_INIT,
                        "max_iter": P3_MAX_ITER,
                        "random_seed": RANDOM_SEED,
                        "cohorte_min_dias_train": P3_MIN_DIAS_TRAIN,
                        "n_variables": X.shape[1],
                        "transformacion": "sqrt(p)/sqrt(2)",
                    }
                )
                mlflow.log_metrics(
                    {
                        "train_silhouette": silueta,
                        "train_inertia": float(modelo.inertia_),
                        "train_n_estaciones": len(train),
                        "train_min_cluster_size": int(tamanos.min()),
                        "train_max_cluster_size": int(tamanos.max()),
                    }
                )

                mlflow.log_metrics(
                    {f"val_{nombre}": float(valor) for nombre, valor in metricas_val.items()}
                )
                mlflow.set_tag("validation_period", "2026-08")

                print(
                    f"k={k} | silhouette={silueta:.6f} | "
                    f"inertia={modelo.inertia_:.6f} | "
                    f"tamanos={tamanos.tolist()}"
                )


if __name__ == "__main__":
    main()
