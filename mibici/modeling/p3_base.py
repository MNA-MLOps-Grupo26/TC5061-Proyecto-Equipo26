"""P3 en un solo comando: perfiles, KMeans, mapa y run de MLflow.

Reproduce sin pasos manuales la configuración que el equipo fijó en
``references/p3_frozen.json``: KMeans con k=3 sobre las estaciones con al menos
180 días con eventos en entrenamiento. Los centroides se ajustan solo con
entrenamiento; agosto (validación) y septiembre (prueba) se asignan a esos
centroides fijos, sin reajustar.

Solo necesita la ingesta (``python -m mibici.dataset``). El protocolo de
auditoría de ``mibici.modeling.p3`` se conserva aparte para quien quiera
repetir la evaluación congelada con sus evidencias intermedias.
"""

import json
import shutil

import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from mibici.compare_p3 import N_INIT, SEED, guardar_grafica, matriz_forma
from mibici.config import (
    DATA_END_EXCLUSIVE,
    FIGURES_DIR,
    MANIFESTS_DIR,
    P3_K,
    P3_MAX_ITER,
    P3_MIN_DIAS_TRAIN,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    TEST_START,
    TRAIN_START,
    VALIDATION_START,
)
from mibici.features_p3 import preparar_perfiles_p3
from mibici.history_p3 import cohorte_historia
from mibici.modeling.p3 import guardar_mapa, resumir_fijo, sha256
from mibici.tracking import iniciar_run, registrar_artefacto, registrar_metricas
from mibici.validate_p3 import AUG_MIN_DAYS, AUG_MIN_MOVEMENTS, leer_verificado, perfiles_agosto

DESTINO = FIGURES_DIR / "p3"


def leer_catalogo_verificado(ruta, registro: dict) -> pd.DataFrame:
    """Lee el catálogo de estaciones solo si su hash y su conteo coinciden con el manifiesto.

    Las coordenadas del mapa salen de este archivo; un catálogo distinto al de la
    ingesta registrada cambiaría el mapa sin que la versión de datos lo reflejara.
    """
    if sha256(ruta) != registro["sha256"]:
        raise ValueError(
            f"SHA-256 de {ruta.name} no coincide con el manifiesto. "
            "Vuelvan a correr: python -m mibici.dataset"
        )
    estaciones = pd.read_parquet(ruta)
    if len(estaciones) != registro["records"]:
        raise ValueError(
            f"{ruta.name} tiene {len(estaciones)} registros; el manifiesto indica "
            f"{registro['records']}."
        )
    return estaciones


def cargar_periodo(records: dict, inicio, fin_exclusivo) -> pd.DataFrame:
    """Lee los Parquet mensuales del periodo y verifica hash y conteo contra el manifiesto."""
    meses = pd.period_range(inicio, pd.Timestamp(fin_exclusivo) - pd.Timedelta(days=1), freq="M")
    frames = [
        leer_verificado(f"datos_abiertos_{mes.year}_{mes.month:02d}.parquet", records)[0]
        for mes in meses
    ]
    return pd.concat(frames, ignore_index=True)


def llegadas_desde(viajes: pd.DataFrame, momento) -> pd.DataFrame:
    """Viajes del mes anterior que terminan dentro del periodo evaluado.

    Sin ellos se perderían las llegadas del primer día (viajes que empiezan el
    día 31 y terminan el día 1).
    """
    return viajes.loc[pd.to_datetime(viajes.fin_viaje).ge(pd.Timestamp(momento))]


def comparar_con_congelado(resultados: dict) -> bool:
    """Indica si las métricas coinciden con las registradas al congelar la configuración."""
    spec = json.loads((PROJ_ROOT / "references/p3_frozen.json").read_text(encoding="utf-8"))
    esperado = spec["expected_results"]
    return all(
        key in resultados[conjunto] and np.isclose(resultados[conjunto][key], valor, rtol=1e-9)
        for conjunto in ("train", "val", "test")
        for key, valor in esperado[conjunto].items()
    )


def ejecutar(viajes=None, estaciones=None) -> None:
    """Entrena y evalúa P3 y registra el run.

    Los argumentos se aceptan por compatibilidad con ``train.py`` y se ignoran:
    P3 lee los viajes por mes y el catálogo después de verificarlos contra el
    manifiesto.
    """
    manifest = json.loads((MANIFESTS_DIR / "mibici_processed.json").read_text(encoding="utf-8"))
    records = {r["filename"]: r for r in manifest["files"]}
    estaciones = leer_catalogo_verificado(
        PROCESSED_DATA_DIR / manifest["stations"]["filename"], manifest["stations"]
    )

    if DESTINO.exists():
        shutil.rmtree(DESTINO)
    DESTINO.mkdir(parents=True)

    # 1. Perfiles y ajuste: solo entrenamiento.
    viajes_train = cargar_periodo(records, TRAIN_START, VALIDATION_START)
    train = cohorte_historia(preparar_perfiles_p3(viajes_train, estaciones), P3_MIN_DIAS_TRAIN)
    modelo = KMeans(
        n_clusters=P3_K, random_state=SEED, n_init=N_INIT, max_iter=P3_MAX_ITER, algorithm="lloyd"
    ).fit(matriz_forma(train))
    labels = modelo.labels_

    # 2. Validación y prueba: asignación a centroides fijos, sin reajustar.
    agosto = cargar_periodo(records, VALIDATION_START, TEST_START)
    val = perfiles_agosto(
        pd.concat([llegadas_desde(viajes_train, VALIDATION_START), agosto], ignore_index=True),
        train,
    )
    septiembre = cargar_periodo(records, TEST_START, DATA_END_EXCLUSIVE)
    test = perfiles_agosto(
        pd.concat([llegadas_desde(agosto, TEST_START), septiembre], ignore_index=True),
        train,
        inicio=TEST_START,
        fin=DATA_END_EXCLUSIVE,
    )
    del viajes_train, agosto, septiembre

    _, asign_val, metricas_val, _ = resumir_fijo(modelo, train, val)
    q_test, asign_test, metricas_test, grupos_test = resumir_fijo(modelo, train, test)

    # 3. Evidencia: mapa, perfiles medios y asignaciones por estación (sin datos personales).
    mapeables, sin_ubicacion = guardar_mapa(train, labels, estaciones, DESTINO)
    guardar_grafica(train, labels, "train", DESTINO)
    guardar_grafica(
        q_test,
        asign_test.grupo_periodo.to_numpy(),
        "test_septiembre",
        DESTINO,
        contexto="septiembre: asignación a centroides de entrenamiento",
    )
    train[["estacion_id", "nombre", "dias_con_eventos", "total_movimientos"]].assign(
        grupo_train=labels
    ).to_csv(DESTINO / "asignaciones_train.csv", index=False)
    asign_val.to_csv(DESTINO / "asignaciones_val.csv", index=False)
    asign_test.to_csv(DESTINO / "asignaciones_test.csv", index=False)
    grupos_test.to_csv(DESTINO / "metricas_grupos_test.csv", index=False)

    metricas_train = {
        "n_estaciones": len(train),
        "silhouette": float(silhouette_score(matriz_forma(train), labels)),
        "n_mapeables": mapeables,
        "n_sin_ubicacion": sin_ubicacion,
    }
    resultados = {"train": metricas_train, "val": metricas_val, "test": metricas_test}
    (DESTINO / "resultados.json").write_text(json.dumps(resultados, indent=2), encoding="utf-8")
    coincide = comparar_con_congelado(resultados)

    params = {
        "k": P3_K,
        "cohorte_min_dias_activos_train": P3_MIN_DIAS_TRAIN,
        "n_init": N_INIT,
        "max_iter": P3_MAX_ITER,
        "algorithm": "lloyd",
        "n_variables": 336,
        "transformacion": "sqrt(p)/sqrt(2)",
        "distancia": "euclidean",
        "min_dias_mes": AUG_MIN_DAYS,
        "min_movimientos_mes": AUG_MIN_MOVEMENTS,
        "fit": "train_only",
        "catalogo_coordenadas": manifest["stations"]["source_filename"],
    }
    with iniciar_run("P3", "p3-kmeans-k3-historia180", "KMeans", params):
        mlflow.set_tags(
            {
                "interpretacion": "descriptiva_con_solapamiento",
                "reproduce_configuracion_congelada": str(coincide).lower(),
            }
        )
        registrar_metricas("train", metricas_train)
        registrar_metricas("val", metricas_val)
        registrar_metricas("test", metricas_test)
        for ruta in sorted(DESTINO.iterdir()):
            registrar_artefacto(ruta, "p3")
