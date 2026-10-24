"""P3 en un solo comando: perfiles, KMeans, mapa y run de MLflow.

Reproduce sin pasos manuales la configuración que el equipo fijó en
``references/p3_frozen.json``: KMeans con k=3 sobre las estaciones con al menos
180 días con eventos en entrenamiento. Los centroides se ajustan solo con
entrenamiento; agosto (validación) y septiembre (prueba) se asignan a esos
centroides fijos, sin reajustar.

Solo necesita la ingesta (``python -m mibici.dataset``). La selección de esta
configuración se documenta en ``notebooks/2.0-fvep-p3-seleccion.ipynb``.

Este módulo también conserva la auditoría de la evaluación congelada
(``congelar`` y ``evaluar_congelado``), documentada en
``notebooks/3.0-fvep-p3-auditoria.ipynb`` y ejecutable con
``python -m mibici.modeling.p3 [--solo-congelar | --reproducir]``.
"""

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from mibici.config import (
    BUSQUEDAS,
    DATA_END_EXCLUSIVE,
    FIGURES_DIR,
    MANIFESTS_DIR,
    P3_EVALUACION_CONGELADA_DIR,
    P3_HIPERPARAMETROS,
    P3_MES_MIN_DIAS,
    P3_MES_MIN_MOVIMIENTOS,
    P3_MIN_DIAS_TRAIN,
    P3_N_INIT,
    P3_PREPARACION_PARQUET,
    P3_REPRODUCCION_DIR,
    P3_SENSIBILIDAD_DIR,
    P3_VALIDACION_DIR,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    RANDOM_SEED,
    TEST_START,
    TRAIN_START,
    VALIDATION_START,
)
from mibici.dataset import leer_catalogo_verificado, leer_verificado, sha256
from mibici.evaluation import comparar_con_congelado, resumir_fijo, verificar_reproduccion
from mibici.features import (
    cohorte_historia,
    columnas_perfil_p3,
    matriz_forma,
    perfiles_agosto,
    preparar_perfiles_p3,
)
from mibici.plots import guardar_grafica, guardar_mapa
from mibici.tracking import (
    iniciar_run,
    medir_tiempo,
    obtener_commit,
    registrar_artefacto,
    registrar_candidato,
    registrar_metricas,
)

DESTINO = FIGURES_DIR / "p3"


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


def preparar() -> dict:
    """Perfiles de entrenamiento, validación y prueba; se calculan una sola vez.

    La cohorte (estaciones con al menos ``P3_MIN_DIAS_TRAIN`` días con eventos)
    es la misma para todos los valores de k, así que la búsqueda compara solo k.
    """
    manifest = json.loads((MANIFESTS_DIR / "mibici_processed.json").read_text(encoding="utf-8"))
    records = {r["filename"]: r for r in manifest["files"]}
    estaciones = leer_catalogo_verificado(
        PROCESSED_DATA_DIR / manifest["stations"]["filename"], manifest["stations"]
    )
    viajes_train = cargar_periodo(records, TRAIN_START, VALIDATION_START)
    train = cohorte_historia(preparar_perfiles_p3(viajes_train, estaciones), P3_MIN_DIAS_TRAIN)
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
    return {
        "manifest": manifest,
        "estaciones": estaciones,
        "train": train,
        "val": val,
        "test": test,
    }


def run_kmeans(prep: dict, hiper: dict | None = None, etapa: str | None = None) -> None:
    """Ajusta KMeans con entrenamiento, asigna validación y prueba sin reajustar y registra.

    Sin cambios de hiperparámetros reproduce la configuración congelada
    (``references/p3_frozen.json``), que es el baseline de P3.
    """
    hiper = {**P3_HIPERPARAMETROS, **(hiper or {})}
    es_congelada = hiper == P3_HIPERPARAMETROS
    etapa = etapa or ("baseline" if es_congelada else "candidate")
    k = hiper["n_clusters"]
    nombre = f"p3-kmeans-k{k}-historia{P3_MIN_DIAS_TRAIN}"
    train, estaciones = prep["train"], prep["estaciones"]

    destino = DESTINO / nombre
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)

    params = {
        "k": k,
        "cohorte_min_dias_activos_train": P3_MIN_DIAS_TRAIN,
        "n_init": hiper["n_init"],
        "max_iter": hiper["max_iter"],
        "algorithm": "lloyd",
        "n_variables": 336,
        "transformacion": "sqrt(p)/sqrt(2)",
        "distancia": "euclidean",
        "min_dias_mes": P3_MES_MIN_DIAS,
        "min_movimientos_mes": P3_MES_MIN_MOVIMIENTOS,
        "fit": "train_only",
        "catalogo_coordenadas": prep["manifest"]["stations"]["source_filename"],
        "selection_metric": "val_ari_vs_train",
    }
    with iniciar_run("P3", nombre, "KMeans", params, etapa=etapa):
        x_train = matriz_forma(train)
        with medir_tiempo("fit_time_s"):
            modelo = KMeans(random_state=RANDOM_SEED, algorithm="lloyd", **hiper).fit(x_train)
        labels = modelo.labels_

        _, asign_val, metricas_val, _ = resumir_fijo(modelo, train, prep["val"])
        q_test, asign_test, metricas_test, grupos_test = resumir_fijo(modelo, train, prep["test"])

        # Evidencia: mapa, perfiles medios y asignaciones por estación (sin datos personales).
        mapeables, sin_ubicacion = guardar_mapa(train, labels, estaciones, destino)
        guardar_grafica(train, labels, "train", destino)
        guardar_grafica(
            q_test,
            asign_test.grupo_periodo.to_numpy(),
            "test_septiembre",
            destino,
            contexto="septiembre: asignación a centroides de entrenamiento",
        )
        train[["estacion_id", "nombre", "dias_con_eventos", "total_movimientos"]].assign(
            grupo_train=labels
        ).to_csv(destino / "asignaciones_train.csv", index=False)
        asign_val.to_csv(destino / "asignaciones_val.csv", index=False)
        asign_test.to_csv(destino / "asignaciones_test.csv", index=False)
        grupos_test.to_csv(destino / "metricas_grupos_test.csv", index=False)

        tamanos = np.bincount(labels, minlength=k)
        metricas_train = {
            "n_estaciones": len(train),
            "silhouette": float(silhouette_score(x_train, labels)),
            "inertia": float(modelo.inertia_),
            "min_cluster_size": int(tamanos.min()),
            "n_mapeables": mapeables,
            "n_sin_ubicacion": sin_ubicacion,
        }
        resultados = {"train": metricas_train, "val": metricas_val, "test": metricas_test}
        (destino / "resultados.json").write_text(
            json.dumps(resultados, indent=2), encoding="utf-8"
        )
        etiquetas = {"interpretacion": "descriptiva_con_solapamiento"}
        if es_congelada:
            congelado = {
                c: {m: v for m, v in r.items() if m not in ("inertia", "min_cluster_size")}
                for c, r in resultados.items()
            }
            etiquetas["reproduce_configuracion_congelada"] = str(
                comparar_con_congelado(congelado)
            ).lower()
        mlflow.set_tags(etiquetas)

        registrar_metricas("train", metricas_train)
        registrar_metricas("val", metricas_val)
        registrar_metricas("test", metricas_test)
        for ruta in sorted(destino.iterdir()):
            registrar_artefacto(ruta, "plots" if ruta.suffix == ".png" else "tables")
        registrar_candidato(
            modelo,
            pd.DataFrame(x_train, columns=columnas_perfil_p3()),
            hiper,
            nombre_datos="p3-perfiles-train",
            extra_config={
                "pregunta": "P3",
                "modelo": "kmeans",
                "cohorte_min_dias_activos_train": P3_MIN_DIAS_TRAIN,
            },
        )


def ejecutar(viajes=None, estaciones=None, modelo="default", hiper=None) -> None:
    """Entrena y evalúa P3 y registra el run.

    ``viajes`` y ``estaciones`` se aceptan por compatibilidad con ``train.py`` y se
    ignoran: P3 lee los viajes por mes y el catálogo después de verificarlos
    contra el manifiesto.
    """
    if modelo not in ("default", "kmeans"):
        raise ValueError(f"Modelo no disponible para P3: {modelo}")
    run_kmeans(preparar(), hiper)


def buscar(viajes=None, estaciones=None, modelo="kmeans") -> None:
    """Búsqueda de k con nested runs: un padre y un hijo por combinación de BUSQUEDAS."""
    combinaciones = BUSQUEDAS[("P3", "kmeans")]
    prep = preparar()
    with iniciar_run(
        "P3",
        "p3-busqueda-kmeans",
        "KMeans",
        {"search_method": "grid", "n_combinations": len(combinaciones)},
        etapa="search",
    ):
        for cambios in combinaciones:
            run_kmeans(prep, cambios, etapa="search")


# --- Auditoría de la evaluación congelada --------------------------------------


def fuentes_codigo():
    nombres = [
        "mibici/config.py",
        "mibici/dataset.py",
        "mibici/features.py",
        "mibici/plots.py",
        "mibici/evaluation.py",
        "mibici/modeling/p3.py",
        "mibici/modeling/train.py",
        "mibici/tracking.py",
    ]
    return {n: sha256(PROJ_ROOT / n) for n in nombres}


def congelar(destino=P3_EVALUACION_CONGELADA_DIR, *, reproduccion=False):
    """Se llama antes de abrir septiembre. Nunca selecciona usando prueba."""
    if destino.exists():
        raise FileExistsError("Configuracion ya congelada; no se sobrescribe")
    spec = json.loads((PROJ_ROOT / "references/p3_frozen.json").read_text(encoding="utf-8"))
    if sha256(P3_PREPARACION_PARQUET) != spec["training_preparation_sha256"]:
        raise ValueError("La preparacion cambio desde la revision")
    train = cohorte_historia(pd.read_parquet(P3_PREPARACION_PARQUET), 180)
    if len(train) != 369:
        raise ValueError("Cambio la cohorte revisada de369estaciones")
    modelo = KMeans(
        n_clusters=3, random_state=RANDOM_SEED, n_init=P3_N_INIT, max_iter=300, algorithm="lloyd"
    ).fit(matriz_forma(train))
    centers_sha = hashlib.sha256(
        np.asarray(modelo.cluster_centers_, dtype="<f8").tobytes()
    ).hexdigest()
    if centers_sha != spec["centers_float64_sha256"]:
        raise ValueError("No reproduce centroides congelados")
    destino.mkdir(parents=True)
    joblib.dump(modelo, destino / "kmeans_train_congelado.joblib")
    train.to_parquet(destino / "perfiles_train_congelados.parquet", index=False)
    p = {
        "congelado_utc": datetime.now(UTC).isoformat(),
        "modelo": "KMeans",
        "k": 3,
        "cohorte_min_dias_train": 180,
        "n_train": 369,
        "random_state": RANDOM_SEED,
        "n_init": P3_N_INIT,
        "max_iter": 300,
        "algorithm": "lloyd",
        "distancia": "euclidea sobre sqrt(p)/sqrt(2),336variables;igual peso direcciones",
        "escalado": "sin StandardScaler;probabilidades separadas por direccion",
        "umbral_mes_dias": P3_MES_MIN_DIAS,
        "umbral_mes_movimientos": P3_MES_MIN_MOVIMIENTOS,
        "salvaguardas_mes": "7weekdays y ambas direcciones positivas",
        "test_inicio": str(TEST_START),
        "test_fin_exclusivo": str(DATA_END_EXCLUSIVE),
        "preparacion_sha256": sha256(P3_PREPARACION_PARQUET),
        "modelo_sha256": sha256(destino / "kmeans_train_congelado.joblib"),
        "fuentes_codigo_sha256": fuentes_codigo(),
        "septiembre_inspeccionado_al_congelar": reproduccion,
        "decision_original_utc": spec["decision_frozen_before_first_test_utc"],
        "reproduccion_sin_tuning": reproduccion,
        "decision": "Reproduccion autorizada sin tuning"
        if reproduccion
        else "Autorizada antes de prueba; no ajustar tras ver septiembre",
        "limites": "Grupos descriptivos superpuestos; disponibilidad/cobertura desconocidas",
    }
    (destino / "CONFIGURACION_CONGELADA.json").write_text(
        json.dumps(p, indent=2), encoding="utf-8"
    )
    print("Configuracion congelada ANTES de septiembre", p["congelado_utc"], flush=True)
    return p


def evaluar_congelado(*, destino=P3_EVALUACION_CONGELADA_DIR, reproduccion=False):
    """Auditoría: evalúa septiembre una sola vez con la configuración congelada.

    Congela primero si ``destino`` no existe. Con ``reproduccion=True`` exige Git
    limpio y compara contra ``references/p3_frozen.json``. Un marcador impide
    repetir o sobrescribir la evaluación.
    """
    if reproduccion and obtener_commit()[1]:
        raise ValueError("La reproduccion requiere estado Git versionado limpio")
    if not destino.exists():
        congelar(destino, reproduccion=reproduccion)
    if (destino / "EVALUACION_INICIADA.json").exists():
        raise FileExistsError("Evaluacion ya iniciada: no se repite ni sobrescribe")
    frozen = json.loads((destino / "CONFIGURACION_CONGELADA.json").read_text(encoding="utf-8"))
    if frozen["fuentes_codigo_sha256"] != fuentes_codigo() or frozen["modelo_sha256"] != sha256(
        destino / "kmeans_train_congelado.joblib"
    ):
        raise ValueError("Fuentes/modelo cambiaron despues de congelar")
    modelo = joblib.load(destino / "kmeans_train_congelado.joblib")
    centros = modelo.cluster_centers_.copy()
    train = pd.read_parquet(destino / "perfiles_train_congelados.parquet")
    labels = modelo.predict(matriz_forma(train))
    # Marca persistente ANTES de leer septiembre: no se repite la prueba por accidente.
    (destino / "EVALUACION_INICIADA.json").write_text(
        json.dumps(
            {
                "utc": datetime.now(UTC).isoformat(),
                "configuracion_congelada_sha256": sha256(destino / "CONFIGURACION_CONGELADA.json"),
            }
        ),
        encoding="utf-8",
    )
    manifest = json.loads((MANIFESTS_DIR / "mibici_processed.json").read_text(encoding="utf-8"))
    records = {r["filename"]: r for r in manifest["files"]}
    agosto, check_aug = leer_verificado("datos_abiertos_2026_08.parquet", records)
    boundary = agosto.loc[pd.to_datetime(agosto.fin_viaje).ge(pd.Timestamp(TEST_START))]
    septiembre, check_test = leer_verificado("datos_abiertos_2026_09.parquet", records)
    periodo = perfiles_agosto(
        pd.concat([boundary, septiembre], ignore_index=True),
        train,
        inicio=TEST_START,
        fin=DATA_END_EXCLUSIVE,
    ).rename(columns={"eventos_agosto_1": "eventos_septiembre_1"})
    periodo.to_parquet(destino / "perfiles_septiembre.parquet", index=False)
    q, a, testmetrics, groups = resumir_fijo(modelo, train, periodo)
    a.to_csv(destino / "asignaciones_test.csv", index=False)
    groups.to_csv(destino / "metricas_grupos_test.csv", index=False)
    periodo.loc[
        ~periodo.evaluable,
        ["estacion_id", "nombre", "dias_con_eventos", "total_movimientos", "motivo_exclusion"],
    ].to_csv(destino / "exclusiones_test.csv", index=False)
    pd.crosstab(a.grupo_train, a.grupo_periodo).reindex(
        index=range(3), columns=range(3), fill_value=0
    ).to_csv(destino / "transiciones_train_test.csv")
    for g in range(3):
        a[a.grupo_periodo.eq(g)].nsmallest(3, "distancia_centro_asignado").to_csv(
            destino / f"representantes_test_grupo_{g}.csv", index=False
        )
    guardar_grafica(train, labels, "train_congelado", destino)
    guardar_grafica(
        q,
        a.grupo_periodo.to_numpy(),
        "test_septiembre",
        destino,
        contexto="septiembre: asignacion a centroides train fijos",
    )
    matched = train.set_index("estacion_id").loc[q.estacion_id].reset_index()
    guardar_grafica(matched, a.grupo_train.to_numpy(), "train_mismos_ids_test", destino)
    station_path = PROCESSED_DATA_DIR / manifest["stations"]["filename"]
    if sha256(station_path) != manifest["stations"]["sha256"]:
        raise ValueError("SHA catalogo no coincide")
    estaciones = pd.read_parquet(station_path)
    mapped, missing = guardar_mapa(train, labels, estaciones, destino)
    train[["estacion_id", "nombre", "dias_con_eventos", "total_movimientos"]].assign(
        grupo_train=labels
    ).to_csv(destino / "asignaciones_train.csv", index=False)
    val = pd.read_parquet(P3_VALIDACION_DIR / "perfiles_agosto.parquet")
    _, aval, valmetrics, _ = resumir_fijo(modelo, train, val)
    aval.to_csv(destino / "asignaciones_val.csv", index=False)
    valmetrics_previous = json.loads(
        (P3_VALIDACION_DIR / "resumen.json").read_text(encoding="utf-8")
    )
    if not np.isclose(valmetrics["fraccion_estable"], valmetrics_previous["fraccion_estable"]):
        raise ValueError("Validacion no reproduce resumen revisado")
    trainmetrics = {
        "n_estaciones": len(train),
        "silhouette": float(silhouette_score(matriz_forma(train), labels)),
        "n_mapeables": mapped,
        "n_sin_ubicacion": missing,
    }
    pd.DataFrame(
        [
            check_aug,
            check_test,
            {
                "archivo": manifest["stations"]["filename"],
                "sha256": manifest["stations"]["sha256"],
                "registros": len(estaciones),
            },
        ]
    ).to_csv(destino / "integridad_fuentes_finales.csv", index=False)
    summary = {
        "train": trainmetrics,
        "val": valmetrics,
        "test": testmetrics,
        "centroides_sin_refit": bool(np.array_equal(centros, modelo.cluster_centers_)),
        "catalogo_fuente": manifest["stations"]["source_filename"],
        "limites": "Disponibilidad/cobertura desconocidas; mes vs anio; distancias p95train no calibradas; sin tuning tras test",
        "llegadas_limite_desde_agosto_registros": len(boundary),
    }
    (destino / "resultados.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not summary["centroides_sin_refit"]:
        raise AssertionError("Centroides modificados")
    if reproduccion:
        verificar_reproduccion(summary, destino)
    params = {
        "k": 3,
        "cohorte_min_dias_activos_train": 180,
        "n_init": P3_N_INIT,
        "max_iter": 300,
        "algorithm": "lloyd",
        "n_variables": 336,
        "transformacion": "sqrt(p)/sqrt(2)",
        "distancia": "euclidean",
        "min_dias_mes": P3_MES_MIN_DIAS,
        "min_movimientos_mes": P3_MES_MIN_MOVIMIENTOS,
        "fit": "train_only",
        "configuracion_congelada_utc": frozen["congelado_utc"],
        "catalogo_coordenadas": manifest["stations"]["source_filename"],
    }
    with iniciar_run("P3", "p3-kmeans-k3-historia180-congelado", "KMeans", params) as run:
        mlflow.set_tags(
            {
                "p3_code_uncommitted": str(obtener_commit()[1]).lower(),
                "evaluation_role": "frozen_configuration_test_once",
                "p3_config_sha256": sha256(destino / "CONFIGURACION_CONGELADA.json"),
                "interpretacion": "descriptiva_con_solapamiento",
            }
        )
        registrar_metricas("train", trainmetrics)
        registrar_metricas("val", valmetrics)
        registrar_metricas("test", testmetrics)
        for path in destino.iterdir():
            if path.is_file():
                registrar_artefacto(path, "evidencia_p3")
        for relative in frozen["fuentes_codigo_sha256"]:
            registrar_artefacto(
                PROJ_ROOT / relative, "codigo/" + str(Path(relative).parent).replace("\\", "/")
            )
        for previous in (
            P3_SENSIBILIDAD_DIR / "comparacion_cohortes.csv",
            P3_SENSIBILIDAD_DIR / "estabilidad_entre_cohortes.csv",
            P3_VALIDACION_DIR / "grupo_domingo_recurrencia.csv",
        ):
            registrar_artefacto(previous, "diagnosticos_previos")
        run_id = run.info.run_id
    (destino / "mlflow_run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "experiment_id": run.info.experiment_id,
                "git_dirty": obtener_commit()[1],
                "status": "FINISHED",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    print("MLflow run:", run_id, flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auditoría de la evaluación congelada de P3.")
    parser.add_argument("--solo-congelar", action="store_true")
    parser.add_argument(
        "--reproducir",
        action="store_true",
        help="Replay exacto en salida separada; requiere Git limpio",
    )
    args = parser.parse_args()
    if args.solo_congelar and args.reproducir:
        parser.error("Elegir solo una fase")
    if args.reproducir:
        evaluar_congelado(destino=P3_REPRODUCCION_DIR, reproduccion=True)
    elif args.solo_congelar:
        congelar()
    else:
        evaluar_congelado()
