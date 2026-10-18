"""P3 congelado: KMeans descriptivo, evaluacion unica de septiembre y MLflow."""

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from mibici.compare_p3 import N_INIT, SEED, guardar_grafica, matriz_forma
from mibici.config import (
    DATA_END_EXCLUSIVE,
    MANIFESTS_DIR,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    REPORTS_DIR,
    TEST_START,
)
from mibici.history_p3 import cohorte_historia
from mibici.tracking import iniciar_run, obtener_commit, registrar_artefacto, registrar_metricas
from mibici.validate_p3 import (
    AUG_MIN_DAYS,
    AUG_MIN_MOVEMENTS,
    asignar_fijo,
    leer_verificado,
    perfiles_agosto,
)

DESTINO = REPORTS_DIR / "p3-evaluacion-congelada-20261018"
PREPARACION = REPORTS_DIR / "p3-preparacion-20261018/perfiles_entrenamiento.parquet"
VALIDACION = REPORTS_DIR / "p3-validacion-agosto-tentativa-20261018"
SENSIBILIDAD = REPORTS_DIR / "p3-sensibilidad-historia-20261018"
COLORES = ("#2166ac", "#d6604d", "#1b7837")


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fuentes_codigo():
    nombres = [
        "mibici/config.py",
        "mibici/features.py",
        "mibici/features_p3.py",
        "mibici/compare_p3.py",
        "mibici/history_p3.py",
        "mibici/validate_p3.py",
        "mibici/modeling/p3.py",
        "mibici/modeling/train.py",
        "mibici/tracking.py",
    ]
    return {n: sha256(PROJ_ROOT / n) for n in nombres}


def congelar(destino=DESTINO, *, reproduccion=False):
    """Se llama antes de abrir septiembre. Nunca selecciona usando prueba."""
    if destino.exists():
        raise FileExistsError("Configuracion ya congelada; no se sobrescribe")
    spec = json.loads((PROJ_ROOT / "references/p3_frozen.json").read_text(encoding="utf-8"))
    if sha256(PREPARACION) != spec["training_preparation_sha256"]:
        raise ValueError("La preparacion cambio desde la revision")
    train = cohorte_historia(pd.read_parquet(PREPARACION), 180)
    if len(train) != 369:
        raise ValueError("Cambio la cohorte revisada de369estaciones")
    modelo = KMeans(
        n_clusters=3, random_state=SEED, n_init=N_INIT, max_iter=300, algorithm="lloyd"
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
        "random_state": SEED,
        "n_init": N_INIT,
        "max_iter": 300,
        "algorithm": "lloyd",
        "distancia": "euclidea sobre sqrt(p)/sqrt(2),336variables;igual peso direcciones",
        "escalado": "sin StandardScaler;probabilidades separadas por direccion",
        "umbral_mes_dias": AUG_MIN_DAYS,
        "umbral_mes_movimientos": AUG_MIN_MOVEMENTS,
        "salvaguardas_mes": "7weekdays y ambas direcciones positivas",
        "test_inicio": str(TEST_START),
        "test_fin_exclusivo": str(DATA_END_EXCLUSIVE),
        "preparacion_sha256": sha256(PREPARACION),
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


def resumir_fijo(modelo, train, periodo):
    q = periodo.loc[periodo.evaluable].sort_values("estacion_id").reset_index(drop=True)
    if len(q) < 2:
        raise ValueError("Muy pocas estaciones evaluables")
    x, labels, dist = asignar_fijo(modelo, q)
    matched = train.set_index("estacion_id").loc[q.estacion_id].reset_index()
    xt = matriz_forma(matched)
    lt = modelo.predict(xt)
    fullx = matriz_forma(train)
    full_labels = modelo.predict(fullx)
    dtrain = modelo.transform(fullx)[np.arange(len(fullx)), full_labels]
    p95 = np.array([np.quantile(dtrain[full_labels == g], 0.95) for g in range(3)])
    shift = np.linalg.norm(x - xt, axis=1)
    a = q[["estacion_id", "nombre", "dias_con_eventos", "total_movimientos"]].copy()
    a["grupo_train"] = lt
    a["grupo_periodo"] = labels
    a["estable"] = lt == labels
    a["distancia_centro_asignado"] = dist
    a["cambio_perfil_pareado"] = shift
    a["supera_p95_train_grupo_asignado"] = dist > p95[labels]
    metrics = {
        "n_evaluables": len(q),
        "n_excluidas": len(periodo) - len(q),
        "fraccion_evaluable": len(q) / len(train),
        "fraccion_estable": float((lt == labels).mean()),
        "ari_vs_train": float(adjusted_rand_score(lt, labels)),
        "distancia_mediana": float(np.median(dist)),
        "distancia_p95": float(np.quantile(dist, 0.95)),
        "cambio_perfil_mediana": float(np.median(shift)),
        "cambio_perfil_p95": float(np.quantile(shift, 0.95)),
        "fraccion_sobre_p95_train": float((dist > p95[labels]).mean()),
    }
    if 2 <= len(np.unique(labels)) < len(labels):
        metrics["silhouette_asignacion_fija"] = float(silhouette_score(x, labels))
    groups = []
    for g in range(3):
        original = lt == g
        assigned = labels == g
        metrics[f"n_grupo_{g}"] = int(assigned.sum())
        groups.append(
            {
                "grupo": g,
                "n_train": int((full_labels == g).sum()),
                "n_train_evaluables": int(original.sum()),
                "n_periodo": int(assigned.sum()),
                "retencion": float((labels[original] == g).mean()) if original.any() else None,
                "distancia_train_mediana": float(np.median(dtrain[full_labels == g])),
                "distancia_train_p95": float(p95[g]),
                "distancia_periodo_mediana": float(np.median(dist[assigned]))
                if assigned.any()
                else None,
            }
        )
    return q, a, metrics, pd.DataFrame(groups)


def guardar_mapa(train, labels, estaciones, destino):
    """Mapa de puntos sin basemap; coordenadas del catalogo septiembre2026."""
    datos = (
        train[["estacion_id", "nombre"]]
        .assign(grupo_train=labels)
        .merge(
            estaciones[["estacion_id", "latitud", "longitud"]],
            on="estacion_id",
            how="left",
            validate="one_to_one",
        )
    )
    valid = datos.latitud.between(-90, 90) & datos.longitud.between(-180, 180)
    datos["ubicacion_valida"] = valid
    datos.to_csv(destino / "estaciones_mapa.csv", index=False)
    datos.loc[~valid].to_csv(destino / "estaciones_sin_ubicacion.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    for g in range(3):
        z = datos.loc[valid & datos.grupo_train.eq(g)]
        ax.scatter(
            z.longitud,
            z.latitud,
            s=25,
            c=COLORES[g],
            label=f"Grupo {g}: {len(z)}",
            alpha=0.8,
            edgecolors="white",
            linewidths=0.3,
        )
    for station in (200, 11, 196):
        z = datos.loc[valid & datos.estacion_id.eq(station)]
        if len(z):
            ax.annotate(
                str(station),
                (z.longitud.iloc[0], z.latitud.iloc[0]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )
    if valid.any():
        ax.set_aspect(1 / np.cos(np.deg2rad(datos.loc[valid, "latitud"].mean())))
    ax.set(
        title=f"P3 · grupos de entrenamiento congelados\nCatalogo septiembre2026 · {int(valid.sum())}/{len(datos)} con ubicacion",
        xlabel="Longitud (grados)",
        ylabel="Latitud (grados)",
    )
    ax.text(0.98, 0.98, "N ↑", transform=ax.transAxes, ha="right", va="top")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.text(
        0.5,
        0.015,
        "Mapa de puntos sin cartografia base. Coordenadas actuales no prueban ubicacion historica ni disponibilidad.\nUbicacion y volumen NO usados para agrupar; etiquetas descriptivas con solapamiento.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(destino / "mapa_grupos_train.png", dpi=180)
    plt.close(fig)
    return int(valid.sum()), int((~valid).sum())


def verificar_reproduccion(summary, destino, spec=None):
    """Compara evidencia congelada; una discrepancia detiene, no provoca tuning."""
    if spec is None:
        spec = json.loads((PROJ_ROOT / "references/p3_frozen.json").read_text(encoding="utf-8"))
    expected = spec["expected_results"]
    for split in ("train", "val", "test"):
        if set(summary[split]) != set(expected[split]):
            raise ValueError(f"Metricas diferentes: {split}")
        for key, value in expected[split].items():
            if not np.isclose(summary[split][key], value, rtol=1e-12, atol=1e-12):
                raise ValueError(f"Reproduccion difiere: {split}.{key}")
    checks = {}
    for nombre, digest in spec["assignment_file_sha256"].items():
        checks[nombre] = sha256(destino / nombre) == digest
    if not all(checks.values()):
        raise ValueError("Asignaciones diferentes; no reajustar configuracion")
    (destino / "verificacion_reproduccion.json").write_text(
        json.dumps(
            {
                "historical_run_id": spec["historical_run_id"],
                "metricas_coinciden_tolerancia": "rtol=atol=1e-12",
                "asignaciones_sha256_coinciden": checks,
                "centroides_sin_refit": summary["centroides_sin_refit"],
                "tuning": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def ejecutar(viajes=None, estaciones=None, *, destino=DESTINO, reproduccion=False):
    """Entrada compartida; P3 verifica archivos por periodo, no usa viajes globales."""
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
    val = pd.read_parquet(VALIDACION / "perfiles_agosto.parquet")
    _, aval, valmetrics, _ = resumir_fijo(modelo, train, val)
    aval.to_csv(destino / "asignaciones_val.csv", index=False)
    valmetrics_previous = json.loads((VALIDACION / "resumen.json").read_text(encoding="utf-8"))
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
        "n_init": N_INIT,
        "max_iter": 300,
        "algorithm": "lloyd",
        "n_variables": 336,
        "transformacion": "sqrt(p)/sqrt(2)",
        "distancia": "euclidean",
        "min_dias_mes": AUG_MIN_DAYS,
        "min_movimientos_mes": AUG_MIN_MOVEMENTS,
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
            SENSIBILIDAD / "comparacion_cohortes.csv",
            SENSIBILIDAD / "estabilidad_entre_cohortes.csv",
            VALIDACION / "grupo_domingo_recurrencia.csv",
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

    parser = argparse.ArgumentParser(description=__doc__)
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
        ejecutar(destino=REPORTS_DIR / "p3-reproduccion-limpia-20261018", reproduccion=True)
    elif args.solo_congelar:
        congelar()
    else:
        ejecutar()
