"""Comparación y evaluación reutilizable.

P3: comparación de configuraciones, sensibilidad a la historia mínima,
validación tentativa en agosto, métricas con centroides fijos y verificación de
la reproducción congelada. Las funciones ``exportar_*`` guardan la evidencia de
cada etapa en una carpeta nueva; nunca sobrescriben. La secuencia se documenta
en ``notebooks/2.0-fvep-p3-seleccion.ipynb`` y la auditoría en
``notebooks/3.0-fvep-p3-auditoria.ipynb``.

Este módulo no importa ``mibici.modeling``: ``modeling/p3.py`` lo usa a él.
"""

import hashlib
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from mibici.config import (
    P3_COHORTES,
    P3_K_EXPLORACION,
    P3_MES_MIN_DIAS,
    P3_MES_MIN_MOVIMIENTOS,
    P3_N_INIT,
    PROJ_ROOT,
    RANDOM_SEED,
    TRAIN_START,
    VALIDATION_START,
)
from mibici.dataset import leer_verificado, sha256
from mibici.features import (
    cohorte_historia,
    columnas_perfil_p3,
    eventos_periodo,
    matriz_forma,
    perfiles_agosto,
)
from mibici.plots import graficar_comparacion_k, guardar_grafica

# --- Asignación y métricas con centroides fijos -----------------------------


def asignar_fijo(modelo, perfiles):
    """Solo predict/transform: nunca ajusta centroides a validacion."""
    x = matriz_forma(perfiles)
    labels = modelo.predict(x)
    dist = modelo.transform(x)
    return x, labels, dist[np.arange(len(x)), labels]


def resumir_fijo(modelo, train, periodo):
    """Asigna el periodo a centroides fijos y resume estabilidad; válido para cualquier k."""
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
    k = modelo.n_clusters
    p95 = np.array([np.quantile(dtrain[full_labels == g], 0.95) for g in range(k)])
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
    for g in range(k):
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


def comparar_con_congelado(resultados: dict) -> bool:
    """Indica si las métricas coinciden con las registradas al congelar la configuración."""
    spec = json.loads((PROJ_ROOT / "references/p3_frozen.json").read_text(encoding="utf-8"))
    esperado = spec["expected_results"]
    return all(
        key in resultados[conjunto] and np.isclose(resultados[conjunto][key], valor, rtol=1e-9)
        for conjunto in ("train", "val", "test")
        for key, valor in esperado[conjunto].items()
    )


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


# --- Comparación de configuraciones (selección de k) -------------------------


def comparar(perfiles):
    """Diez configuraciones predeclaradas; etiquetas no implican ranking final."""
    if len(perfiles) <= max(P3_K_EXPLORACION):
        raise ValueError("Se necesitan mas de seis estaciones")
    x = matriz_forma(perfiles)
    filas, etiquetas, representantes = [], {}, []
    for metodo in ("kmeans", "ward"):
        for k in P3_K_EXPLORACION:
            modelo = (
                KMeans(
                    n_clusters=k,
                    random_state=RANDOM_SEED,
                    n_init=P3_N_INIT,
                    max_iter=300,
                    algorithm="lloyd",
                )
                if metodo == "kmeans"
                else AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")
            )
            labels = modelo.fit_predict(x)
            sizes = np.bincount(labels, minlength=k)
            if len(np.unique(labels)) != k:
                raise ValueError("Configuracion degenerada: menos grupos que k")
            nombre = f"{metodo}_k{k}"
            etiquetas[nombre] = labels
            filas.append(
                {
                    "configuracion": nombre,
                    "metodo": metodo,
                    "k": k,
                    "silhouette": silhouette_score(x, labels, metric="euclidean"),
                    "tamanos": " / ".join(map(str, sizes)),
                    "minimo": int(sizes.min()),
                    "maximo": int(sizes.max()),
                    "fraccion_mayor": float(sizes.max() / len(x)),
                    "cv_tamano": float(sizes.std() / sizes.mean()),
                    "grupos_unitarios": int((sizes == 1).sum()),
                }
            )
            for grupo in range(k):
                miembros = np.flatnonzero(labels == grupo)
                centro = x[miembros].mean(axis=0)
                dist = np.linalg.norm(x[miembros] - centro, axis=1)
                for rank, pos in enumerate(np.argsort(dist, kind="stable")[:3], 1):
                    row = perfiles.iloc[miembros[pos]]
                    representantes.append(
                        {
                            "configuracion": nombre,
                            "grupo": grupo,
                            "n_estaciones": int(sizes[grupo]),
                            "rango": rank,
                            "estacion_id": row.estacion_id,
                            "nombre": row.get("nombre", ""),
                            "distancia_centro": float(dist[pos]),
                            "movimientos_contexto": row.get("total_movimientos", np.nan),
                        }
                    )
    return pd.DataFrame(filas), etiquetas, pd.DataFrame(representantes)


def exportar_comparacion(salida, entrada, p, tabla, etiquetas, reps):
    """Guarda la evidencia de la comparación de k: tablas, asignaciones, gráficas y protocolo."""
    if salida.exists():
        raise FileExistsError("La salida debe ser nueva")
    salida.mkdir(parents=True)
    tabla.to_csv(salida / "comparacion.csv", index=False)
    reps.to_csv(salida / "representantes.csv", index=False)
    asignaciones = p[["estacion_id", "nombre"]].copy()
    for nombre, labels in etiquetas.items():
        asignaciones[nombre] = labels
        guardar_grafica(p, labels, nombre, salida)
    asignaciones.to_csv(salida / "asignaciones.csv", index=False)
    graficar_comparacion_k(tabla, len(p), salida)
    with entrada.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    resumen = {
        "estaciones": len(p),
        "variables": 336,
        "k": list(P3_K_EXPLORACION),
        "semilla": RANDOM_SEED,
        "n_init": P3_N_INIT,
        "max_iter": 300,
        "entrada": str(entrada.resolve()),
        "sha256": sha,
        "distancia": "euclidea sobre sqrt(p)/sqrt(2); cada bloque direccional suma p=1",
        "escalado": "sin StandardScaler; cada bloque aporta norma cuadrada 1/2",
        "seleccion_final": None,
        "holdouts_leidos": False,
        "mlflow_registrado": False,
    }
    (salida / "protocolo.json").write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    print(tabla.to_string(index=False))


# --- Sensibilidad a la historia mínima ---------------------------------------


def eta_historia(dias, labels):
    """Fraccion descriptiva de variacion de dias entre grupos; no prueba causal."""
    dias = np.asarray(dias, dtype=float)
    total = ((dias - dias.mean()) ** 2).sum()
    if total == 0:
        return 0.0
    entre = sum(
        (labels == g).sum() * (dias[labels == g].mean() - dias.mean()) ** 2
        for g in np.unique(labels)
    )
    return float(entre / total)


def sensibilidad_historia(p, original):
    """Repite la comparación de k en cada cohorte predeclarada de historia mínima.

    ``p`` es la preparación de entrenamiento y ``original`` las asignaciones de la
    comparación previa (mismas estaciones elegibles). No lee validación ni prueba.
    """
    base = p.loc[p.elegible].sort_values("estacion_id").reset_index(drop=True)
    if set(base.estacion_id) != set(original.estacion_id):
        raise ValueError("Asignaciones anteriores no corresponden a elegibles")
    cohortes, contexto, patrones, conteos, asociaciones = [], [], [], [], []
    for minimo in P3_COHORTES:
        q = cohorte_historia(p, minimo)
        prefijo = f"dias_ge_{minimo}"
        tabla, etiquetas, reps = comparar(q)
        tabla.insert(0, "cohorte", prefijo)
        tabla.insert(1, "n_estaciones", len(q))
        asignaciones = q[["estacion_id", "nombre", "dias_con_eventos", "total_movimientos"]].copy()
        prev = q[["estacion_id"]].merge(original, on="estacion_id", validate="one_to_one")
        if not np.array_equal(q.estacion_id, prev.estacion_id):
            raise ValueError("Orden de estaciones inconsistente")
        conteos.append(
            {
                "cohorte": prefijo,
                "n": len(q),
                "excluidos_de_429": len(base) - len(q),
                "ward_original_grupo0": int(prev.ward_k2.eq(0).sum()),
                "ward_original_grupo1": int(prev.ward_k2.eq(1).sum()),
            }
        )
        for nombre, labels in etiquetas.items():
            asignaciones[nombre] = labels
            tabla.loc[tabla.configuracion.eq(nombre), "ari_vs_original_mismas_estaciones"] = (
                adjusted_rand_score(prev[nombre], labels)
            )
            tabla.loc[tabla.configuracion.eq(nombre), "eta2_dias_historia"] = eta_historia(
                q.dias_con_eventos, labels
            )
            for grupo in np.unique(labels):
                z = q.loc[labels == grupo]
                contexto.append(
                    {
                        "cohorte": prefijo,
                        "configuracion": nombre,
                        "grupo": int(grupo),
                        "n": len(z),
                        "dias_min": int(z.dias_con_eventos.min()),
                        "dias_mediana": float(z.dias_con_eventos.median()),
                        "dias_max": int(z.dias_con_eventos.max()),
                        "movimientos_mediana": float(z.total_movimientos.median()),
                    }
                )
                mean = z[columnas_perfil_p3()].to_numpy().mean(axis=0).reshape(2, 7, 24)
                for d, direccion in enumerate(("salidas", "llegadas")):
                    hora = mean[d].sum(axis=0)
                    dia = mean[d].sum(axis=1)
                    patrones.append(
                        {
                            "cohorte": prefijo,
                            "configuracion": nombre,
                            "grupo": int(grupo),
                            "direccion": direccion,
                            "hora_pico": int(hora.argmax()),
                            "pct_06_10": float(hora[6:10].sum() * 100),
                            "pct_17_21": float(hora[17:21].sum() * 100),
                            "pct_fin_semana": float(dia[5:].sum() * 100),
                            "pct_domingo": float(dia[6] * 100),
                        }
                    )
        for k in range(2, 7):
            asociaciones.append(
                {
                    "cohorte": prefijo,
                    "k": k,
                    "ari_kmeans_ward": adjusted_rand_score(
                        etiquetas[f"kmeans_k{k}"], etiquetas[f"ward_k{k}"]
                    ),
                }
            )
        print(prefijo, "n=", len(q), flush=True)
        print(tabla.to_string(index=False), flush=True)
        cohortes.append(
            {
                "prefijo": prefijo,
                "perfiles": q,
                "tabla": tabla,
                "etiquetas": etiquetas,
                "representantes": reps,
                "asignaciones": asignaciones,
            }
        )
    b = base[["estacion_id", "dias_con_eventos"]].merge(original, on="estacion_id")
    asociacion_base = {
        c: eta_historia(b.dias_con_eventos, b[c].to_numpy())
        for c in original
        if c not in ("estacion_id", "nombre")
    }
    return {
        "cohortes": cohortes,
        "contexto": pd.DataFrame(contexto),
        "patrones": pd.DataFrame(patrones),
        "conteos": pd.DataFrame(conteos),
        "asociaciones": pd.DataFrame(asociaciones),
        "asociacion_base": asociacion_base,
    }


def exportar_sensibilidad(salida, entrada, anterior, resultado):
    """Guarda la evidencia de la sensibilidad a la historia: tablas por cohorte, gráficas y protocolo."""
    if salida.exists():
        raise FileExistsError("La salida debe ser nueva")
    salida.mkdir(parents=True)
    tablas = []
    for cohorte in resultado["cohortes"]:
        destino = salida / cohorte["prefijo"]
        destino.mkdir()
        cohorte["representantes"].to_csv(destino / "representantes.csv", index=False)
        for nombre, labels in cohorte["etiquetas"].items():
            guardar_grafica(cohorte["perfiles"], labels, f"{cohorte['prefijo']}_{nombre}", destino)
        cohorte["tabla"].to_csv(destino / "comparacion.csv", index=False)
        tablas.append(cohorte["tabla"])
        cohorte["asignaciones"].to_csv(destino / "asignaciones.csv", index=False)
    pd.concat(tablas).to_csv(salida / "comparacion_cohortes.csv", index=False)
    resultado["contexto"].to_csv(salida / "contexto_grupos.csv", index=False)
    resultado["patrones"].to_csv(salida / "resumen_patrones.csv", index=False)
    resultado["conteos"].to_csv(salida / "conteos.csv", index=False)
    resultado["asociaciones"].to_csv(salida / "acuerdo_metodos.csv", index=False)
    with entrada.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    (salida / "protocolo.json").write_text(
        json.dumps(
            {
                "cohortes_predeclaradas_dias_activos": P3_COHORTES,
                "entrada": str(entrada.resolve()),
                "sha256": sha,
                "comparacion_previa": str(anterior.resolve()),
                "protocolo": "mismo comparar(): KMeans seed42 n_init20 max_iter300, Ward, k2-6",
                "distancia": "euclidea sqrt(p)/sqrt(2); 336 variables, peso igual por direccion",
                "eta2_historia_original": resultado["asociacion_base"],
                "seleccion_final": None,
                "holdouts_leidos": False,
                "limite": "Filtrar poblacion no mejora cobertura ni demuestra causalidad",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


# --- Validación tentativa en agosto ------------------------------------------


def exigir_protocolo_previo(salida):
    """La validación en agosto exige un protocolo escrito a mano antes de leer agosto.

    ``PROTOCOLO_PREVIO.txt`` se escribe antes de ejecutar la etapa y no se genera
    desde el código. Una salida que ya tiene ``resumen.json`` no se vuelve a usar.
    """
    if not (salida / "PROTOCOLO_PREVIO.txt").is_file() or (salida / "resumen.json").exists():
        raise ValueError("Requiere protocolo previo y salida no ejecutada")


def modelo_tentativo(preparacion, anterior, salida):
    """Reajusta en entrenamiento el KMeans tentativo y comprueba que reproduce la comparación previa."""
    exigir_protocolo_previo(salida)
    train = cohorte_historia(preparacion, 180)
    x = matriz_forma(train)
    modelo = KMeans(
        n_clusters=3, random_state=RANDOM_SEED, n_init=P3_N_INIT, max_iter=300, algorithm="lloyd"
    ).fit(x)
    prev = pd.read_csv(anterior / "dias_ge_180/asignaciones.csv")
    if not np.array_equal(train.estacion_id, prev.estacion_id) or not np.array_equal(
        modelo.labels_, prev.kmeans_k3
    ):
        raise ValueError("No reproduce modelo tentativo anterior")
    centros = modelo.cluster_centers_.copy()
    return train, x, modelo, centros


def grupo_domingo(train, modelo):
    """Identifica el grupo con mayor peso de domingo usando solo entrenamiento."""
    raw = train[columnas_perfil_p3()].to_numpy().reshape(-1, 2, 7, 24)
    dom = raw[:, :, 6, :].sum(axis=(1, 2)) / 2
    # Identificar el grupo domingo por media descriptiva train, no por validacion.
    grupo_dom = int(max(range(3), key=lambda g: dom[modelo.labels_ == g].mean()))
    miembros = train.loc[modelo.labels_ == grupo_dom].copy()
    miembros["grupo_train"] = grupo_dom
    miembros["fraccion_domingo_media_direcciones"] = dom[modelo.labels_ == grupo_dom]
    print(
        f"Train: {len(train)}; grupo domingo {grupo_dom}: {len(miembros)} estaciones", flush=True
    )
    return grupo_dom, miembros


def recurrencia_grupo_domingo(miembros, records):
    """Recurrencia mensual del uso en domingo del grupo; lee y verifica solo meses de entrenamiento.

    Devuelve también las llegadas de julio que terminan en agosto, necesarias para
    el primer día de la validación.
    """
    integridad, eventos_domingo = [], []
    julio = None
    for mes in pd.period_range(
        TRAIN_START, pd.Timestamp(VALIDATION_START) - pd.Timedelta(days=1), freq="M"
    ):
        nombre = f"datos_abiertos_{mes.year}_{mes.month:02d}.parquet"
        frame, check = leer_verificado(nombre, records)
        integridad.append(check)
        if mes == pd.Period("2026-07", freq="M"):
            julio = frame.loc[
                pd.to_datetime(frame.fin_viaje).ge(pd.Timestamp(VALIDATION_START))
            ].copy()
        mask = frame.origen_id.isin(miembros.estacion_id) | frame.destino_id.isin(
            miembros.estacion_id
        )
        eventos_domingo.append(eventos_periodo(frame.loc[mask], TRAIN_START, VALIDATION_START))
    e = pd.concat(eventos_domingo, ignore_index=True)
    e = e[e.estacion_id.isin(miembros.estacion_id)].merge(
        miembros[["estacion_id", "inicio_denominador_proxy"]],
        on="estacion_id",
        validate="many_to_one",
    )
    e = e[e.fecha.ge(pd.to_datetime(e.inicio_denominador_proxy))]
    diario = e.groupby(["estacion_id", "fecha", "tipo"]).size().rename("eventos").reset_index()
    recurrence, mensual = [], {}
    for station in miembros.itertuples():
        s = e[e.estacion_id.eq(station.estacion_id)]
        calendar = pd.date_range(
            station.inicio_denominador_proxy, pd.Timestamp(VALIDATION_START) - pd.Timedelta(days=1)
        )
        sundays = calendar[calendar.dayofweek == 6]
        days = s.groupby("fecha").size()
        sunday_counts = days.reindex(sundays, fill_value=0)
        months = []
        for month in sorted(set(calendar.to_period("M"))):
            cal = calendar[calendar.to_period("M") == month]
            counts = days.reindex(cal, fill_value=0)
            sunday_rate = counts[cal.dayofweek == 6].mean()
            other_rate = counts[cal.dayofweek != 6].mean()
            months.append(
                {
                    "estacion_id": station.estacion_id,
                    "mes": str(month),
                    "tasa_domingo": sunday_rate,
                    "tasa_otros_dias": other_rate,
                    "domingo_mayor": bool(sunday_rate > other_rate),
                }
            )
        mensual[station.estacion_id] = pd.DataFrame(months)
        recurrence.append(
            {
                "estacion_id": station.estacion_id,
                "nombre": station.nombre,
                "dias_train": station.dias_con_eventos,
                "movimientos_train": station.total_movimientos,
                "domingos_calendario_proxy": len(sundays),
                "domingos_con_eventos": int((sunday_counts > 0).sum()),
                "fraccion_domingo_maximo": float(sunday_counts.max() / sunday_counts.sum())
                if sunday_counts.sum()
                else None,
                "meses_domingo_mayor": sum(m["domingo_mayor"] for m in months),
                "meses_evaluados": len(months),
                "pct_domingo_perfil": station.fraccion_domingo_media_direcciones * 100,
            }
        )
    return {
        "diario": diario,
        "recurrencia": pd.DataFrame(recurrence),
        "mensual": mensual,
        "integridad": integridad,
        "julio": julio,
    }


def validacion_agosto(modelo, train, x, julio, records, integridad, salida):
    """Asigna agosto a los centroides de entrenamiento, sin reajustar, y resume la estabilidad."""
    exigir_protocolo_previo(salida)
    agosto, check = leer_verificado("datos_abiertos_2026_08.parquet", records)
    integridad.append(check)
    val = perfiles_agosto(pd.concat([julio, agosto], ignore_index=True), train)
    q = val[val.evaluable].sort_values("estacion_id").reset_index(drop=True)
    xv, labels, distances = asignar_fijo(modelo, q)
    matched = train.set_index("estacion_id").loc[q.estacion_id].reset_index()
    train_labels = modelo.predict(matriz_forma(matched))
    dtrain = modelo.transform(x)[np.arange(len(x)), modelo.labels_]
    p95 = np.array([np.quantile(dtrain[modelo.labels_ == g], 0.95) for g in range(3)])
    paired_shift = np.linalg.norm(xv - matriz_forma(matched), axis=1)
    asignaciones = q[["estacion_id", "nombre", "dias_con_eventos", "total_movimientos"]].copy()
    asignaciones["grupo_train"] = train_labels
    asignaciones["grupo_agosto"] = labels
    asignaciones["estable"] = labels == train_labels
    asignaciones["distancia_centro_train_asignado"] = distances
    asignaciones["cambio_perfil_pareado"] = paired_shift
    asignaciones["supera_p95_train_grupo_asignado"] = distances > p95[labels]
    transiciones = pd.crosstab(asignaciones.grupo_train, asignaciones.grupo_agosto).reindex(
        index=range(3), columns=range(3), fill_value=0
    )
    grupos, representantes = [], {}
    for g in range(3):
        ids = np.flatnonzero(labels == g)
        if not len(ids):
            continue
        order = ids[np.argsort(distances[ids], kind="stable")[:3]]
        representantes[g] = asignaciones.iloc[order].copy()
        grupos.append(
            {
                "grupo": g,
                "n_train": int((modelo.labels_ == g).sum()),
                "n_train_evaluables": int((train_labels == g).sum()),
                "n_asignados_agosto": len(ids),
                "retencion_train": float((labels[train_labels == g] == g).mean()),
                "distancia_train_mediana": float(np.median(dtrain[modelo.labels_ == g])),
                "distancia_train_p95": float(p95[g]),
                "distancia_val_mediana": float(np.median(distances[ids])),
                "n_val_sobre_p95_train": int((distances[ids] > p95[g]).sum()),
            }
        )
    return {
        "val": val,
        "q": q,
        "matched": matched,
        "xv": xv,
        "labels": labels,
        "train_labels": train_labels,
        "distances": distances,
        "p95": p95,
        "paired_shift": paired_shift,
        "asignaciones": asignaciones,
        "transiciones": transiciones,
        "grupos": grupos,
        "representantes": representantes,
    }


def resumen_validacion(train, modelo, grupo_dom, miembros, julio, v):
    """Resumen de la validación tentativa; deja explícito que septiembre no se leyó."""
    labels, train_labels = v["labels"], v["train_labels"]
    return {
        "cohorte_primaria": ">=180 dias train",
        "n_train": len(train),
        "n_evaluables_agosto": len(v["q"]),
        "n_excluidas_agosto": len(v["val"]) - len(v["q"]),
        "grupo_domingo_train": grupo_dom,
        "n_grupo_domingo_train": len(miembros),
        "tamanos_train": np.bincount(modelo.labels_, minlength=3).tolist(),
        "tamanos_agosto": np.bincount(labels, minlength=3).tolist(),
        "fraccion_estable": float((labels == train_labels).mean()),
        "ari_train_val": float(adjusted_rand_score(train_labels, labels)),
        "silhouette_val_asignacion_fija": float(silhouette_score(v["xv"], labels))
        if 2 <= len(set(labels)) < len(labels)
        else None,
        "cambio_perfil_pareado_mediana": float(np.median(v["paired_shift"])),
        "cambio_perfil_pareado_p95": float(np.quantile(v["paired_shift"], 0.95)),
        "distancia_val_mediana": float(np.median(v["distances"])),
        "fraccion_val_sobre_p95_train_asignado": float((v["distances"] > v["p95"][labels]).mean()),
        "centroides_sin_refit": True,
        "septiembre_leido": False,
        "seleccion_final": None,
        "umbral_agosto_dias": P3_MES_MIN_DIAS,
        "umbral_agosto_movimientos": P3_MES_MIN_MOVIMIENTOS,
        "llegadas_limite_izquierdo_desde_julio": len(julio),
        "eventos_agosto1_estaciones_train": int(v["val"].eventos_agosto_1.sum()),
    }


def exportar_validacion(salida, train, modelo, centros, miembros, recurrencia, v, resumen):
    """Guarda la evidencia de la validación en agosto; comprueba que los centroides no cambiaron."""
    recurrencia["diario"].to_csv(salida / "grupo_domingo_eventos_diarios_train.csv", index=False)
    for estacion_id, meses in recurrencia["mensual"].items():
        meses.to_csv(salida / f"recurrencia_mensual_estacion_{estacion_id}.csv", index=False)
    recurrencia["recurrencia"].to_csv(salida / "grupo_domingo_recurrencia.csv", index=False)
    miembros.to_csv(salida / "grupo_domingo_miembros_train.csv", index=False)
    val = v["val"]
    val.to_parquet(salida / "perfiles_agosto.parquet", index=False)
    val.loc[
        ~val.evaluable,
        ["estacion_id", "nombre", "dias_con_eventos", "total_movimientos", "motivo_exclusion"],
    ].to_csv(salida / "excluidas_agosto.csv", index=False)
    v["asignaciones"].to_csv(salida / "asignaciones_agosto.csv", index=False)
    v["transiciones"].to_csv(salida / "transiciones.csv")
    guardar_grafica(
        v["q"],
        v["labels"],
        "agosto_asignacion_centroides_train",
        salida,
        contexto="agosto: centroides ajustados en train",
    )
    guardar_grafica(v["matched"], v["train_labels"], "train_mismos_ids_evaluables", salida)
    for g, reps in v["representantes"].items():
        reps.to_csv(salida / f"representantes_agosto_grupo_{g}.csv", index=False)
    pd.DataFrame(v["grupos"]).to_csv(salida / "metricas_por_grupo.csv", index=False)
    if not np.array_equal(centros, modelo.cluster_centers_):
        raise AssertionError("Centroides modificados")
    joblib.dump(modelo, salida / "kmeans_train_tentativo.joblib")
    pd.DataFrame(centros, columns=columnas_perfil_p3()).to_csv(
        salida / "centroides_transformados_train.csv", index=False
    )
    pd.DataFrame(recurrencia["integridad"]).to_csv(salida / "integridad_fuentes.csv", index=False)
    train.assign(grupo_train=modelo.labels_)[["estacion_id", "nombre", "grupo_train"]].to_csv(
        salida / "asignaciones_train.csv", index=False
    )
    (salida / "resumen.json").write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    print(json.dumps(resumen, indent=2), flush=True)
    print(pd.DataFrame(v["grupos"]).to_string(index=False), flush=True)
    print(recurrencia["recurrencia"].describe().to_string(), flush=True)
