"""Validacion tentativa P3 en agosto con modelo ajustado exclusivamente en train."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from mibici.compare_p3 import N_INIT, SEED, guardar_grafica, matriz_forma
from mibici.config import (
    MANIFESTS_DIR,
    PROCESSED_DATA_DIR,
    TEST_START,
    TRAIN_START,
    VALIDATION_START,
)
from mibici.features import validar_columnas_viajes
from mibici.features_p3 import columnas_perfil_p3
from mibici.history_p3 import cohorte_historia

AUG_MIN_DAYS = 14
AUG_MIN_MOVEMENTS = 100


def eventos_periodo(viajes, inicio, fin):
    validar_columnas_viajes(viajes)
    partes = []
    for tipo, id_col, time_col in (
        ("salidas", "origen_id", "inicio_viaje"),
        ("llegadas", "destino_id", "fin_viaje"),
    ):
        fechas = pd.to_datetime(viajes[time_col], errors="raise")
        if fechas.isna().any() or fechas.dt.tz is not None:
            raise ValueError("Timestamps locales validos sin zona requeridos")
        mask = fechas.ge(pd.Timestamp(inicio)) & fechas.lt(pd.Timestamp(fin))
        if viajes.loc[mask, id_col].isna().any():
            raise ValueError("Evento sin estacion")
        partes.append(
            pd.DataFrame(
                {
                    "estacion_id": viajes.loc[mask, id_col].to_numpy(),
                    "momento": fechas[mask].to_numpy(),
                    "tipo": tipo,
                }
            )
        )
    eventos = pd.concat(partes, ignore_index=True)
    eventos["fecha"] = eventos.momento.dt.normalize()
    return eventos


def perfiles_agosto(viajes, entrenadas, *, inicio=VALIDATION_START, fin=TEST_START):
    """Proxy mensual completo; defaults agosto, mismos umbrales en septiembre.

    El nombre eventos_agosto_1 se conserva por compatibilidad; al evaluar otro
    periodo representa su primer dia y el consumidor debe renombrarlo.
    """
    if (
        entrenadas.primer_evento.isna().any()
        or not pd.to_datetime(entrenadas.primer_evento).lt(pd.Timestamp(VALIDATION_START)).all()
        or entrenadas.estacion_id.duplicated().any()
    ):
        raise ValueError("Requiere estaciones unicas observadas antes de agosto")
    ev = eventos_periodo(viajes, inicio, fin)
    calendario = pd.date_range(inicio, pd.Timestamp(fin) - pd.Timedelta(days=1))
    denominadores = np.bincount(calendario.dayofweek, minlength=7)
    grupos = dict(tuple(ev.groupby("estacion_id")))
    filas = []
    for meta in entrenadas.itertuples():
        e = grupos.get(meta.estacion_id, ev.iloc[:0])
        dias = e.fecha.drop_duplicates()
        weekdays = np.bincount(dias.dt.dayofweek.to_numpy(dtype=int), minlength=7)
        razones = []
        if len(dias) < AUG_MIN_DAYS:
            razones.append("menos_14_dias_con_eventos")
        if len(e) < AUG_MIN_MOVEMENTS:
            razones.append("menos_100_movimientos")
        if (weekdays == 0).any():
            razones.append("weekday_sin_eventos")
        row = {
            "estacion_id": meta.estacion_id,
            "nombre": meta.nombre,
            "conjunto": "val" if inicio == VALIDATION_START else "test",
            "periodo_inicio": str(inicio),
            "periodo_fin_exclusivo": str(fin),
            "dias_con_eventos": len(dias),
            "dias_denominador_proxy": len(calendario),
            "total_movimientos": len(e),
            "eventos_agosto_1": int(e.fecha.eq(pd.Timestamp(inicio)).sum()),
            "cobertura_fuente_diaria": "desconocida",
            "disponibilidad_estacion": "desconocida",
        }
        for dia in range(7):
            row[f"dias_proxy_d{dia}"] = int(denominadores[dia])
            row[f"dias_con_eventos_d{dia}"] = int(weekdays[dia])
        for tipo in ("salidas", "llegadas"):
            sub = e[e.tipo.eq(tipo)]
            counts = np.zeros((7, 24))
            grouped = sub.groupby([sub.momento.dt.dayofweek, sub.momento.dt.hour]).size()
            for (dia, hora), n in grouped.items():
                counts[int(dia), int(hora)] = n
            medias = counts / denominadores[:, None]
            forma = medias / medias.sum() if medias.sum() > 0 else np.full((7, 24), np.nan)
            for dia in range(7):
                for hora in range(24):
                    row[f"{tipo}_d{dia}_h{hora:02d}"] = forma[dia, hora]
            row[f"total_{tipo}"] = len(sub)
            if sub.empty:
                razones.append(f"sin_{tipo}")
        row["evaluable"] = not razones
        row["motivo_exclusion"] = ";".join(razones)
        filas.append(row)
    return pd.DataFrame(filas)


def asignar_fijo(modelo, perfiles):
    """Solo predict/transform: nunca ajusta centroides a validacion."""
    x = matriz_forma(perfiles)
    labels = modelo.predict(x)
    dist = modelo.transform(x)
    return x, labels, dist[np.arange(len(x)), labels]


def leer_verificado(nombre, records):
    path = PROCESSED_DATA_DIR / nombre
    with path.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    if sha != records[nombre]["sha256"]:
        raise ValueError(f"SHA no coincide: {nombre}")
    frame = pd.read_parquet(
        path, columns=["viaje_id", "inicio_viaje", "fin_viaje", "origen_id", "destino_id"]
    )
    if len(frame) != records[nombre]["records"]:
        raise ValueError(f"Conteo no coincide: {nombre}")
    return frame, {"archivo": nombre, "registros": len(frame), "sha256": sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preparacion", type=Path)
    parser.add_argument("anterior", type=Path)
    parser.add_argument("salida", type=Path)
    args = parser.parse_args()
    if (
        not (args.salida / "PROTOCOLO_PREVIO.txt").is_file()
        or (args.salida / "resumen.json").exists()
    ):
        raise ValueError("Requiere protocolo previo y salida no ejecutada")
    train = cohorte_historia(pd.read_parquet(args.preparacion), 180)
    x = matriz_forma(train)
    modelo = KMeans(
        n_clusters=3, random_state=SEED, n_init=N_INIT, max_iter=300, algorithm="lloyd"
    ).fit(x)
    prev = pd.read_csv(args.anterior / "dias_ge_180/asignaciones.csv")
    if not np.array_equal(train.estacion_id, prev.estacion_id) or not np.array_equal(
        modelo.labels_, prev.kmeans_k3
    ):
        raise ValueError("No reproduce modelo tentativo anterior")
    centros = modelo.cluster_centers_.copy()
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
    manifest = json.loads((MANIFESTS_DIR / "mibici_processed.json").read_text(encoding="utf-8"))
    records = {r["filename"]: r for r in manifest["files"]}
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
    diario.to_csv(args.salida / "grupo_domingo_eventos_diarios_train.csv", index=False)
    recurrence = []
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
        pd.DataFrame(months).to_csv(
            args.salida / f"recurrencia_mensual_estacion_{station.estacion_id}.csv", index=False
        )
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
    pd.DataFrame(recurrence).to_csv(args.salida / "grupo_domingo_recurrencia.csv", index=False)
    miembros.to_csv(args.salida / "grupo_domingo_miembros_train.csv", index=False)
    agosto, check = leer_verificado("datos_abiertos_2026_08.parquet", records)
    integridad.append(check)
    val = perfiles_agosto(pd.concat([julio, agosto], ignore_index=True), train)
    val.to_parquet(args.salida / "perfiles_agosto.parquet", index=False)
    val.loc[
        ~val.evaluable,
        ["estacion_id", "nombre", "dias_con_eventos", "total_movimientos", "motivo_exclusion"],
    ].to_csv(args.salida / "excluidas_agosto.csv", index=False)
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
    asignaciones.to_csv(args.salida / "asignaciones_agosto.csv", index=False)
    pd.crosstab(asignaciones.grupo_train, asignaciones.grupo_agosto).reindex(
        index=range(3), columns=range(3), fill_value=0
    ).to_csv(args.salida / "transiciones.csv")
    guardar_grafica(
        q,
        labels,
        "agosto_asignacion_centroides_train",
        args.salida,
        contexto="agosto: centroides ajustados en train",
    )
    guardar_grafica(matched, train_labels, "train_mismos_ids_evaluables", args.salida)
    grupos = []
    for g in range(3):
        ids = np.flatnonzero(labels == g)
        if not len(ids):
            continue
        order = ids[np.argsort(distances[ids], kind="stable")[:3]]
        reps = asignaciones.iloc[order].copy()
        reps.to_csv(args.salida / f"representantes_agosto_grupo_{g}.csv", index=False)
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
    pd.DataFrame(grupos).to_csv(args.salida / "metricas_por_grupo.csv", index=False)
    if not np.array_equal(centros, modelo.cluster_centers_):
        raise AssertionError("Centroides modificados")
    joblib.dump(modelo, args.salida / "kmeans_train_tentativo.joblib")
    pd.DataFrame(centros, columns=columnas_perfil_p3()).to_csv(
        args.salida / "centroides_transformados_train.csv", index=False
    )
    pd.DataFrame(integridad).to_csv(args.salida / "integridad_fuentes.csv", index=False)
    train.assign(grupo_train=modelo.labels_)[["estacion_id", "nombre", "grupo_train"]].to_csv(
        args.salida / "asignaciones_train.csv", index=False
    )
    resumen = {
        "cohorte_primaria": ">=180 dias train",
        "n_train": len(train),
        "n_evaluables_agosto": len(q),
        "n_excluidas_agosto": len(val) - len(q),
        "grupo_domingo_train": grupo_dom,
        "n_grupo_domingo_train": len(miembros),
        "tamanos_train": np.bincount(modelo.labels_, minlength=3).tolist(),
        "tamanos_agosto": np.bincount(labels, minlength=3).tolist(),
        "fraccion_estable": float((labels == train_labels).mean()),
        "ari_train_val": float(adjusted_rand_score(train_labels, labels)),
        "silhouette_val_asignacion_fija": float(silhouette_score(xv, labels))
        if 2 <= len(set(labels)) < len(labels)
        else None,
        "cambio_perfil_pareado_mediana": float(np.median(paired_shift)),
        "cambio_perfil_pareado_p95": float(np.quantile(paired_shift, 0.95)),
        "distancia_val_mediana": float(np.median(distances)),
        "fraccion_val_sobre_p95_train_asignado": float((distances > p95[labels]).mean()),
        "centroides_sin_refit": True,
        "septiembre_leido": False,
        "seleccion_final": None,
        "umbral_agosto_dias": AUG_MIN_DAYS,
        "umbral_agosto_movimientos": AUG_MIN_MOVEMENTS,
        "llegadas_limite_izquierdo_desde_julio": len(julio),
        "eventos_agosto1_estaciones_train": int(val.eventos_agosto_1.sum()),
    }
    (args.salida / "resumen.json").write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    print(json.dumps(resumen, indent=2), flush=True)
    print(pd.DataFrame(grupos).to_string(index=False), flush=True)
    print(pd.DataFrame(recurrence).describe().to_string(), flush=True)


if __name__ == "__main__":
    main()
