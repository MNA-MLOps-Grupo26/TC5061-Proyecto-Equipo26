"""Sensibilidad exploratoria P3 a historia registrada, sin usar holdouts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from mibici.compare_p3 import comparar, guardar_grafica
from mibici.config import P3_COHORTES, TRAIN_START, VALIDATION_START
from mibici.features_p3 import columnas_perfil_p3

COHORTES = P3_COHORTES


def cohorte_historia(perfiles, minimo):
    if minimo not in COHORTES:
        raise ValueError("Umbral fuera del protocolo predeclarado")
    if not (
        perfiles.conjunto.eq("train").all()
        and perfiles.periodo_inicio.eq(TRAIN_START.isoformat()).all()
        and perfiles.periodo_fin_exclusivo.eq(VALIDATION_START.isoformat()).all()
    ):
        raise ValueError("Solo entrenamiento")
    if perfiles.estacion_id.duplicated().any():
        raise ValueError("IDs duplicados")
    return (
        perfiles.loc[perfiles.elegible & perfiles.dias_con_eventos.ge(minimo)]
        .sort_values("estacion_id")
        .reset_index(drop=True)
    )


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entrada", type=Path)
    parser.add_argument("anterior", type=Path)
    parser.add_argument("salida", type=Path)
    args = parser.parse_args()
    if args.salida.exists():
        raise FileExistsError("La salida debe ser nueva")
    p = pd.read_parquet(args.entrada)
    original = pd.read_csv(args.anterior / "asignaciones.csv")
    base = p.loc[p.elegible].sort_values("estacion_id").reset_index(drop=True)
    if set(base.estacion_id) != set(original.estacion_id):
        raise ValueError("Asignaciones anteriores no corresponden a elegibles")
    args.salida.mkdir(parents=True)
    tablas, contexto, patrones, conteos, asociaciones = [], [], [], [], []
    for minimo in COHORTES:
        q = cohorte_historia(p, minimo)
        prefijo = f"dias_ge_{minimo}"
        destino = args.salida / prefijo
        destino.mkdir()
        tabla, etiquetas, reps = comparar(q)
        tabla.insert(0, "cohorte", prefijo)
        tabla.insert(1, "n_estaciones", len(q))
        tabla.to_csv(destino / "comparacion.csv", index=False)
        reps.to_csv(destino / "representantes.csv", index=False)
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
            guardar_grafica(q, labels, f"{prefijo}_{nombre}", destino)
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
        tabla.to_csv(destino / "comparacion.csv", index=False)
        tablas.append(tabla)
        asignaciones.to_csv(destino / "asignaciones.csv", index=False)
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
    pd.concat(tablas).to_csv(args.salida / "comparacion_cohortes.csv", index=False)
    pd.DataFrame(contexto).to_csv(args.salida / "contexto_grupos.csv", index=False)
    pd.DataFrame(patrones).to_csv(args.salida / "resumen_patrones.csv", index=False)
    pd.DataFrame(conteos).to_csv(args.salida / "conteos.csv", index=False)
    pd.DataFrame(asociaciones).to_csv(args.salida / "acuerdo_metodos.csv", index=False)
    b = base[["estacion_id", "dias_con_eventos"]].merge(original, on="estacion_id")
    asociacion_base = {
        c: eta_historia(b.dias_con_eventos, b[c].to_numpy())
        for c in original
        if c not in ("estacion_id", "nombre")
    }
    with args.entrada.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    (args.salida / "protocolo.json").write_text(
        json.dumps(
            {
                "cohortes_predeclaradas_dias_activos": COHORTES,
                "entrada": str(args.entrada.resolve()),
                "sha256": sha,
                "comparacion_previa": str(args.anterior.resolve()),
                "protocolo": "mismo comparar(): KMeans seed42 n_init20 max_iter300, Ward, k2-6",
                "distancia": "euclidea sqrt(p)/sqrt(2); 336 variables, peso igual por direccion",
                "eta2_historia_original": asociacion_base,
                "seleccion_final": None,
                "holdouts_leidos": False,
                "limite": "Filtrar poblacion no mejora cobertura ni demuestra causalidad",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
