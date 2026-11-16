"""Rastrea las latitudes del CSV del catálogo hasta el panel y su efecto en v1/v3.

Uso desde la raíz, con el entorno de requirements-accuracy.txt (los Parquet no están en Git):
    python docs/week9-evidence/coordenadas-20261115/diagnostico_coordenadas.py \
        --catalogo data/raw/nomenclatura_2026_09.csv \
        --panel-reporte RUTA/current.parquet      # panel compartido por Kim
        --panel-local RUTA/current.parquet        # panel generado con --coordenadas pandas
        --original-model RUTA_EXPORT_V1 \
        --out docs/week9-evidence/coordenadas-20261115/diagnostico-coordenadas.json

Solo lee archivos. Las sustituciones de columnas existen en memoria para medir efecto.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import skops.io

from mibici.ab_testing import load_v3, sha256
from mibici.accuracy_monitoring import metrics
from mibici.coordenadas import convertir
from mibici.features import VARIABLES_P1
from mibici.residual_model import predict_residual_demand


def iguales(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (a == b) | (np.isnan(a) & np.isnan(b))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--catalogo", required=True, type=Path)
    p.add_argument("--panel-reporte", required=True, type=Path)
    p.add_argument("--panel-local", required=True, type=Path)
    p.add_argument("--original-model", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    keys = ["estacion_id", "fecha"]
    rep = pd.read_parquet(a.panel_reporte)
    loc = pd.read_parquet(a.panel_local)
    rep = rep.set_index(keys).loc[pd.MultiIndex.from_frame(loc[keys])].reset_index()
    resultado = {
        "entorno": {
            "plataforma": platform.platform(),
            "maquina": platform.machine(),
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
        },
        "sha256": {
            "catalogo_csv": sha256(a.catalogo),
            "panel_reporte": sha256(a.panel_reporte),
            "panel_local": sha256(a.panel_local),
        },
    }
    # 1. Columnas que difieren entre paneles.
    diferencias = {}
    for col in [c for c in loc.columns if c not in keys]:
        if loc[col].dtype.kind in "fiub":
            d = ~iguales(loc[col], rep[col])
            if d.any():
                diferencias[col] = {
                    "filas": int(d.sum()),
                    "estaciones": int(loc.loc[d, "estacion_id"].nunique()),
                    "max_abs": float(
                        np.abs(loc[col].to_numpy(float) - rep[col].to_numpy(float))[d].max()
                    ),
                }
        elif not loc[col].astype(str).equals(rep[col].astype(str)):
            diferencias[col] = "no numérica distinta"
    resultado["columnas_distintas_entre_paneles"] = diferencias
    resultado["dtype_fecha"] = {"reporte": str(rep.fecha.dtype), "local": str(loc.fecha.dtype)}
    # 2. Texto del CSV -> float con cada conversión.
    texto = {}
    with a.catalogo.open(encoding="utf-8-sig", newline="") as f:
        for fila in csv.DictReader(f):
            texto.setdefault(int(fila["id"]), fila["latitude"])
    lat_rep = rep.groupby("estacion_id").latitud.first()
    lat_loc = loc.groupby("estacion_id").latitud.first()
    ids = [s for s in lat_rep.index if s in texto and pd.notna(lat_rep[s])]
    leido = pd.read_csv(a.catalogo, encoding="utf-8-sig").drop_duplicates("id").set_index("id")
    conversiones = {
        "pandas read_csv (este equipo)": leido.latitude,
        "Python float() (redondeo correcto)": pd.Series({s: float(texto[s]) for s in ids}),
        "precise_xstrtod sin FMA": pd.Series({s: convertir(texto[s], "sin_fma") for s in ids}),
        "precise_xstrtod con FMA": pd.Series({s: convertir(texto[s], "con_fma") for s in ids}),
    }
    resultado["conversion_csv_a_float"] = {
        "estaciones": len(ids),
        "coincidencias": {
            n: {
                "panel_reporte": int(iguales(c.reindex(ids), lat_rep.reindex(ids)).sum()),
                "panel_local": int(iguales(c.reindex(ids), lat_loc.reindex(ids)).sum()),
            }
            for n, c in conversiones.items()
        },
    }
    distintas = [s for s in ids if lat_rep[s] != lat_loc[s]]
    resultado["estaciones_con_latitud_distinta"] = distintas
    # 3. Umbrales de los modelos y efecto en predicciones.
    art = a.original_model / "model.skops"
    registry = json.loads((a.original_model / "registry.json").read_text("utf-8"))
    if sha256(art) != registry["archivos"]["model.skops"]:
        raise ValueError("v1 export integrity mismatch")
    v1 = skops.io.load(art, trusted=skops.io.get_untrusted_types(file=art))
    v3, _, _ = load_v3(Path("models/week9-v3"))
    j = VARIABLES_P1.index("latitud")
    umbrales = {}
    for nombre, est in [("v1_oficial", v1), ("v3", v3)]:
        thr = np.unique(
            np.concatenate(
                [
                    n.nodes["num_threshold"][
                        (n.nodes["feature_idx"] == j) & (n.nodes["is_leaf"] == 0)
                    ]
                    for it in est._predictors
                    for n in it
                ]
            )
        )
        umbrales[nombre] = {
            "umbral_en_valor_reporte": [s for s in distintas if lat_rep[s] in thr],
            "umbral_en_valor_local": [s for s in distintas if lat_loc[s] in thr],
        }
    resultado["umbrales_latitud_en_estaciones_distintas"] = umbrales
    X = lambda f: f[VARIABLES_P1].astype("float64")  # noqa: E731
    sust = loc.assign(latitud=rep.latitud.to_numpy())
    p_loc, p_rep, p_sust = v1.predict(X(loc)), v1.predict(X(rep)), v1.predict(X(sust))
    q_loc, q_rep = predict_residual_demand(v3, loc), predict_residual_demand(v3, rep)
    resultado["efecto_en_predicciones"] = {
        "v1_predicciones_distintas": int((p_loc != p_rep).sum()),
        "v1_panel_local_con_latitud_del_reporte_igual_al_reporte": bool(
            np.array_equal(p_sust, p_rep)
        ),
        "v3_predicciones_distintas": int((q_loc != q_rep).sum()),
        "v1_mae": {
            "panel_local": metrics(loc.assign(prediction=p_loc))["mae"],
            "panel_reporte": metrics(rep.assign(prediction=p_rep))["mae"],
        },
        "huella_predicciones_v1_reporte": hashlib.sha256(
            np.asarray(p_rep, "<f8").tobytes()
        ).hexdigest(),
    }
    a.out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
