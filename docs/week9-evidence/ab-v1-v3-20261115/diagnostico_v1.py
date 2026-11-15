"""Diagnóstico de la diferencia de v1 frente al reporte; NO genera resultados del A/B.

Uso desde la raíz, con el entorno de requirements-accuracy.txt:
    python docs/week9-evidence/ab-v1-v3-20261115/diagnostico_v1.py \
        --current RUTA/current.parquet --original-model RUTA_EXPORT_V1 \
        --out docs/week9-evidence/ab-v1-v3-20261115/diagnostico-v1.json

Mide cuántas latitudes de estaciones coinciden exactamente con umbrales de división de
v1 y v3, y cómo cambian las métricas si `latitud` se desplaza un ulp. Las entradas
alteradas existen solo en memoria para medir sensibilidad; no se guardan ni se usan
como predicciones del A/B.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import skops.io

from mibici.ab_testing import load_v3, sha256
from mibici.accuracy_monitoring import metrics
from mibici.features import VARIABLES_P1
from mibici.residual_model import predict_residual_demand

KEYS = ["mae", "rmse", "wape", "bias_prediction_minus_actual"]


def thresholds(estimator, column):
    j = VARIABLES_P1.index(column)
    return np.unique(
        np.concatenate(
            [
                p.nodes["num_threshold"][(p.nodes["feature_idx"] == j) & (p.nodes["is_leaf"] == 0)]
                for iteration in estimator._predictors
                for p in iteration
            ]
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--original-model", required=True, type=Path)
    parser.add_argument("--reference", default=Path("docs/week9-evidence/comparison.json"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    current = pd.read_parquet(args.current)
    registry = json.loads((args.original_model / "registry.json").read_text("utf-8"))
    artifact = args.original_model / "model.skops"
    if sha256(artifact) != registry["archivos"]["model.skops"]:
        raise ValueError("v1 export integrity mismatch")
    # Same trust decision as mlflow.sklearn loading of the verified export.
    v1 = skops.io.load(artifact, trusted=skops.io.get_untrusted_types(file=artifact))
    v3, _, _ = load_v3(Path("models/week9-v3"))
    x = current[VARIABLES_P1].astype("float64")
    station_lat = current.groupby("estacion_id").latitud.first()
    shifts = {
        "sin_cambio": lambda v: v,
        "mas_1_ulp": lambda v: np.nextafter(v, np.inf),
        "menos_1_ulp": lambda v: np.nextafter(v, -np.inf),
    }
    result = {
        "current_sha256": sha256(args.current),
        "v1_model_sha256": registry["archivos"]["model.skops"],
        "reference": {
            "original": {k: json.loads(args.reference.read_text())["original"][k] for k in KEYS},
            "candidate": {k: json.loads(args.reference.read_text())["candidate"][k] for k in KEYS},
        },
        "models": {},
    }
    base = {"v1": v1.predict(x), "v3": predict_residual_demand(v3, current)}
    for name, est in [("v1", v1[-1] if hasattr(v1, "steps") else v1), ("v3", v3)]:
        thr = thresholds(est, "latitud")
        entry = {
            "latitud_thresholds": int(thr.size),
            "stations_with_latitud_equal_to_threshold": int(np.isin(station_lat, thr).sum()),
            "stations_one_ulp_below_threshold": int(
                np.isin(np.nextafter(station_lat.to_numpy(), np.inf), thr).sum()
            ),
            "metrics": {},
        }
        for label, shift in shifts.items():
            moved = current.copy()
            moved["latitud"] = shift(moved["latitud"].to_numpy(dtype=float))
            if name == "v1":
                pred = v1.predict(moved[VARIABLES_P1].astype("float64"))
            else:
                pred = predict_residual_demand(v3, moved)
            m = metrics(current.assign(prediction=pred))
            entry["metrics"][label] = {
                "rows_changed": int((pred != base[name]).sum()),
                **{k: m[k] for k in KEYS},
            }
        result["models"][name] = entry
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
