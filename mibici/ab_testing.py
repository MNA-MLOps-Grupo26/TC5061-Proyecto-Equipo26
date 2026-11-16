"""Paired A/B comparison of P1 original v1 versus residual candidate v3 on October.

Two subcommands, both run from the repository root:

``predict``
    Builds ``predictions.parquet`` (``estacion_id``, ``fecha``, ``salidas``, ``old``,
    ``new``) from the outputs of ``mibici.accuracy_monitoring`` (``current.parquet`` and
    ``october_predictions.parquet`` of the verified v1 export) and the portable v3 copy in
    ``models/week9-v3``. The v3 estimator only predicts residuals, so demand is obtained
    with ``predict_residual_demand`` (adds ``salidas_media7`` and clips at zero).

``compare``
    Validates ``predictions.parquet`` and writes ``ab_results.json`` and ``ab_table.md``
    with MAE, RMSE, WAPE and bias per model, v3 - v1 differences and the paired
    moving-block bootstrap intervals of ``challenger_monitoring.paired_intervals``.

No model is fitted, registered or promoted.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pandas as pd

from mibici.accuracy_monitoring import metrics
from mibici.challenger_monitoring import paired_intervals

REQUIRED_COLUMNS = ["estacion_id", "fecha", "salidas", "old", "new"]
MODELS = {"old": "original_v1", "new": "candidate_v3"}
# Parameters fixed inside challenger_monitoring.paired_intervals; recorded, not tunable here.
BOOTSTRAP = {
    "method": "paired moving-block bootstrap over calendar dates",
    "block_length_days": 7,
    "repetitions": 2000,
    "seed": 42,
    "rng": "numpy.random.default_rng",
    "interval": "percentile 95% (quantiles 0.025 and 0.975)",
    "unit": "date; all stations of a date stay together",
    "pairing": "same resampled dates for both models in every replicate",
    "difference": "candidate_v3 minus original_v1",
    "implementation": "mibici.challenger_monitoring.paired_intervals",
}
V3_TRUSTED_TYPES = [
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "numpy.dtype",
]


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def validate(frame):
    """Return a validated copy sorted by date and station; raise ValueError otherwise."""
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if frame.empty:
        raise ValueError("Empty predictions")
    f = frame[REQUIRED_COLUMNS].copy()
    if f.estacion_id.isna().any():
        raise ValueError("Missing station id")
    try:
        f["fecha"] = pd.to_datetime(f.fecha, errors="raise")
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid dates") from error
    if f.fecha.isna().any():
        raise ValueError("Missing dates")
    if getattr(f.fecha.dt, "tz", None) is not None:
        raise ValueError("Dates must be timezone-naive calendar days")
    if (f.fecha != f.fecha.dt.normalize()).any():
        raise ValueError("Dates must be calendar days without time")
    if f.duplicated(["estacion_id", "fecha"]).any():
        raise ValueError("Duplicate station-date key")
    for column in ["salidas", "old", "new"]:
        try:
            values = f[column].to_numpy(dtype=float)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Non-numeric {column}") from error
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite {column}")
        f[column] = values
    if (f.salidas < 0).any():
        raise ValueError("Negative actual demand")
    if f.salidas.sum() <= 0:
        raise ValueError("WAPE needs positive total demand")
    dates = pd.DatetimeIndex(sorted(f.fecha.unique()))
    if len(dates) < BOOTSTRAP["block_length_days"]:
        raise ValueError("At least seven dates required for seven-day blocks")
    expected = pd.date_range(dates[0], dates[-1], freq="D")
    if not dates.equals(expected):
        gaps = expected.difference(dates)
        raise ValueError(f"Dates must be consecutive days; missing {list(gaps.date)}")
    return f.sort_values(["fecha", "estacion_id"], kind="mergesort").reset_index(drop=True)


def wape_views(proportion):
    return {"proportion": proportion, "percent": 100 * proportion}


def compare(frame, provenance=None):
    f = validate(frame)
    per_model = {}
    for column, label in MODELS.items():
        m = metrics(f.assign(prediction=f[column]))
        per_model[label] = {
            "mae": m["mae"],
            "rmse": m["rmse"],
            "wape": wape_views(m["wape"]),
            "bias_prediction_minus_actual": m["bias_prediction_minus_actual"],
            "actual_mean": m["actual_mean"],
        }
    old, new = per_model["original_v1"], per_model["candidate_v3"]
    difference = {
        "mae": new["mae"] - old["mae"],
        "rmse": new["rmse"] - old["rmse"],
        "wape": {
            "proportion": new["wape"]["proportion"] - old["wape"]["proportion"],
            "percentage_points": new["wape"]["percent"] - old["wape"]["percent"],
        },
        "bias_prediction_minus_actual": new["bias_prediction_minus_actual"]
        - old["bias_prediction_minus_actual"],
    }
    raw = paired_intervals(f, repetitions=BOOTSTRAP["repetitions"])
    intervals = {
        "mae": raw["mae"],
        "rmse": raw["rmse"],
        "wape": {
            "proportion": raw["wape"],
            "percentage_points": {k: 100 * v for k, v in raw["wape"].items()},
        },
        "bias": "not computed (not part of the published method)",
    }
    dates = f.fecha.drop_duplicates()
    return {
        "design": "Paired comparison of two models on identical station-date rows",
        "data": {
            "n": len(f),
            "dates": int(dates.size),
            "first_date": str(dates.min().date()),
            "last_date": str(dates.max().date()),
            "stations": int(f.estacion_id.nunique()),
            "actual_total": float(f.salidas.sum()),
        },
        "metrics": per_model,
        "candidate_minus_original": difference,
        "paired_bootstrap_95pct_intervals": intervals,
        "bootstrap": BOOTSTRAP,
        "units": {
            "mae_rmse_bias": "trips per station-day",
            "wape.proportion": "sum|error| / sum(actual), 0-1 scale",
            "wape.percent": "proportion x 100",
            "wape.percentage_points": "difference of percents (pp)",
            "bias": "mean(prediction - actual); positive = overprediction",
        },
        "provenance": provenance or {},
        "limitations": [
            (
                "Retrospective evaluation over a single month (October 2026), already inspected "
                "during drift/accuracy work; not a pristine holdout."
            ),
            (
                "Intervals are exploratory: only about four or five independent weekly blocks, "
                "dependent observations and nonstationarity."
            ),
            (
                "Original trained through July 2026, candidate through September 2026: algorithm "
                "and training-history changes are confounded."
            ),
            "Daily one-step rolling observed history, not a full-month forecast.",
        ],
    }


def against_reference(result, reference, tolerance=1e-9):
    """Computed minus published values (week9 comparison.json layout); never edits either."""
    m, ci = result["metrics"], result["paired_bootstrap_95pct_intervals"]
    computed = {}
    for label, key in [("original_v1", "original"), ("candidate_v3", "candidate")]:
        for metric in ["mae", "rmse", "wape", "bias_prediction_minus_actual"]:
            value = m[label][metric]
            computed[f"{key}.{metric}"] = value["proportion"] if metric == "wape" else value
    for metric in ["mae", "rmse", "wape"]:
        bounds = ci[metric]["proportion"] if metric == "wape" else ci[metric]
        for side in ["low", "high"]:
            computed[f"intervals.{metric}.{side}"] = bounds[side]
    rows = {}
    for name, value in computed.items():
        node = reference
        for part in name.split("."):
            node = node[part]
        rows[name] = {
            "computed": value,
            "reference": node,
            "difference": value - node,
            "within_tolerance": abs(value - node) <= tolerance,
        }
    if "n" in reference.get("original", {}):
        n, ref_n = result["data"]["n"], reference["original"]["n"]
        rows["original.n"] = {"computed": n, "reference": ref_n, "within_tolerance": n == ref_n}
    return {
        "tolerance_abs": tolerance,
        "wape_scale": "proportion",
        "all_within_tolerance": all(r["within_tolerance"] for r in rows.values()),
        "values": rows,
    }


def markdown(result):
    m, d = result["metrics"], result["candidate_minus_original"]
    ci = result["paired_bootstrap_95pct_intervals"]
    o, c = m["original_v1"], m["candidate_v3"]
    data = result["data"]

    def interval(v, fmt="{:.4f}"):
        return f"[{fmt.format(v['low'])}, {fmt.format(v['high'])}]"

    rows = [
        "| Métrica | Original v1 | Candidata v3 | v3 − v1 | IC 95% de v3 − v1 |",
        "|---|---:|---:|---:|---|",
        f"| MAE | {o['mae']:.4f} | {c['mae']:.4f} | {d['mae']:.4f} | {interval(ci['mae'])} |",
        (
            f"| RMSE | {o['rmse']:.4f} | {c['rmse']:.4f} | {d['rmse']:.4f} | "
            f"{interval(ci['rmse'])} |"
        ),
        (
            f"| WAPE | {o['wape']['percent']:.4f}% | {c['wape']['percent']:.4f}% | "
            f"{d['wape']['percentage_points']:.4f} pp | "
            f"{interval(ci['wape']['percentage_points'])} pp |"
        ),
        (
            f"| Sesgo (predicción − real) | {o['bias_prediction_minus_actual']:.4f} | "
            f"{c['bias_prediction_minus_actual']:.4f} | {d['bias_prediction_minus_actual']:.4f} | "
            "No calculado |"
        ),
    ]
    b = result["bootstrap"]
    notes = [
        "",
        (
            f"{data['n']} filas estación-día, {data['stations']} estaciones, {data['dates']} fechas "
            f"({data['first_date']} a {data['last_date']})."
        ),
        (
            f"Bootstrap pareado por bloques móviles de {b['block_length_days']} días, "
            f"{b['repetitions']} repeticiones, semilla {b['seed']}, percentiles 2.5/97.5; "
            "mismos bloques para ambos modelos; diferencias v3 − v1. Intervalos exploratorios."
        ),
        "WAPE en porcentaje; su diferencia e intervalo en puntos porcentuales (pp).",
    ]
    return "\n".join(rows + notes) + "\n"


def load_v3(folder):
    """Load the portable v3 estimator after checking its SHA-256 against the manifest."""
    import skops.io

    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        artifact = folder / manifest["artifact"]
        if not artifact.exists():
            # Extract outside the tracked folder so no model binary appears in Git status.
            with zipfile.ZipFile(folder / manifest["archive"]) as z:
                if z.namelist() != [manifest["artifact"]]:
                    raise ValueError("Unexpected archive content")
                artifact = Path(z.extract(manifest["artifact"], tmp))
        actual = sha256(artifact)
        if actual != manifest["sha256"]:
            raise ValueError("v3 artifact SHA-256 mismatch")
        untrusted = set(skops.io.get_untrusted_types(file=artifact))
        if not untrusted <= set(V3_TRUSTED_TYPES):
            extra = sorted(untrusted - set(V3_TRUSTED_TYPES))
            raise ValueError(f"Unexpected skops types: {extra}")
        estimator = skops.io.load(artifact, trusted=V3_TRUSTED_TYPES)
    return estimator, manifest, actual


def predict(current_path, original_path, candidate_folder, original_model=None):
    from mibici.residual_model import predict_residual_demand

    current = pd.read_parquet(current_path)
    original = pd.read_parquet(original_path)
    keys = ["estacion_id", "fecha"]
    if not current[keys].reset_index(drop=True).equals(original[keys].reset_index(drop=True)):
        raise ValueError("current and original predictions are not aligned by station-date")
    if not np.array_equal(current.salidas.to_numpy(), original.salidas.to_numpy()):
        raise ValueError("Actual demand differs between inputs")
    estimator, manifest, digest = load_v3(candidate_folder)
    frame = current[keys + ["salidas"]].copy()
    frame["old"] = original["prediction"].to_numpy(dtype=float)
    frame["new"] = predict_residual_demand(estimator, current)
    provenance = {
        "current_sha256": sha256(current_path),
        "original_predictions_sha256": sha256(original_path),
        "original_predictions_source": "mibici.accuracy_monitoring with verified v1 export",
        "candidate": {
            "registered_model_name": manifest["registered_model_name"],
            "source_registry_version": manifest["source_registry_version"],
            "source_run_id": manifest["source_run_id"],
            "artifact_sha256": digest,
            "loader": "skops.io.load with fixed allowlist "
            "+ mibici.residual_model.predict_residual_demand",
            "trusted_types": V3_TRUSTED_TYPES,
        },
    }
    if original_model is not None:
        registry = json.loads((Path(original_model) / "registry.json").read_text("utf-8"))
        for name, expected in registry["archivos"].items():
            if sha256(Path(original_model) / name) != expected:
                raise ValueError(f"v1 export integrity mismatch: {name}")
        provenance["original"] = {
            "registered_model_name": registry["modelo"]["nombre"],
            "version": registry["modelo"]["version"],
            "run_id": registry["modelo"]["run_id"],
            "model_sha256": registry["archivos"]["model.skops"],
        }
    return frame, provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict", help="Build predictions.parquet for v1 and v3")
    p.add_argument(
        "--current", required=True, type=Path, help="accuracy_monitoring current.parquet"
    )
    p.add_argument(
        "--original-predictions",
        required=True,
        type=Path,
        help="accuracy_monitoring october_predictions.parquet (v1)",
    )
    p.add_argument("--original-model", type=Path, help="v1 export folder (registry.json)")
    p.add_argument("--candidate", default=Path("models/week9-v3"), type=Path)
    p.add_argument("--out", required=True, type=Path, help="Output folder (outside Git)")
    c = sub.add_parser("compare", help="Metrics, v3 - v1 differences and paired intervals")
    c.add_argument("--predictions", required=True, type=Path)
    c.add_argument("--provenance", type=Path, help="JSON written by predict")
    c.add_argument(
        "--reference",
        type=Path,
        help="Published comparison JSON (e.g. docs/week9-evidence/comparison.json) to diff against",
    )
    c.add_argument(
        "--tolerance", type=float, default=1e-9, help="Absolute tolerance for --reference"
    )
    c.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "predict":
        frame, provenance = predict(
            args.current, args.original_predictions, args.candidate, args.original_model
        )
        validate(frame)
        frame.to_parquet(args.out / "predictions.parquet", index=False)
        provenance["predictions_sha256"] = sha256(args.out / "predictions.parquet")
        (args.out / "provenance.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8"
        )
        print(json.dumps(provenance, indent=2))
        return
    provenance = {"predictions_sha256": sha256(args.predictions)}
    if args.provenance:
        recorded = json.loads(args.provenance.read_text(encoding="utf-8"))
        if recorded.get("predictions_sha256") not in (None, provenance["predictions_sha256"]):
            raise ValueError("Provenance does not describe these predictions")
        provenance = {**recorded, **provenance}
    result = compare(pd.read_parquet(args.predictions), provenance)
    if args.reference:
        result["reference_comparison"] = against_reference(
            result, json.loads(args.reference.read_text(encoding="utf-8")), args.tolerance
        )
        result["reference_comparison"]["file"] = args.reference.as_posix()
    (args.out / "ab_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    table = markdown(result)
    (args.out / "ab_table.md").write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
