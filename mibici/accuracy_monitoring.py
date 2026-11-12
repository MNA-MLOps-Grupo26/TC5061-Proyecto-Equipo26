"""Evaluate the unchanged exported P1 model on September and October; never fit."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mibici.features import VARIABLES_P1, dataset_p1
from mibici.monitoring import prepare, validate_windows


def metrics(frame):
    if frame.empty:
        raise ValueError("Empty evaluation cohort")
    y = frame["salidas"].to_numpy(dtype=float)
    p = frame["prediction"].to_numpy(dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Nonfinite target or prediction")
    e = p - y
    if y.sum() <= 0:
        raise ValueError("WAPE needs positive total demand")
    return {
        "n": len(frame),
        "stations": int(frame.estacion_id.nunique()),
        "days": int(frame.fecha.nunique()),
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "wape": float(np.abs(e).sum() / y.sum()),
        "bias_prediction_minus_actual": float(e.mean()),
        "actual_mean": float(y.mean()),
    }


def block_interval(september, october, seed=42, repetitions=2000):
    """Exploratory moving 7-day-block bootstrap; resample all stations together."""

    def daily(frame):
        f = frame.assign(
            ae=(frame.prediction - frame.salidas).abs(), se=(frame.prediction - frame.salidas) ** 2
        )
        return (
            f.groupby("fecha")
            .agg(ae=("ae", "sum"), se=("se", "sum"), y=("salidas", "sum"), n=("salidas", "size"))
            .to_numpy()
        )

    arrays = [daily(september), daily(october)]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repetitions):
        estimates = []
        for a in arrays:
            n = len(a)
            if n < 7:
                raise ValueError("At least seven dates required")
            starts = rng.integers(0, n - 7 + 1, size=int(np.ceil(n / 7)))
            ix = np.concatenate([np.arange(s, s + 7) for s in starts])[:n]
            ae, se, y, count = a[ix].sum(axis=0)
            estimates.append([ae / count, np.sqrt(se / count), ae / y])
        deltas.append(np.subtract(estimates[1], estimates[0]))
    bounds = np.quantile(deltas, [0.025, 0.975], axis=0)
    return {
        name: {"low": float(bounds[0, i]), "high": float(bounds[1, i])}
        for i, name in enumerate(["mae", "rmse", "wape"])
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--october", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", default=Path("reports/accuracy-p1"), type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # Revalidate inputs and rebuild rather than trusting cached panels.
    _, current, _ = prepare(args.source, args.october, args.out)
    manifest = json.loads(Path("manifests/mibici_processed.json").read_text())
    trips = pd.concat(
        [pd.read_parquet(args.source / i["filename"]) for i in manifest["files"]],
        ignore_index=True,
    )
    stations = pd.read_parquet(args.source / "estaciones.parquet")
    previous = dataset_p1(trips, stations)
    previous = previous.loc[
        previous.fecha.between("2026-09-01", "2026-10-01", inclusive="left")
    ].copy()
    validate_windows(previous, current)
    registry = json.loads((args.model / "registry.json").read_text(encoding="utf-8"))
    for name, expected in registry["archivos"].items():
        actual = hashlib.sha256((args.model / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Model integrity mismatch: {name}")
    import mlflow.sklearn

    model = mlflow.sklearn.load_model(str(args.model))
    if list(model.feature_names_in_) != VARIABLES_P1:
        raise ValueError("Model feature order mismatch")
    for frame in (previous, current):
        frame["prediction"] = model.predict(frame[VARIABLES_P1].astype("float64"))
        frame["baseline"] = frame.salidas_mismo_dia_media4.fillna(frame.salidas_media7).fillna(0)
    common = set(previous.estacion_id) & set(current.estacion_id)
    # Primary comparison fixes station composition; full cohorts retained transparently.
    cohorts = {
        "all_stations": (previous, current),
        "common_stations": (
            previous.loc[previous.estacion_id.isin(common)],
            current.loc[current.estacion_id.isin(common)],
        ),
    }
    results = {
        "model": registry["modelo"],
        "model_sha256": registry["archivos"]["model.skops"],
        "method": "Unchanged model; daily one-step rolling observed history; no fitting or tuning. September previously examined. October minus September deltas.",
        "cohorts": {},
    }
    for name, (old, new) in cohorts.items():
        a, b = metrics(old), metrics(new)
        results["cohorts"][name] = {
            "september": a,
            "october": b,
            "delta": {k: b[k] - a[k] for k in ["mae", "rmse", "wape"]},
            "baseline_september": metrics(
                old.drop(columns="prediction").rename(columns={"baseline": "prediction"})
            ),
            "baseline_october": metrics(
                new.drop(columns="prediction").rename(columns={"baseline": "prediction"})
            ),
        }
    a, b = cohorts["common_stations"]
    results["exploratory_95pct_7day_block_intervals"] = block_interval(a, b)
    results["uncertainty_limit"] = (
        "Only 30/31 dates, dependent observations, nonstationarity and calendar composition. Exploratory intervals, not causal proof or a formal independent-sample significance test."
    )
    # Same weekdays reweighted to September frequency as a sensitivity check.
    weights = a.fecha.dt.dayofweek.value_counts(normalize=True).sort_index()
    weighted = {}
    for label, f in [("september", a), ("october", b)]:
        values = []
        for day, w in weights.items():
            d = f.loc[f.fecha.dt.dayofweek == day]
            e = d.prediction - d.salidas
            values.append((w, e.abs().mean(), (e**2).mean(), d.salidas.mean()))
        z = np.asarray(values)
        mae = float(np.sum(z[:, 0] * z[:, 1]))
        mse = float(np.sum(z[:, 0] * z[:, 2]))
        actual = float(np.sum(z[:, 0] * z[:, 3]))
        weighted[label] = {"mae": mae, "rmse": float(np.sqrt(mse)), "wape": mae / actual}
    results["weekday_reweighted_common_stations"] = weighted
    (args.out / "accuracy.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for label, f in [("september", previous), ("october", current)]:
        f[["estacion_id", "fecha", "salidas", "prediction", "baseline"]].to_parquet(
            args.out / f"{label}_predictions.parquet", index=False
        )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
