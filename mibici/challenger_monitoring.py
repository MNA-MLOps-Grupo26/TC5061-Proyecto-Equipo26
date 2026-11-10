"""Fixed Poisson challenger; October never used for fitting or tuning."""

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mibici.accuracy_monitoring import metrics
from mibici.features import VARIABLES_P1, dataset_p1
from mibici.monitoring import prepare

NAME = "mibici-equipo26-p1-demand"


def paired_intervals(frame, repetitions=2000):
    f = frame.assign(
        oldae=(frame.old - frame.salidas).abs(),
        oldse=(frame.old - frame.salidas) ** 2,
        newae=(frame.new - frame.salidas).abs(),
        newse=(frame.new - frame.salidas) ** 2,
    )
    a = (
        f.groupby("fecha")
        .agg(
            oldae=("oldae", "sum"),
            oldse=("oldse", "sum"),
            newae=("newae", "sum"),
            newse=("newse", "sum"),
            y=("salidas", "sum"),
            n=("salidas", "size"),
        )
        .to_numpy()
    )
    if len(a) < 7:
        raise ValueError("At least seven dates required")
    rng = np.random.default_rng(42)
    changes = []
    for _ in range(repetitions):
        starts = rng.integers(0, len(a) - 6, size=int(np.ceil(len(a) / 7)))
        ix = np.concatenate([np.arange(s, s + 7) for s in starts])[: len(a)]
        oa, os, na, ns, y, n = a[ix].sum(axis=0)
        changes.append([(na - oa) / n, np.sqrt(ns / n) - np.sqrt(os / n), (na - oa) / y])
    bounds = np.quantile(changes, [0.025, 0.975], axis=0)
    return {
        k: {"low": float(bounds[0, i]), "high": float(bounds[1, i])}
        for i, k in enumerate(["mae", "rmse", "wape"])
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--october", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--store", required=True, type=Path)
    p.add_argument("--resume-original-only", action="store_true")
    p.add_argument("--out", default=Path("reports/challenger-p1"), type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    _, current, _ = prepare(args.source, args.october, args.out)
    manifest = json.loads(Path("manifests/mibici_processed.json").read_text())
    trips = pd.concat(
        [pd.read_parquet(args.source / i["filename"]) for i in manifest["files"]],
        ignore_index=True,
    )
    train = dataset_p1(trips, pd.read_parquet(args.source / "estaciones.parquet"))
    if train.fecha.max() >= current.fecha.min():
        raise ValueError("Temporal leakage")
    if train.duplicated(["estacion_id", "fecha"]).any():
        raise ValueError("Duplicate training key")
    registry = json.loads((args.model / "registry.json").read_text())
    for name, expected in registry["archivos"].items():
        if hashlib.sha256((args.model / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Model integrity mismatch")
    import mlflow
    from mlflow.models import infer_signature
    import mlflow.sklearn

    original = mlflow.sklearn.load_model(str(args.model))
    # Existing repo GLM default: alpha=1.0, max_iter=1000. Frozen before new fit.
    candidate = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        PoissonRegressor(alpha=1.0, max_iter=1000),
    )
    xtrain = train[VARIABLES_P1].astype("float64")
    xtest = current[VARIABLES_P1].astype("float64")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        candidate.fit(xtrain, train.salidas)
    current["old"] = original.predict(xtest)
    current["new"] = candidate.predict(xtest)
    evaluated = {}
    for key in ["old", "new"]:
        evaluated[key] = metrics(current.assign(prediction=current[key]))
    result = {
        "model_name": NAME,
        "original_run_id": registry["modelo"]["run_id"],
        "original_sha256": registry["archivos"]["model.skops"],
        "challenger": "Median imputation + StandardScaler + PoissonRegressor(alpha=1.0,max_iter=1000)",
        "train_window": ["2025-08-01", "2026-10-01"],
        "train_n": len(train),
        "train_last_date": str(train.fecha.max()),
        "evaluation_window": ["2026-10-01", "2026-11-01"],
        "evaluation_n": len(current),
        "metrics": evaluated,
        "candidate_minus_original": {
            k: evaluated["new"][k] - evaluated["old"][k] for k in ["mae", "rmse", "wape"]
        },
        "paired_exploratory_95pct_intervals": paired_intervals(current),
        "candidate_n_iter": int(candidate[-1].n_iter_),
        "limitations": [
            "October was already inspected in drift/accuracy diagnostics; held out from fitting, not pristine confirmatory evidence.",
            "No hyperparameter search or repeated candidate selection against October.",
            "Original trained through July; challenger trained through September. Algorithm and training-history changes are confounded.",
            "Daily one-step rolling observed history, not monthly forecast.",
            "Single month; moving-week paired bootstrap is exploratory.",
        ],
    }
    (args.out / "comparison-before-registration.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    # New dedicated registry, never the canonical team database; do not replace aliases.
    if args.store.exists() and not args.resume_original_only:
        raise ValueError("Store exists; inspect before explicit original-only resume")
    args.store.mkdir(parents=True, exist_ok=args.resume_original_only)
    mlflow.set_tracking_uri("sqlite:///" + (args.store / "mlflow.db").resolve().as_posix())
    client = mlflow.MlflowClient()
    if args.resume_original_only:
        prior = list(client.search_model_versions(f"name='{NAME}'"))
        if (
            len(prior) != 1
            or str(prior[0].version) != "1"
            or prior[0].tags.get("purpose") != "original_export_copy"
        ):
            raise ValueError("Resume requires exactly original-copy version 1")
        previous_run = client.get_run(prior[0].run_id)
        if previous_run.data.tags.get("origin_champion_run") != registry["modelo"]["run_id"]:
            raise ValueError("Original lineage mismatch")
        experiment = previous_run.info.experiment_id
    else:
        experiment = client.create_experiment(
            "mibici-equipo26-comparison-20261110",
            artifact_location=(args.store / "artifacts").resolve().as_uri(),
        )
    registrations = []
    for label, model, values in [
        ("original_export_copy", original, evaluated["old"]),
        ("poisson_challenger", candidate, evaluated["new"]),
    ]:
        if label == "original_export_copy" and args.resume_original_only:
            registrations.append(
                {
                    "label": label,
                    "version": "1",
                    "run_id": prior[0].run_id,
                    "model_uri": prior[0].source,
                }
            )
            continue
        with mlflow.start_run(experiment_id=experiment, run_name=label) as run:
            mlflow.set_tags(
                {
                    "registry_scope": "isolated_local_comparison_not_canonical_team_store",
                    "origin_champion_run": registry["modelo"]["run_id"],
                    "evaluation_period": "2026-10",
                    "fit_scope": "train_through_2026-09-30"
                    if label == "poisson_challenger"
                    else "unchanged_export_train_through_2026-07-31",
                    "october_used_for_fit": "false",
                    "october_previously_inspected": "true",
                }
            )
            mlflow.log_metrics({k: v for k, v in values.items() if isinstance(v, (int, float))})
            if label == "poisson_challenger":
                mlflow.log_params(
                    {
                        "alpha": 1.0,
                        "max_iter": 1000,
                        "imputer": "median_train_only",
                        "scaler": "train_only",
                        "training_rows": len(train),
                    }
                )
            info = mlflow.sklearn.log_model(
                model,
                name="model",
                serialization_format="skops",
                skops_trusted_types=[
                    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
                    "numpy.dtype",
                ]
                if label == "original_export_copy"
                else ["numpy.dtype"],
                signature=infer_signature(xtest, model.predict(xtest)),
                input_example=xtest.head(5),
            )
            version = mlflow.register_model(info.model_uri, NAME)
            client.set_model_version_tag(NAME, version.version, "purpose", label)
            registrations.append(
                {
                    "label": label,
                    "version": version.version,
                    "run_id": run.info.run_id,
                    "model_uri": info.model_uri,
                }
            )
    result["local_registry"] = {
        "scope": "Isolated local comparison; not canonical team registry",
        "versions": registrations,
        "champion_alias_changed": False,
    }
    (args.out / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    current[["estacion_id", "fecha", "salidas", "old", "new"]].to_parquet(
        args.out / "predictions.parquet", index=False
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
