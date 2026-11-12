"""Local residual-demand models, selected on September only."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from mibici.accuracy_monitoring import metrics
from mibici.features import VARIABLES_P1, dataset_p1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--original-predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/improved-p1-local-residual"))
    args = parser.parse_args()
    source = args.source
    out = args.out
    out.mkdir(exist_ok=False)
    protocol = {
        "design": "Predict correction to prior seven-day average, then clip total demand at zero",
        "candidates": ["squared_error", "absolute_error"],
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 50,
        "l2_regularization": 10,
        "learning_rate": 0.05,
        "selection": "September WAPE; do not choose loss using October",
        "disclosure": "Iterative retrospective development; October inspected previously, not pristine holdout",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    panel = dataset_p1(
        pd.concat(
            [pd.read_parquet(p) for p in sorted(source.glob("datos_abiertos_*.parquet"))],
            ignore_index=True,
        ),
        pd.read_parquet(source / "estaciones.parquet"),
    )
    train = panel.loc[panel.fecha < "2026-09-01"]
    val = panel.loc[panel.fecha >= "2026-09-01"]

    def fit(loss, frame):
        m = HistGradientBoostingRegressor(
            loss=loss,
            max_iter=300,
            max_leaf_nodes=31,
            min_samples_leaf=50,
            l2_regularization=10,
            learning_rate=0.05,
            early_stopping=False,
            random_state=42,
        )
        with threadpool_limits(limits=4):
            m.fit(frame[VARIABLES_P1].astype("float64"), frame.salidas - frame.salidas_media7)
        return m

    def predict(m, frame):
        from mibici.residual_model import predict_residual_demand

        return predict_residual_demand(m, frame)

    results = []
    for loss in protocol["candidates"]:
        m = fit(loss, train)
        r = {"loss": loss, "validation": metrics(val.assign(prediction=predict(m, val)))}
        results.append(r)
        print(json.dumps(r), flush=True)
    selected = min(results, key=lambda r: r["validation"]["wape"])
    (out / "selection.json").write_text(
        json.dumps({"candidates": results, "selected": selected}, indent=2)
    )
    # Stop if this design does not improve on previously validated raw HGB; no October fishing.
    if selected["validation"]["wape"] >= 0.20007477000606802:
        print("No validation improvement; no October evaluation", flush=True)
    else:
        final = fit(selected["loss"], panel)
        current = pd.read_parquet(args.current)
        original = pd.read_parquet(args.original_predictions)
        assert current[["estacion_id", "fecha"]].equals(original[["estacion_id", "fecha"]])
        prediction_column = "old" if "old" in original else "prediction"
        current["old"] = original[prediction_column].to_numpy()
        current["new"] = predict(final, current)
        old = metrics(current.assign(prediction=current.old))
        new = metrics(current.assign(prediction=current.new))
        result = {
            "protocol": protocol,
            "selected": selected,
            "original": old,
            "candidate": new,
            "fit_n": len(panel),
            "evaluation_n": len(current),
            "no_github_write": True,
            "no_registry_write": True,
        }
        current[["estacion_id", "fecha", "salidas", "old", "new"]].to_parquet(
            out / "predictions.parquet", index=False
        )
        import skops.io

        skops.io.dump(final, out / "residual_estimator.skops")
        restored = skops.io.load(
            out / "residual_estimator.skops",
            trusted=[
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
                "numpy.dtype",
            ],
        )
        np.testing.assert_allclose(predict(restored, current), current.new, rtol=1e-12, atol=1e-12)
        result["roundtrip_verified"] = True
        (out / "comparison.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
