"""Register frozen residual model in isolated local comparison registry only."""

import argparse
import json
from pathlib import Path

import mlflow
from mlflow.models import infer_signature
import numpy as np
import pandas as pd

from mibici.accuracy_monitoring import metrics
from mibici.features import VARIABLES_P1


class ResidualDemandModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import skops.io

        self.estimator = skops.io.load(
            context.artifacts["estimator"],
            trusted=[
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
                "numpy.dtype",
            ],
        )

    def predict(self, context, model_input, params=None):
        from mibici.residual_model import predict_residual_demand

        return predict_residual_demand(self.estimator, model_input)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path("reports/improved-p1-local-residual"))
    args = parser.parse_args()
    store = args.store.resolve()
    assert store.name == "TC5061-mlflow-comparison-20261110-final"
    mlflow.set_tracking_uri("sqlite:///" + (store / "mlflow.db").as_posix())
    mlflow.set_registry_uri(mlflow.get_tracking_uri())
    client = mlflow.MlflowClient()
    name = "mibici-equipo26-p1-demand"
    versions = list(client.search_model_versions(f"name='{name}'"))
    assert sorted(int(v.version) for v in versions) == [1, 2], "Inspect registry before rerun"
    experiment = client.get_run(
        next(v for v in versions if str(v.version) == "1").run_id
    ).info.experiment_id
    base = args.base
    current = pd.read_parquet(args.current)
    expected = pd.read_parquet(base / "predictions.parquet")
    assert current[["estacion_id", "fecha"]].equals(expected[["estacion_id", "fecha"]])
    x = current[VARIABLES_P1].astype("float64")
    with mlflow.start_run(
        experiment_id=experiment, run_name="residual_hgb_verified_20261111"
    ) as run:
        mlflow.set_tags(
            {
                "scope": "isolated_local_comparison_not_canonical",
                "purpose": "improved_residual_challenger",
                "evaluation": "exploratory_rolling_one_day_ahead",
                "fit_through": "2026-09-30",
                "selection_month": "2026-09",
                "october_previously_inspected": "true",
            }
        )
        mlflow.log_params(json.loads((base / "protocol.json").read_text()))
        mlflow.log_metrics(
            {
                k: v
                for k, v in metrics(current.assign(prediction=expected.new)).items()
                if isinstance(v, (float, int))
            }
        )
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=ResidualDemandModel(),
            artifacts={"estimator": str((base / "residual_estimator.skops").resolve())},
            code_paths=["mibici"],
            signature=infer_signature(x, expected.new.to_numpy()),
            input_example=x.head(5),
            pip_requirements=[
                "mlflow==3.17.2",
                "scikit-learn==1.9.1",
                "numpy==2.5.4",
                "pandas==3.0.6",
                "skops==0.16.0",
                "threadpoolctl==3.8.0",
                "loguru==0.7.3",
                "python-dotenv==1.2.4",
            ],
        )
        version = mlflow.register_model(info.model_uri, name)
        client.set_model_version_tag(
            name, version.version, "scope", "isolated_local_comparison_not_canonical"
        )
        client.set_model_version_tag(
            name, version.version, "purpose", "improved_residual_challenger"
        )
        loaded = mlflow.pyfunc.load_model(f"models:/{name}/{version.version}")
        prediction = loaded.predict(x)
        np.testing.assert_allclose(prediction, expected.new, rtol=1e-12, atol=1e-12)
        result = {
            "name": name,
            "version": version.version,
            "run_id": run.info.run_id,
            "model_uri": info.model_uri,
            "tracking_uri": mlflow.get_tracking_uri(),
            "scope": "isolated_local_comparison_not_canonical",
            "prior_versions_preserved": [1, 2],
            "aliases_modified": False,
            "signature": loaded.metadata.signature.to_dict(),
            "max_abs_difference": float(np.max(abs(prediction - expected.new))),
            "metrics": metrics(current.assign(prediction=prediction)),
        }
        (base / "registry-verification-20261111.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
