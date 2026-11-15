"""Compara entornos sin publicar datasets ni modificar el registro MLflow."""

import argparse
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def frame_fingerprint(frame):
    import numpy as np
    import pandas as pd

    columns = {}
    for name in frame.columns:
        series = frame[name]
        hashes = pd.util.hash_pandas_object(series, index=False).to_numpy(dtype="<u8")
        info = {
            "dtype": str(series.dtype),
            "nulls": int(series.isna().sum()),
            "values_pandas_sha256": hashlib.sha256(hashes.tobytes()).hexdigest(),
        }
        if pd.api.types.is_numeric_dtype(series):
            values = series.to_numpy(dtype="<f8", na_value=np.nan).copy()
            values[np.isnan(values)] = np.nan
            info["numeric_float64_sha256"] = hashlib.sha256(values.tobytes()).hexdigest()
            rounded = np.round(values, decimals=10)
            info["rounded_10_decimals_sha256"] = hashlib.sha256(rounded.tobytes()).hexdigest()
        columns[name] = info
    index = pd.util.hash_pandas_object(frame.index).to_numpy(dtype="<u8")
    rows = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype="<u8")
    return {
        "rows": len(frame),
        "column_order": list(frame.columns),
        "index_dtype": str(frame.index.dtype),
        "index_sha256": hashlib.sha256(index.tobytes()).hexdigest(),
        "full_frame_pandas_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
        "columns": columns,
    }


def run(args, result):
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    import mlflow
    from mlflow.data.digest_utils import compute_pandas_digest
    import numpy as np
    import pandas as pd

    from mibici import preprocess
    from mibici.config import P1_HIPERPARAMETROS, RANDOM_SEED
    from mibici.dataset import cargar_estaciones, cargar_viajes
    from mibici.features import VARIABLES_P1, dataset_p1
    from mibici.modeling.p1 import calcular_metricas, construir_modelo

    result["environment"] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "pandas", "pyarrow", "mlflow", "scikit-learn")
        },
        "threads": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    result["code_sha256"] = {
        str(p.relative_to(repo)).replace("\\", "/"): sha(p)
        for p in [
            repo / "mibici" / f
            for f in (
                "config.py",
                "dataset.py",
                "preprocess.py",
                "features.py",
                "tracking.py",
                "modeling/p1.py",
            )
        ]
    }
    result["digest_function_sha256"] = hashlib.sha256(
        inspect.getsource(compute_pandas_digest).encode()
    ).hexdigest()
    original = json.loads((repo / "manifests/mibici_processed.json").read_text())
    downloads = json.loads((repo / "manifests/mibici_downloads.json").read_text())
    if args.ingest:
        if list((repo / "data/raw").glob("*.csv")) or list(
            (repo / "data/processed").glob("*.parquet")
        ):
            raise RuntimeError("La prueba desde cero requiere data/raw y data/processed vacíos")
        subprocess.run([sys.executable, "-m", "mibici.dataset"], cwd=repo, check=True)
    result["ingest_from_empty_directories"] = args.ingest
    result["raw_files"] = []
    for item in downloads:
        observed = sha(repo / "data/raw" / item["filename"])
        result["raw_files"].append(
            {
                "filename": item["filename"],
                "sha256": observed,
                "matches": observed == item["sha256"],
            }
        )
    if not all(x["matches"] for x in result["raw_files"]):
        raise RuntimeError("Los CSV no coinciden con los hashes congelados")
    current = json.loads((repo / "manifests/mibici_processed.json").read_text())
    result["data_version"] = current["data_version"]
    result["processed_files"] = []
    records = [*original["files"], original["stations"]]
    for item in records:
        observed = sha(repo / "data/processed" / item["filename"])
        result["processed_files"].append(
            {
                "filename": item["filename"],
                "sha256": observed,
                "expected_sha256": item["sha256"],
                "matches": observed == item["sha256"],
            }
        )

    stations = cargar_estaciones()
    # Regenerar únicamente en un directorio temporal: no altera los datos existentes.
    station_source = next(x for x in downloads if x["filename"].startswith("nomenclatura_"))
    with tempfile.TemporaryDirectory(prefix="catalogo-diagnostico-") as temp:
        old_dir = preprocess.PROCESSED_DATA_DIR
        try:
            preprocess.PROCESSED_DATA_DIR = Path(temp)
            preprocess.normalizar_estaciones(
                repo / "data/raw" / station_source["filename"], station_source["encoding"]
            )
            path = Path(temp) / "estaciones.parquet"
            rebuilt = pd.read_parquet(path)
            result["station_parquet_regenerated_sha256"] = sha(path)
        finally:
            preprocess.PROCESSED_DATA_DIR = old_dir
    pd.testing.assert_frame_equal(stations, rebuilt, check_exact=True)
    result["station_regeneration_exact_values"] = True
    result["stations"] = frame_fingerprint(rebuilt)
    print("Catálogo verificado; construyendo variables P1", flush=True)
    panel = dataset_p1(cargar_viajes(), rebuilt)
    train = panel.loc[panel["conjunto"] == "train"]
    inputs = train[VARIABLES_P1].assign(salidas=train["salidas"])
    result["training_frame"] = frame_fingerprint(inputs)
    result["mlflow_training_digest"] = mlflow.data.from_pandas(
        inputs, name="p1-train-estacion-dia", targets="salidas"
    ).digest
    result["historical_digest"] = "10f3651e"
    result["mac_reproduction_digest"] = "e3b4786b"
    result["hyperparameters"] = {**P1_HIPERPARAMETROS, "random_state": RANDOM_SEED}
    print("Digest:", result["mlflow_training_digest"], "Entrenando HGB P1", flush=True)
    model = construir_modelo("hgb", P1_HIPERPARAMETROS)
    started = time.monotonic()
    model.fit(train[VARIABLES_P1], train["salidas"])
    result["fit_seconds"] = time.monotonic() - started
    result["metrics"] = {}
    result["prediction_sha256"] = {}
    for label, partition in [("train", "train"), ("val", "validation"), ("test", "test")]:
        subset = panel.loc[panel["conjunto"] == partition]
        pred = model.predict(subset[VARIABLES_P1])
        result["metrics"].update(
            {f"{label}_{k}": v for k, v in calcular_metricas(subset, pred).items()}
        )
        result["prediction_sha256"][label] = hashlib.sha256(
            np.asarray(pred, dtype="<f8").tobytes()
        ).hexdigest()
    baseline_path = repo / "docs/evidence/2026-11-15-catalogo-historico/resultados.json"
    historical = json.loads(baseline_path.read_text())["reproduccion_p1"]["training"]["comparison"]
    result["historical_metrics_comparison"] = {
        k: {
            "historical": historical[k]["original"],
            "current": value,
            "abs_difference": abs(value - historical[k]["original"]),
        }
        for k, value in result["metrics"].items()
    }
    examples = json.loads((repo / "docs/evidencia-api/peticion-batch.json").read_text())[
        "registros"
    ]
    example_frame = pd.DataFrame(examples)[VARIABLES_P1].astype("float64")
    result["retrained_model_example_predictions"] = model.predict(example_frame).tolist()
    result["scope"] = (
        "Diagnóstico de ingesta, representación y reentrenamiento P1; sin registro, exportación ni API. No reconstruye el entorno histórico de Windows."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ingest", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = {"started_utc": datetime.now(UTC).isoformat(), "status": "running"}
    started = time.monotonic()
    try:
        run(args, result)
        result["status"] = "completed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print("Resultado:", args.output, flush=True)


if __name__ == "__main__":
    main()
