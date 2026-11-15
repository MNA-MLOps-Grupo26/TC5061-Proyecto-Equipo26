"""Prueba E2E desde cero: octubre verificado y v1 reentrenado para el A/B v1 frente a v3.

Pasos previos: ``python -m mibici.dataset`` con data/raw y data/processed vacíos.

    python scripts/e2e_ab_v1_v3.py october --out e2e/raw-october
    python scripts/e2e_ab_v1_v3.py train-v1 --out e2e/v1-reentrenado --summary e2e/entrenamiento-v1.json

``train-v1`` reproduce el ajuste del diagnóstico del PR #26 (HGB con P1_HIPERPARAMETROS y
semilla 42 sobre ``conjunto == "train"``, agosto 2025 a julio 2026). Guarda el modelo con
``mlflow.sklearn.save_model`` en una carpeta local, junto con un ``registry.json`` con sus
hashes. Es el formato que aceptan ``mibici.accuracy_monitoring`` y ``mibici.ab_testing``.
No usa ni modifica ningún registro MLflow. Es un v1 REENTRENADO, no la exportación oficial.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

OCTOBER_URL = "https://mibici.net/site/assets/files/133188/datos_abiertos_2026_10.csv"
OCTOBER_SHA256 = "0cdfb05299e6db658ac9cc7723494a9704b0749e3671c29c8566826ea889b900"
OFFICIAL_V1_SHA256 = "2cac5ef209865bf0eb70a388c594dfd4791dfab73b51088802a3df07dbb09fae"
TRUSTED = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor", "numpy.dtype"]


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def october(out):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "datos_abiertos_2026_10.csv"
    request = urllib.request.Request(OCTOBER_URL, headers={"User-Agent": "mibici-e2e"})
    with urllib.request.urlopen(request, timeout=300) as response, path.open("wb") as f:
        while chunk := response.read(1 << 20):
            f.write(chunk)
    if sha(path) != OCTOBER_SHA256:
        raise RuntimeError("El CSV de octubre no coincide con el SHA-256 de la evidencia")
    print("Octubre verificado:", OCTOBER_SHA256, flush=True)


def train_v1(out, summary_path):
    import mlflow
    import mlflow.sklearn
    import numpy as np

    from mibici.config import P1_HIPERPARAMETROS, RANDOM_SEED
    from mibici.dataset import cargar_estaciones, cargar_viajes
    from mibici.features import VARIABLES_P1, dataset_p1
    from mibici.modeling.p1 import calcular_metricas, construir_modelo

    if out.exists():
        raise RuntimeError("La carpeta de salida ya existe")
    panel = dataset_p1(cargar_viajes(), cargar_estaciones())
    train = panel.loc[panel["conjunto"] == "train"]
    inputs = train[VARIABLES_P1].assign(salidas=train["salidas"])
    digest = mlflow.data.from_pandas(
        inputs, name="p1-train-estacion-dia", targets="salidas"
    ).digest
    model = construir_modelo("hgb", P1_HIPERPARAMETROS)
    started = time.monotonic()
    model.fit(train[VARIABLES_P1], train["salidas"])
    seconds = time.monotonic() - started
    metrics = {}
    for label, partition in [("train", "train"), ("val", "validation"), ("test", "test")]:
        subset = panel.loc[panel["conjunto"] == partition]
        pred = model.predict(subset[VARIABLES_P1])
        metrics.update({f"{label}_{k}": v for k, v in calcular_metricas(subset, pred).items()})
    mlflow.sklearn.save_model(
        model, str(out), serialization_format="skops", skops_trusted_types=TRUSTED
    )
    files = sorted(p for p in out.rglob("*") if p.is_file())
    registry = {
        "modelo": {
            "nombre": "mibici-equipo26-p1-demand-reentrenado-e2e",
            "version": "reentrenado-e2e",
            "run_id": "sin-registro",
            "nota": "Reentrenado desde cero en esta corrida; no es la exportación oficial v1",
        },
        "archivos": {p.relative_to(out).as_posix(): sha(p) for p in files},
    }
    (out / "registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    baseline = Path("docs/evidence/2026-11-15-catalogo-historico/resultados.json")
    historical = json.loads(baseline.read_text(encoding="utf-8"))["reproduccion_p1"]["training"][
        "comparison"
    ]
    model_sha = registry["archivos"]["model.skops"]
    summary = {
        "mlflow_training_digest": digest,
        "historical_digest": "10f3651e",
        "mac_reproduction_digest": "e3b4786b",
        "training_rows": len(train),
        "fit_seconds": seconds,
        "hyperparameters": {**P1_HIPERPARAMETROS, "random_state": RANDOM_SEED},
        "model_skops_sha256": model_sha,
        "official_v1_model_skops_sha256": OFFICIAL_V1_SHA256,
        "model_bytes_equal_official_v1": model_sha == OFFICIAL_V1_SHA256,
        "metrics_vs_historical": {
            k: {
                "historical": historical[k]["original"],
                "current": v,
                "abs_difference": abs(v - historical[k]["original"]),
            }
            for k, v in metrics.items()
        },
        "max_abs_metric_difference": float(
            np.max([abs(v - historical[k]["original"]) for k, v in metrics.items()])
        ),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def summary(root, out):
    """Reúne resultados y compara con la evidencia de Kim (Windows); no edita referencias."""
    import platform
    import sys

    def load(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    mine = load(root / "accuracy-p1" / "accuracy.json")
    kim = load("reports/accuracy-p1/accuracy.json")
    accuracy = {}
    for period in ["september", "october"]:
        for metric in ["mae", "rmse", "wape", "bias_prediction_minus_actual"]:
            a = mine["cohorts"]["all_stations"][period][metric]
            b = kim["cohorts"]["all_stations"][period][metric]
            accuracy[f"{period}.{metric}"] = {"e2e": a, "kim": b, "difference": a - b}
    drift = load(root / "accuracy-p1" / "summary.json")
    ab = load(root / "ab" / "ab_results.json")
    result = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "v1": "reentrenado desde cero en esta corrida (no la exportación oficial)",
        "training": load(root / "entrenamiento-v1.json"),
        "catalog_manifest_mismatch": drift["catalog_manifest_mismatch"],
        "catalog_sha256": drift["catalog_sha256"],
        "accuracy_all_stations_vs_kim": accuracy,
        "ab_reference_comparison": ab["reference_comparison"],
        "ab_metrics": ab["metrics"],
        "ab_intervals": ab["paired_bootstrap_95pct_intervals"],
        "latitud_diagnostic": load(root / "diagnostico-v1.json")["models"],
    }
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["ab_reference_comparison"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    o = sub.add_parser("october")
    o.add_argument("--out", required=True, type=Path)
    t = sub.add_parser("train-v1")
    t.add_argument("--out", required=True, type=Path)
    t.add_argument("--summary", required=True, type=Path)
    s = sub.add_parser("summary")
    s.add_argument("--root", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "october":
        october(args.out)
    elif args.command == "train-v1":
        train_v1(args.out, args.summary)
    else:
        summary(args.root, args.out)


if __name__ == "__main__":
    main()
