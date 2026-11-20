"""Controlador ligero de decisiones CT basado en evidencias existentes.

No entrena modelos, no modifica MLflow y no promueve versiones.
"""
import argparse
import json
import math
from pathlib import Path


def load_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def evaluate(evidence_dir):
    evidently = load_json(evidence_dir / "evidently.json")
    ab = load_json(
        evidence_dir / "ab-v1-v3-20261115" / "ab_results.json"
    )

    drift_metrics = [
        item["result"]
        for item in evidently["metrics"]
        if item["metric"] == "DatasetDriftMetric"
    ]
    if len(drift_metrics) != 1:
        raise ValueError("Se requiere exactamente un DatasetDriftMetric")

    drift = drift_metrics[0]
    total = drift["number_of_columns"]
    affected = drift["number_of_drifted_columns"]

    if not (
        isinstance(total, int)
        and isinstance(affected, int)
        and total > 0
        and 0 <= affected <= total
    ):
        raise ValueError("Conteos de drift invalidos")

    if not isinstance(drift["dataset_drift"], bool):
        raise ValueError("Indicador dataset_drift invalido")

    intervals = ab["paired_bootstrap_95pct_intervals"]
    metrics = ("mae", "rmse", "wape")
    interval_results = {}

    for metric in metrics:
        values = intervals[metric]
        if metric == "wape":
            values = values["proportion"]

        low = float(values["low"])
        high = float(values["high"])

        if not (math.isfinite(low) and math.isfinite(high) and low <= high):
            raise ValueError(f"Intervalo invalido: {metric}")

        interval_results[metric] = {
            "low": low,
            "high": high,
            "improvement_supported": high < 0,
        }

    limitations = ab.get("limitations", [])
    if not isinstance(limitations, list) or not limitations:
        raise ValueError("Faltan limitaciones de la evaluacion A/B")

    reference = ab.get("reference_comparison", {})
    reference_consistent = reference.get("all_within_tolerance") is True

    # Una mejora en las tres metricas debe estar respaldada por
    # intervalos completamente negativos y evidencia reproducible.
    promotion_supported = (
        all(x["improvement_supported"] for x in interval_results.values())
        and reference_consistent
        and not limitations
    )

    # El reporte actual contiene limitaciones retrospectivas.
    # Aun con intervalos favorables, CT no promueve automaticamente:
    # cualquier promocion requiere validacion temporal independiente.

    return {
        "evidence": "week9",
        "drift": {
            "detected": drift["dataset_drift"],
            "affected_columns": affected,
            "total_columns": total,
            "share": affected / total,
        },
        "retraining": {
            "review_required": drift["dataset_drift"],
            "automatic_execution": False,
            "decision": (
                "REVISAR_REENTRENAMIENTO"
                if drift["dataset_drift"]
                else "CONTINUAR_MONITOREO"
            ),
        },
        "ab_intervals": interval_results,
        "promotion": {
            "automatic_execution": False,
            "supported": promotion_supported,
            "decision": (
                "EVALUAR_PROMOCION_MANUAL"
                if promotion_supported
                else "CONSERVAR_V1"
            ),
            "reason": (
                "Se requiere validacion temporal independiente; "
                "el A/B es retrospectivo y sus limitaciones impiden "
                "una promocion automatica."
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path("docs/week9-evidence"),
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    try:
        result = evaluate(args.evidence_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"ERROR CT: {exc}\n")

    output = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
