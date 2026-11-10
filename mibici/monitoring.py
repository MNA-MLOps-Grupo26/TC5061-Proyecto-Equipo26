"""Reproducible P1 monitoring; no fitting or registry changes."""

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path

import pandas as pd

from mibici import features
from mibici.preprocess import COLUMNAS, COLUMNAS_ESTACIONES

CATEGORICAL = ["dia_semana", "es_fin_semana", "mes"]


def validate_windows(reference, current):
    for frame in (reference, current):
        if frame.empty:
            raise ValueError("Empty period")
        if frame.duplicated(["estacion_id", "fecha"]).any():
            raise ValueError("Duplicate station-day")
    if reference.fecha.max() >= current.fecha.min():
        raise ValueError("Reference/current periods overlap")


def coverage(trips, start, end):
    expected = pd.date_range(start, end, inclusive="left")
    observed = pd.DatetimeIndex(trips.inicio_viaje.dt.normalize().unique())
    missing = expected.difference(observed)
    if len(missing):
        raise ValueError(f"Missing calendar dates: {missing.tolist()}")
    return len(expected)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def prepare(source, october, out):
    manifest = json.loads(Path("manifests/mibici_processed.json").read_text())
    provenance = []
    frames = []
    for item in manifest["files"]:
        path = source / item["filename"]
        actual = digest(path)
        if actual != item["sha256"]:
            raise ValueError(f"Source hash mismatch: {path.name}")
        frame = pd.read_parquet(path)
        if len(frame) != item["records"]:
            raise ValueError("Source row count mismatch")
        frames.append(frame)
        provenance.append({"file": path.name, "sha256": actual, "rows": len(frame)})
    catalog_path = source / "estaciones.parquet"
    catalog_mismatch = digest(catalog_path) != manifest["stations"]["sha256"]
    stations = pd.read_parquet(catalog_path)
    if catalog_mismatch:
        raw_catalog = source.parent / "raw" / manifest["stations"]["source_filename"]
        if digest(raw_catalog) != manifest["stations"]["source_sha256"]:
            raise ValueError("Catalog and source hash mismatch")
        rebuilt = pd.read_csv(
            raw_catalog, encoding="utf-8-sig", usecols=list(COLUMNAS_ESTACIONES)
        ).rename(columns=COLUMNAS_ESTACIONES)
        for col in ["latitud", "longitud", "capacidad"]:
            rebuilt[col] = pd.to_numeric(rebuilt[col], errors="coerce")
        rebuilt.loc[rebuilt.capacidad <= 0, "capacidad"] = float("nan")
        rebuilt = (
            rebuilt.drop_duplicates("estacion_id")
            .sort_values("estacion_id")
            .reset_index(drop=True)
        )
        numerical = ["estacion_id", "latitud", "longitud", "capacidad"]
        pd.testing.assert_frame_equal(
            stations[numerical].reset_index(drop=True),
            rebuilt[numerical],
            check_dtype=False,
            check_exact=True,
        )
        # Re-derive from byte-verified raw source; record the stale parquet manifest.
        stations = rebuilt
    # Keep the original September catalog: otherwise changed metadata could create artificial drift.
    new = pd.read_csv(october, encoding="utf-8-sig", usecols=list(COLUMNAS)).rename(
        columns=COLUMNAS
    )
    raw_n = len(new)
    for col in ["inicio_viaje", "fin_viaje"]:
        new[col] = pd.to_datetime(new[col], errors="raise")
    for col in ["viaje_id", "origen_id", "destino_id"]:
        new[col] = pd.to_numeric(new[col], errors="raise").astype("int64")
    new = new.drop_duplicates()
    dup = raw_n - len(new)
    invalid = int((new.fin_viaje < new.inicio_viaje).sum())
    new = new.loc[new.fin_viaje >= new.inicio_viaje].copy()
    if not new.inicio_viaje.between("2026-10-01", "2026-11-01", inclusive="left").all():
        raise ValueError("October file contains out-of-period starts")
    coverage(new, "2026-10-01", "2026-11-01")
    old = pd.concat(frames, ignore_index=True)
    coverage(old, "2025-08-01", "2026-10-01")
    combined = pd.concat([old, new], ignore_index=True)
    if combined.viaje_id.duplicated().any():
        raise ValueError("Duplicate trip IDs across months")
    # Scope the extension to this call; do not change shared training configuration.
    original_end = features.DATA_END_EXCLUSIVE
    try:
        features.DATA_END_EXCLUSIVE = date(2026, 11, 1)
        panel = features.dataset_p1(combined, stations)
    finally:
        features.DATA_END_EXCLUSIVE = original_end
    reference = panel.loc[panel.fecha < "2026-08-01"].copy()
    current = panel.loc[panel.fecha >= "2026-10-01"].copy()
    validate_windows(reference, current)
    reference.to_parquet(out / "reference.parquet", index=False)
    current.to_parquet(out / "current.parquet", index=False)
    summary = {
        "reference_window": ["2025-08-01", "2026-08-01"],
        "current_window": ["2026-10-01", "2026-11-01"],
        "intervals": "left-closed, right-open",
        "reference_n": len(reference),
        "current_n": len(current),
        "reference_actual_first_date": str(reference.fecha.min()),
        "reference_stations": reference.estacion_id.nunique(),
        "current_stations": current.estacion_id.nunique(),
        "new_station_ids": sorted(set(current.estacion_id) - set(reference.estacion_id)),
        "october_raw_trips": raw_n,
        "october_duplicates_removed": dup,
        "october_invalid_removed": invalid,
        "october_days_present": 31,
        "october_sha256": digest(october),
        "october_url": "https://mibici.net/site/assets/files/133188/datos_abiertos_2026_10.csv",
        "catalog_manifest_mismatch": catalog_mismatch,
        "catalog_source_verified_and_numeric_values_equal": True,
        "catalog_sha256": digest(catalog_path),
        "original_data_version": manifest["data_version"],
        "source_files": provenance,
    }
    for name, frame in [("reference", reference), ("current", current)]:
        summary[name + "_missing"] = (
            frame[features.VARIABLES_P1 + ["salidas"]].isna().sum().to_dict()
        )
        summary[name + "_target_mean"] = float(frame.salidas.mean())
        summary[name + "_date_count"] = frame.fecha.nunique()
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=int), encoding="utf-8")
    return reference, current, summary


def report(reference, current, out):
    from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.legacy.pipeline.column_mapping import ColumnMapping
    from evidently.legacy.report import Report

    cols = features.VARIABLES_P1 + ["salidas"]
    mapping = ColumnMapping(
        target=None,
        prediction=None,
        numerical_features=[c for c in cols if c not in CATEGORICAL],
        categorical_features=CATEGORICAL,
    )
    # Explicit effect-size methods avoid sample-size driven p-values on dependent panel rows.
    result = Report(
        metrics=[
            DataDriftPreset(
                num_stattest="wasserstein",
                cat_stattest="jensenshannon",
                num_stattest_threshold=0.1,
                cat_stattest_threshold=0.1,
            ),
            DataQualityPreset(),
        ]
    )
    result.run(reference_data=reference[cols], current_data=current[cols], column_mapping=mapping)
    result.save_html(str(out / "evidently.html"))
    result.save_json(str(out / "evidently.json"))
    return result.as_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--october", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/drift-p1"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reference, current, summary = prepare(args.source, args.october, args.out)
    print(json.dumps(summary, default=int), flush=True)
    report(reference, current, args.out)
    # Sensitivity: same calendar month and same station cohort, no tuning.
    seasonal = reference.loc[reference.fecha.dt.month == 10].copy()
    common = sorted(set(seasonal.estacion_id) & set(current.estacion_id))
    seasonal = seasonal.loc[seasonal.estacion_id.isin(common)]
    seasonal_current = current.loc[current.estacion_id.isin(common)]
    sensitivity_dir = args.out / "same-month-common-stations"
    sensitivity_dir.mkdir(exist_ok=True)
    validate_windows(seasonal, seasonal_current)
    report(seasonal, seasonal_current, sensitivity_dir)
    sensitivity = {
        "reference_n": len(seasonal),
        "current_n": len(seasonal_current),
        "common_stations": len(common),
        "reference_target_mean": float(seasonal.salidas.mean()),
        "current_target_mean": float(seasonal_current.salidas.mean()),
    }
    (sensitivity_dir / "summary.json").write_text(
        json.dumps(sensitivity, indent=2), encoding="utf-8"
    )
    print("Evidently report saved", flush=True)


if __name__ == "__main__":
    main()
