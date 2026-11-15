"""Synthetic tests for mibici.ab_testing; no real data or models are used."""

import json

import numpy as np
import pandas as pd
import pytest

from mibici import ab_testing
from mibici.ab_testing import compare, main, markdown, validate
from mibici.challenger_monitoring import paired_intervals


def frame(days=14, stations=3, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-10-01", periods=days)
    f = pd.DataFrame(
        [(s, d) for d in dates for s in range(1, stations + 1)], columns=["estacion_id", "fecha"]
    )
    f["salidas"] = rng.integers(5, 60, len(f)).astype(float)
    f["old"] = f.salidas + rng.normal(0, 5, len(f))
    f["new"] = f.salidas + rng.normal(0, 3, len(f))
    return f


def test_known_metrics_bias_and_sign():
    f = frame(days=7, stations=2)
    f["salidas"] = 10.0
    f["old"] = 12.0  # error +2
    f["new"] = 9.0  # error -1
    r = compare(f)
    o, c = r["metrics"]["original_v1"], r["metrics"]["candidate_v3"]
    assert o["mae"] == 2.0 and o["rmse"] == 2.0 and o["bias_prediction_minus_actual"] == 2.0
    assert c["mae"] == 1.0 and c["rmse"] == 1.0 and c["bias_prediction_minus_actual"] == -1.0
    assert o["wape"] == {"proportion": 0.2, "percent": 20.0}
    d = r["candidate_minus_original"]
    assert d["mae"] == -1.0 and d["rmse"] == -1.0  # v3 - v1: negative = v3 better
    assert d["wape"]["proportion"] == pytest.approx(-0.1)
    assert d["wape"]["percentage_points"] == pytest.approx(-10.0)
    assert d["bias_prediction_minus_actual"] == -3.0
    ci = r["paired_bootstrap_95pct_intervals"]
    assert ci["mae"] == {"low": -1.0, "high": -1.0}
    assert ci["wape"]["proportion"]["low"] == pytest.approx(-0.1)
    assert ci["wape"]["percentage_points"]["high"] == pytest.approx(-10.0)


def test_identical_models_zero_differences_and_intervals():
    f = frame()
    f["new"] = f.old
    r = compare(f)
    d = r["candidate_minus_original"]
    assert d["mae"] == d["rmse"] == d["bias_prediction_minus_actual"] == 0.0
    assert d["wape"] == {"proportion": 0.0, "percentage_points": 0.0}
    ci = r["paired_bootstrap_95pct_intervals"]
    for v in [ci["mae"], ci["rmse"], ci["wape"]["proportion"], ci["wape"]["percentage_points"]]:
        assert v == {"low": 0.0, "high": 0.0}


def test_pairing_uses_same_blocks_for_both_models():
    # Old errors vary strongly by date; new = old error + 1 on every row.
    # Paired resampling gives a MAE difference of exactly 1 in every replicate;
    # independent blocks per model would produce a wide interval.
    f = frame(days=21, stations=2)
    f["salidas"] = 100.0
    day = (f.fecha - f.fecha.min()).dt.days.to_numpy()
    f["old"] = 100.0 + (day % 7) * 10
    f["new"] = f.old + 1.0
    ci = compare(f)["paired_bootstrap_95pct_intervals"]
    assert ci["mae"]["low"] == pytest.approx(1.0) and ci["mae"]["high"] == pytest.approx(1.0)


def reference_intervals(f, repetitions, seed=42):
    """Independent restatement of the published procedure for regression checks."""
    daily = []
    for _, g in f.sort_values("fecha").groupby("fecha"):
        eo, en = g.old - g.salidas, g.new - g.salidas
        daily.append(
            [eo.abs().sum(), (eo**2).sum(), en.abs().sum(), (en**2).sum(), g.salidas.sum(), len(g)]
        )
    a = np.array(daily)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(repetitions):
        starts = rng.integers(0, len(a) - 6, size=int(np.ceil(len(a) / 7)))
        ix = np.concatenate([np.arange(s, s + 7) for s in starts])[: len(a)]
        oa, os_, na, ns, y, n = a[ix].sum(axis=0)
        out.append([(na - oa) / n, np.sqrt(ns / n) - np.sqrt(os_ / n), (na - oa) / y])
    q = np.quantile(out, [0.025, 0.975], axis=0)
    return {k: {"low": q[0, i], "high": q[1, i]} for i, k in enumerate(["mae", "rmse", "wape"])}


@pytest.mark.parametrize("days", [7, 10, 31])
def test_matches_published_procedure_blocks_concatenation_truncation(days):
    f = frame(days=days, stations=4, seed=days)
    got = paired_intervals(f, repetitions=300)
    want = reference_intervals(f, repetitions=300)
    for k in got:
        assert got[k]["low"] == pytest.approx(want[k]["low"], rel=1e-12, abs=1e-12)
        assert got[k]["high"] == pytest.approx(want[k]["high"], rel=1e-12, abs=1e-12)
    # The module reports exactly what the shared function returns (2000 repetitions).
    r = compare(f)["paired_bootstrap_95pct_intervals"]
    assert r["mae"] == paired_intervals(validate(f), repetitions=2000)["mae"]


def test_repeatable_with_fixed_seed_and_row_order_invariant():
    f = frame(days=31, stations=5, seed=3)
    a = compare(f)
    b = compare(f.sample(frac=1, random_state=7))
    assert a == b
    assert a["bootstrap"]["seed"] == 42 and a["bootstrap"]["repetitions"] == 2000
    assert a["bootstrap"]["block_length_days"] == 7


def test_dates_are_sorted_and_counted():
    f = frame(days=9, stations=2).iloc[::-1]
    f["fecha"] = f.fecha.dt.strftime("%Y-%m-%d")  # strings are parsed
    v = validate(f)
    assert v.fecha.is_monotonic_increasing
    r = compare(f)["data"]
    assert r == {
        "n": 18,
        "dates": 9,
        "first_date": "2026-10-01",
        "last_date": "2026-10-09",
        "stations": 2,
        "actual_total": float(f.salidas.sum()),
    }


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda f: f.drop(columns="new"), "Missing columns"),
        (lambda f: f.iloc[:0], "Empty"),
        (lambda f: pd.concat([f, f.iloc[:1]]), "Duplicate station-date"),
        (lambda f: f.assign(old=f.old.where(f.index != 0, np.nan)), "Non-finite old"),
        (lambda f: f.assign(new=f.new.where(f.index != 0, np.inf)), "Non-finite new"),
        (lambda f: f.assign(salidas=f.salidas.where(f.index != 0, np.nan)), "Non-finite salidas"),
        (lambda f: f.assign(salidas=0.0), "positive total"),
        (lambda f: f.assign(salidas=f.salidas.where(f.index != 0, -1.0)), "Negative"),
        (lambda f: f.assign(fecha="not a date"), "Invalid dates"),
        (lambda f: f.assign(fecha=f.fecha + pd.Timedelta(hours=3)), "without time"),
        (lambda f: f.loc[f.fecha != "2026-10-05"], "consecutive"),
        (lambda f: f.loc[f.fecha < "2026-10-07"], "seven dates"),
        (lambda f: f.assign(estacion_id=f.estacion_id.where(f.index != 0, None)), "station id"),
    ],
)
def test_rejects_invalid_inputs(mutate, message):
    with pytest.raises(ValueError, match=message):
        compare(mutate(frame()))


def test_markdown_units():
    f = frame(days=7, stations=2)
    f["salidas"], f["old"], f["new"] = 10.0, 12.0, 9.0
    table = markdown(compare(f))
    assert "| WAPE | 20.0000% | 10.0000% | -10.0000 pp | [-10.0000, -10.0000] pp |" in table
    assert "No calculado" in table


def test_cli_compare_writes_json_and_table(tmp_path):
    f = frame(days=14)
    f.to_parquet(tmp_path / "predictions.parquet", index=False)
    main(
        [
            "compare",
            "--predictions",
            str(tmp_path / "predictions.parquet"),
            "--out",
            str(tmp_path / "o"),
        ]
    )
    result = json.loads((tmp_path / "o" / "ab_results.json").read_text())
    assert result["data"]["n"] == len(f)
    assert result["provenance"]["predictions_sha256"] == ab_testing.sha256(
        tmp_path / "predictions.parquet"
    )
    assert (tmp_path / "o" / "ab_table.md").read_text().startswith("| Métrica")


def test_cli_rejects_mismatched_provenance(tmp_path):
    frame(days=14).to_parquet(tmp_path / "p.parquet", index=False)
    (tmp_path / "prov.json").write_text(json.dumps({"predictions_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="Provenance"):
        main(
            [
                "compare",
                "--predictions",
                str(tmp_path / "p.parquet"),
                "--provenance",
                str(tmp_path / "prov.json"),
                "--out",
                str(tmp_path / "o"),
            ]
        )


def test_predict_rejects_misaligned_inputs(tmp_path):
    f = frame(days=7)
    f[["estacion_id", "fecha", "salidas"]].to_parquet(tmp_path / "current.parquet")
    f.iloc[::-1].assign(prediction=f.old).to_parquet(tmp_path / "orig.parquet")
    with pytest.raises(ValueError, match="aligned"):
        ab_testing.predict(tmp_path / "current.parquet", tmp_path / "orig.parquet", tmp_path)


def test_load_v3_rejects_hash_mismatch(tmp_path):
    pytest.importorskip("skops")
    (tmp_path / "residual_estimator.skops").write_bytes(b"not the model")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "artifact": "residual_estimator.skops",
                "archive": "residual_estimator.zip",
                "sha256": "0" * 64,
                "registered_model_name": "x",
                "source_registry_version": 3,
                "source_run_id": "x",
            }
        )
    )
    with pytest.raises(ValueError, match="SHA-256"):
        ab_testing.load_v3(tmp_path)


def test_against_reference_reports_differences_without_editing():
    f = frame(days=14)
    r = compare(f)
    m, ci = r["metrics"], r["paired_bootstrap_95pct_intervals"]

    def block(label):
        v = m[label]
        return {
            "n": len(f),
            "mae": v["mae"],
            "rmse": v["rmse"],
            "wape": v["wape"]["proportion"],
            "bias_prediction_minus_actual": v["bias_prediction_minus_actual"],
        }

    reference = {
        "original": block("original_v1"),
        "candidate": block("candidate_v3"),
        "intervals": {
            "mae": ci["mae"],
            "rmse": ci["rmse"],
            "wape": ci["wape"]["proportion"],
        },
    }
    same = ab_testing.against_reference(r, reference)
    assert same["all_within_tolerance"]
    reference["original"]["mae"] += 0.01
    moved = ab_testing.against_reference(r, reference)
    assert not moved["all_within_tolerance"]
    assert moved["values"]["original.mae"]["difference"] == pytest.approx(-0.01)
