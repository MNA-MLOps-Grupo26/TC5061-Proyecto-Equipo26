import pandas as pd
import pytest

from mibici.challenger_monitoring import paired_intervals


def test_paired_known_error_increase():
    f = pd.DataFrame(
        {
            "fecha": pd.date_range("2026-10-01", periods=14),
            "salidas": [10.0] * 14,
            "old": [11.0] * 14,
            "new": [13.0] * 14,
        }
    )
    r = paired_intervals(f, repetitions=20)
    assert r["mae"] == {"low": 2.0, "high": 2.0}
    assert r["rmse"] == {"low": 2.0, "high": 2.0}
    assert r["wape"]["low"] == pytest.approx(0.2)


def test_paired_identical_models_zero():
    f = pd.DataFrame(
        {
            "fecha": pd.date_range("2026-10-01", periods=14),
            "salidas": [10.0] * 14,
            "old": range(14),
            "new": range(14),
        }
    )
    assert all(
        v == {"low": 0.0, "high": 0.0} for v in paired_intervals(f, repetitions=20).values()
    )


def test_paired_short_series_rejected():
    f = pd.DataFrame(
        {
            "fecha": pd.date_range("2026-10-01", periods=3),
            "salidas": [10.0] * 3,
            "old": [11.0] * 3,
            "new": [13.0] * 3,
        }
    )
    with pytest.raises(ValueError, match="seven"):
        paired_intervals(f)
