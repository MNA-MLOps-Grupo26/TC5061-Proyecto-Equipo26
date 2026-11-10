import pandas as pd
import pytest

from mibici.monitoring import coverage, validate_windows


def frame(dates):
    return pd.DataFrame({"fecha": pd.to_datetime(dates), "estacion_id": [1] * len(dates)})


def test_nonoverlapping_periods():
    validate_windows(frame(["2026-07-31"]), frame(["2026-10-01"]))


def test_overlap_rejected():
    with pytest.raises(ValueError, match="overlap"):
        validate_windows(frame(["2026-10-01"]), frame(["2026-10-01"]))


def test_empty_rejected():
    with pytest.raises(ValueError, match="Empty"):
        validate_windows(frame([]), frame(["2026-10-01"]))


def test_duplicate_station_day_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        validate_windows(frame(["2026-07-31", "2026-07-31"]), frame(["2026-10-01"]))


def test_missing_calendar_day_rejected():
    with pytest.raises(ValueError, match="Missing calendar"):
        coverage(
            pd.DataFrame({"inicio_viaje": pd.to_datetime(["2026-10-01"])}),
            "2026-10-01",
            "2026-10-03",
        )


def test_full_calendar_accepted():
    assert (
        coverage(
            pd.DataFrame({"inicio_viaje": pd.date_range("2026-10-01", "2026-10-31")}),
            "2026-10-01",
            "2026-11-01",
        )
        == 31
    )
