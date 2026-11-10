import numpy as np
import pandas as pd
import pytest

from mibici.accuracy_monitoring import block_interval, metrics


def sample():
    return pd.DataFrame(
        {
            "fecha": pd.date_range("2026-09-01", periods=14),
            "estacion_id": [1] * 14,
            "salidas": [10.0] * 14,
            "prediction": [12.0] * 14,
        }
    )


def test_metrics_exact():
    m = metrics(sample())
    assert m["mae"] == 2
    assert m["rmse"] == 2
    assert m["wape"] == 0.2
    assert m["bias_prediction_minus_actual"] == 2


def test_nonfinite_predictions_rejected():
    f = sample()
    f.loc[0, "prediction"] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        metrics(f)


def test_zero_demand_rejected():
    f = sample()
    f["salidas"] = 0
    with pytest.raises(ValueError, match="positive"):
        metrics(f)


def test_empty_metrics_rejected():
    with pytest.raises(ValueError, match="Empty"):
        metrics(sample().iloc[:0])


def test_block_interval_known_shift():
    a = sample()
    b = sample()
    b["prediction"] = 14
    ci = block_interval(a, b, repetitions=20)
    assert ci["mae"] == {"low": 2.0, "high": 2.0}
    assert ci["rmse"] == {"low": 2.0, "high": 2.0}
    assert ci["wape"]["low"] == pytest.approx(0.2)


def test_bootstrap_repeatable():
    a = sample()
    b = sample()
    b["prediction"] = np.arange(14) + 8
    assert block_interval(a, b, repetitions=30) == block_interval(a, b, repetitions=30)
