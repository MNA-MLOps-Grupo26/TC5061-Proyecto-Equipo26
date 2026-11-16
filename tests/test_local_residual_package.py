"""Regression checks for the local MLflow demand wrapper (no registry writes)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from mibici.features import VARIABLES_P1
from scripts.register_local_residual import ResidualDemandModel


class Estimator:
    def predict(self, frame):
        assert list(frame.columns) == VARIABLES_P1
        return np.array([2.0, -20.0])


def test_pyfunc_outputs_demand_not_residual():
    model = ResidualDemandModel()
    model.estimator = Estimator()
    frame = pd.DataFrame(0.0, index=range(2), columns=VARIABLES_P1)
    frame["salidas_media7"] = [10.0, 5.0]
    np.testing.assert_array_equal(model.predict(None, frame), [12.0, 0.0])


def test_pyfunc_loads_estimator_artifact(monkeypatch):
    import skops.io

    sentinel = object()

    def load(path, trusted):
        assert path == "frozen.skops"
        assert "numpy.dtype" in trusted
        return sentinel

    monkeypatch.setattr(skops.io, "load", load)
    model = ResidualDemandModel()
    model.load_context(SimpleNamespace(artifacts={"estimator": "frozen.skops"}))
    assert model.estimator is sentinel
