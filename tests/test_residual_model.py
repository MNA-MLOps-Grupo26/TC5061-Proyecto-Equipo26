import numpy as np
import pandas as pd
import pytest

from mibici.features import VARIABLES_P1
from mibici.residual_model import predict_residual_demand


class Fake:
    def predict(self, x):
        return np.array([2, -20])


def frame():
    f = pd.DataFrame(0.0, index=range(2), columns=VARIABLES_P1)
    f["salidas_media7"] = [10, 5]
    return f


def test_add_prior_average_and_clip():
    np.testing.assert_array_equal(predict_residual_demand(Fake(), frame()), [12, 0])


def test_does_not_need_target_or_mutate_inputs():
    f = frame()
    original = f.copy()
    predict_residual_demand(Fake(), f)
    pd.testing.assert_frame_equal(f, original)


def test_missing_history_rejected():
    f = frame()
    f.loc[0, "salidas_media7"] = np.nan
    with pytest.raises(ValueError, match="historical"):
        predict_residual_demand(Fake(), f)
