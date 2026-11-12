"""Inference for locally evaluated residual-demand estimator."""

import numpy as np
from threadpoolctl import threadpool_limits

from mibici.features import VARIABLES_P1


def predict_residual_demand(estimator, frame):
    baseline = frame["salidas_media7"].to_numpy(dtype=float)
    if not np.isfinite(baseline).all():
        raise ValueError("Finite historical seven-day demand is required")
    with threadpool_limits(limits=4):
        residual = estimator.predict(frame[VARIABLES_P1].astype("float64"))
    if len(residual) != len(frame) or not np.isfinite(residual).all():
        raise ValueError("Invalid residual prediction")
    return np.maximum(0, baseline + residual)
