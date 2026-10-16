"""Curve metrics on curves with known answers."""

import numpy as np

from pvc_dlif_pipeline.metrics import curve_metrics

TIME_MIN = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0])
MEASURED = np.array([0.0, 8.0, 5.0, 3.0, 2.0, 1.5, 1.0, 0.6])


def test_perfect_prediction():
    metrics = curve_metrics(MEASURED, MEASURED, TIME_MIN)
    assert metrics["rmse"] == 0.0 and metrics["mae"] == 0.0 and metrics["bias"] == 0.0
    assert np.isclose(metrics["r2"], 1.0)
    assert np.isclose(metrics["pearson_r"], 1.0)
    assert np.isclose(metrics["ccc"], 1.0)
    assert np.isclose(metrics["auc_ratio"], 1.0)


def test_constant_offset():
    """An offset of +1 everywhere: RMSE = MAE = bias = 1, still perfectly correlated, but CCC < 1."""
    metrics = curve_metrics(MEASURED + 1.0, MEASURED, TIME_MIN)
    assert np.isclose(metrics["rmse"], 1.0)
    assert np.isclose(metrics["mae"], 1.0)
    assert np.isclose(metrics["bias"], 1.0)
    assert np.isclose(metrics["pearson_r"], 1.0)
    assert metrics["ccc"] < 1.0
    assert np.isclose(metrics["nrmse"], 1.0 / MEASURED.std())


def test_area_ratio_of_a_scaled_curve():
    """Twice the curve has twice the area, whatever the spacing of the samples."""
    assert np.isclose(curve_metrics(2.0 * MEASURED, MEASURED, TIME_MIN)["auc_ratio"], 2.0)


def test_peak_ratio():
    assert np.isclose(curve_metrics(1.5 * MEASURED, MEASURED, TIME_MIN)["peak_ratio"], 1.5)
