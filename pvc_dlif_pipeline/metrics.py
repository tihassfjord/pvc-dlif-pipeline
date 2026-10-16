"""Error metrics of a predicted input function against the measured arterial curve.

Areas are trapezoidal integrals over the sampling times in minutes, because
the samples are far from evenly spaced (5 s early, 5 min late).
"""

import numpy as np


def curve_metrics(predicted, measured, time_min):
    """Return the metrics of one predicted curve as a dictionary.

    rmse, mae, bias    root-mean-square, mean absolute and mean signed error (SUV)
    nrmse              rmse divided by the standard deviation of the measured curve
    r2                 1 - sum of squared errors / total sum of squares of the measured curve
    pearson_r, ccc     Pearson correlation and Lin's concordance correlation coefficient
    auc_ratio          area under the predicted curve / area under the measured curve
    peak_ratio         highest predicted value / highest measured value (descriptive, not tested)
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64)
    time_min = np.asarray(time_min, dtype=np.float64)
    error = predicted - measured

    rmse = float(np.sqrt(np.mean(error ** 2)))
    total_sum_of_squares = float(np.sum((measured - measured.mean()) ** 2))
    covariance = float(np.mean((predicted - predicted.mean()) * (measured - measured.mean())))
    ccc_denominator = predicted.var() + measured.var() + (predicted.mean() - measured.mean()) ** 2
    return {
        "rmse": rmse,
        "nrmse": rmse / float(measured.std()),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": 1.0 - float(np.sum(error ** 2)) / total_sum_of_squares,
        "pearson_r": float(np.corrcoef(predicted, measured)[0, 1]),
        "ccc": float(2 * covariance / ccc_denominator),
        "auc_ratio": float(np.trapezoid(predicted, time_min) / np.trapezoid(measured, time_min)),
        "peak_ratio": float(predicted.max() / measured.max()),
    }
