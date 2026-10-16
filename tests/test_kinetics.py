"""Plasma conversion and relative error (pyPET itself is not needed)."""

import numpy as np

from pvc_dlif_pipeline.kinetics import relative_error, two_tissue_parameter_names, whole_blood_to_plasma


def test_wu_ratio_at_start_and_late():
    """R(t) = 0.386 exp(-0.191 t) + 1.165: 1.551 at t = 0, 1.1663 at 30 min, 1.1650 at 60 min."""
    plasma = whole_blood_to_plasma(np.ones(3), np.array([0.0, 30.0, 60.0]), 0.386, 0.191, 1.165)
    assert np.allclose(plasma, [1.551, 1.16625, 1.16500], atol=1e-5)


def test_relative_error_edge_cases():
    assert relative_error(1.1, 1.0) == 0.10000000000000009
    assert np.isnan(relative_error(1.0, 0.0))
    assert np.isnan(relative_error(np.nan, 1.0))


def test_parameter_names():
    assert "k4" not in two_tissue_parameter_names(irreversible=True)
    assert "k4" in two_tissue_parameter_names(irreversible=False)
