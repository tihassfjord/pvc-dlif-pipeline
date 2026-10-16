"""Crop window: padding at the volume edge, and recovery of a known window."""

import numpy as np
from scipy.ndimage import gaussian_filter

from pvc_dlif_pipeline.crop import crop_series, crop_with_zero_padding, find_crop_window


def test_padding_past_the_last_slice():
    volume = np.ones((10, 6, 6))
    window = crop_with_zero_padding(volume, (7, 0, 0), (5, 6, 6))
    assert window.shape == (5, 6, 6)
    assert np.all(window[:3] == 1.0) and np.all(window[3:] == 0.0)


def test_window_inside_the_volume_is_a_plain_slice():
    volume = np.arange(1000.0).reshape(10, 10, 10)
    assert np.array_equal(crop_with_zero_padding(volume, (2, 3, 4), (4, 4, 4)), volume[2:6, 3:7, 4:8])


def test_find_crop_window_recovers_the_true_start():
    """A smooth random series, and a reference cut out of it at a known start."""
    generator = np.random.default_rng(7)
    native = np.stack([gaussian_filter(generator.random((24, 20, 20)), 1.0) for _ in range(42)])
    true_start = (5, 3, 6)
    reference = crop_series(native, true_start, (12, 10, 10), 1.0)
    start, score = find_crop_window(native, reference)
    assert tuple(start) == true_start
    assert np.isclose(score, 1.0)
