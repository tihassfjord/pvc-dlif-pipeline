"""The crop from the reconstructed volume to the network input.

The group's preprocessing is a pure crop at native resolution: a 96 x 48 x 48
voxel window cut out of the 128 x 92 x 92 reconstruction, with no resampling,
and one intensity factor per scan (about 0.99). The window position differs
between scans. It is recovered here by matching the uncorrected series against
the group's distributed network input for the same scan, and is then reused
unchanged for every corrected series, so the conditions differ only by the
correction and never by the crop.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

# Frames averaged for the matching: mid and late frames, where the animal is
# clearly visible. The first frames are nearly empty.
MATCHING_FRAMES = [8, 14, 20, 30, 41]


def crop_with_zero_padding(volume_zyx, start_zyx, shape_zyx):
    """Cut a window out of a 3D volume, filling with zeros where it leaves the volume.

    Seven of the group's windows reach 1 to 8 slices past the end of the
    128-slice reconstruction, and their distributed inputs hold exact zeros in
    those slices. Plain slicing would return a shorter array instead.
    """
    window = np.zeros(shape_zyx, dtype=volume_zyx.dtype)
    source_slices = []
    window_slices = []
    for axis in range(3):
        start = start_zyx[axis]
        stop = start + shape_zyx[axis]
        source_start = max(start, 0)
        source_stop = min(stop, volume_zyx.shape[axis])
        if source_start >= source_stop:
            return window
        source_slices.append(slice(source_start, source_stop))
        window_slices.append(slice(source_start - start, source_stop - start))
    window[tuple(window_slices)] = volume_zyx[tuple(source_slices)]
    return window


def correlation(first, second):
    """Pearson correlation of two arrays of equal shape (-1 if either is constant)."""
    first = first.ravel()
    second = second.ravel()
    if first.std() < 1e-12 or second.std() < 1e-12:
        return -1.0
    return float(np.corrcoef(first, second)[0, 1])


def find_crop_window(native_series, reference_series):
    """Find the window position where the native series matches the reference input.

    Both series are averaged over MATCHING_FRAMES. The search runs in two passes:
      1. a coarse pass over every position at two-voxel steps, scored on
         smoothed volumes sampled at every second voxel;
      2. a fine pass at one-voxel steps within three voxels of the coarse
         winner, scored on the unsmoothed volumes.
    Smoothing in the coarse pass makes a near miss still score high; on the raw
    volumes a one-voxel error already destroys the correlation. The window may
    hang up to a quarter of its size outside the volume (see
    crop_with_zero_padding).

    Returns (start (z, y, x), correlation at that start).
    """
    window_shape = reference_series.shape[1:]
    native_mean = native_series[MATCHING_FRAMES].mean(axis=0).astype(np.float64)
    reference_mean = reference_series[MATCHING_FRAMES].mean(axis=0).astype(np.float64)
    smooth_native = gaussian_filter(native_mean, 1.5)
    smooth_reference = gaussian_filter(reference_mean, 1.5)

    lowest_start = [-(size // 4) for size in window_shape]
    highest_start = [native_mean.shape[axis] - window_shape[axis] + window_shape[axis] // 4
                     for axis in range(3)]

    best_score = -2.0
    best_start = None
    for z in range(lowest_start[0], highest_start[0] + 1, 2):
        for y in range(lowest_start[1], highest_start[1] + 1, 2):
            for x in range(lowest_start[2], highest_start[2] + 1, 2):
                window = crop_with_zero_padding(smooth_native, (z, y, x), window_shape)
                score = correlation(window[::2, ::2, ::2], smooth_reference[::2, ::2, ::2])
                if score > best_score:
                    best_score, best_start = score, (z, y, x)

    coarse_start = best_start
    best_score = -2.0
    for z in range(coarse_start[0] - 3, coarse_start[0] + 4):
        for y in range(coarse_start[1] - 3, coarse_start[1] + 4):
            for x in range(coarse_start[2] - 3, coarse_start[2] + 4):
                window = crop_with_zero_padding(native_mean, (z, y, x), window_shape)
                score = correlation(window, reference_mean)
                if score > best_score:
                    best_score, best_start = score, (z, y, x)
    return best_start, best_score


def intensity_factor(native_series, reference_series, start_zyx):
    """Least-squares factor s that makes s * window closest to the reference.

    Computed on the same frame average as the window search.
    """
    window_shape = reference_series.shape[1:]
    native_mean = native_series[MATCHING_FRAMES].mean(axis=0).astype(np.float64)
    reference_mean = reference_series[MATCHING_FRAMES].mean(axis=0).astype(np.float64)
    window = crop_with_zero_padding(native_mean, start_zyx, window_shape)
    return float(np.dot(window.ravel(), reference_mean.ravel()) / np.dot(window.ravel(), window.ravel()))


def crop_series(series_tzyx, start_zyx, shape_zyx, factor):
    """Apply the crop and the intensity factor to every frame of a series."""
    cropped = np.empty((series_tzyx.shape[0], *shape_zyx), dtype=np.float32)
    for frame in range(series_tzyx.shape[0]):
        cropped[frame] = crop_with_zero_padding(series_tzyx[frame], start_zyx, shape_zyx) * factor
    return cropped
