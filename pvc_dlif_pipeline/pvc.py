"""Partial volume correction with PETPVC, one 3D frame at a time.

PETPVC's deconvolution methods work on 3D volumes, so each frame is written to
a temporary NIfTI file, corrected by the petpvc program, and read back.
PETPVC reads the voxel size from the NIfTI header and is given the PSF FWHM in
millimetres, so it does the conversion from millimetres to voxels itself.

Method codes: Richardson-Lucy is "RL"; reblurred Van Cittert is "VC" in
PETPVC (the program only implements the reblurred variant).
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import nibabel
import numpy as np

PETPVC_METHOD_CODE = {"RL": "RL", "RVC": "VC"}


def correction_folder_name(correction):
    """{method: RL, iterations: 15, psf_scale: 0.9} -> "rl_i15_psf0p90"."""
    name = f"{correction['method'].lower()}_i{correction['iterations']}"
    if correction["psf_scale"] != 1.0:
        name += "_psf" + f"{correction['psf_scale']:.2f}".replace(".", "p")
    return name


def find_petpvc():
    """Return the path of the petpvc program, or stop if it is not installed."""
    petpvc_path = shutil.which("petpvc")
    if petpvc_path is None:
        raise SystemExit("petpvc not found. Install it with: conda install -c conda-forge petpvc")
    return petpvc_path


def correct_volume(petpvc_path, volume_zyx, voxel_mm_xyz, method, iterations, fwhm_mm_xyz, rvc_relaxation):
    """Run PETPVC on one 3D volume (z, y, x) and return the corrected volume.

    The stopping criterion is set to 0 (-s 0), so exactly `iterations`
    iterations are run.
    """
    affine = np.diag([voxel_mm_xyz[0], voxel_mm_xyz[1], voxel_mm_xyz[2], 1.0])
    with tempfile.TemporaryDirectory(prefix="petpvc_") as temporary_folder:
        input_path = Path(temporary_folder) / "frame.nii.gz"
        output_path = Path(temporary_folder) / "frame_pvc.nii.gz"
        volume_xyz = np.ascontiguousarray(np.asarray(volume_zyx, dtype=np.float32).transpose(2, 1, 0))
        nibabel.save(nibabel.Nifti1Image(volume_xyz, affine), str(input_path))

        command = [petpvc_path, "-i", str(input_path), "-o", str(output_path),
                   "-p", PETPVC_METHOD_CODE[method],
                   "-x", f"{fwhm_mm_xyz[0]:.6g}", "-y", f"{fwhm_mm_xyz[1]:.6g}", "-z", f"{fwhm_mm_xyz[2]:.6g}",
                   "-k", str(iterations)]
        if method == "RVC":
            command += ["-a", f"{rvc_relaxation:.6g}"]
        command += ["-s", "0"]

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 or not output_path.exists():
            raise RuntimeError(f"PETPVC failed (exit {result.returncode}):\n{' '.join(command)}\n"
                               f"{result.stdout}\n{result.stderr}")
        corrected_xyz = np.asarray(nibabel.load(str(output_path)).get_fdata(dtype=np.float32))
    return np.ascontiguousarray(corrected_xyz.transpose(2, 1, 0), dtype=np.float32)


def background_mask(volume):
    """Low-activity voxels inside the body, used for the noise estimate.

    The body is every voxel above 10 % of the 85th percentile of the positive
    voxels; the background is the body voxels between its 20th and 60th
    percentile. The mask is taken from the uncorrected frame and used for both
    the uncorrected and the corrected frame.
    """
    positive = volume[volume > 0]
    if positive.size == 0:
        return np.zeros_like(volume, dtype=bool)
    body_threshold = np.percentile(positive, 85)
    body = volume > 0.1 * body_threshold
    if body.sum() < 100:
        body = volume > 0
    low, high = np.percentile(volume[body], [20, 60])
    return body & (volume >= low) & (volume <= high)


def blood_mask(peak_frame_volume):
    """The hottest 0.1 % of voxels (at least 8) in the frame with the highest maximum."""
    values = peak_frame_volume.ravel()
    n_voxels = max(int(values.size * 0.001), 8)
    threshold = np.partition(values, -n_voxels)[-n_voxels]
    return peak_frame_volume >= threshold


def percent_std(volume, mask):
    """Coefficient of variation in percent inside the mask (NaN if undefined)."""
    values = volume[mask]
    if values.size < 2 or abs(float(values.mean())) < 1e-12:
        return float("nan")
    return float(100.0 * values.std(ddof=1) / values.mean())


def frame_diagnostics(uncorrected_series, corrected_series, frame_start_s, frame_duration_s):
    """One row per frame: what the correction did to noise, the blood peak and the sign.

    Peak recovery is blood_peak_after / blood_peak_before, and noise
    amplification is noise_pct_std_after / noise_pct_std_before.
    """
    peak_frame = int(np.argmax(uncorrected_series.reshape(uncorrected_series.shape[0], -1).max(axis=1)))
    blood = blood_mask(uncorrected_series[peak_frame])
    rows = []
    for frame in range(uncorrected_series.shape[0]):
        before = uncorrected_series[frame]
        after = corrected_series[frame]
        background = background_mask(before)
        rows.append({
            "frame": frame,
            "time_min": (frame_start_s[frame] + frame_duration_s[frame] / 2.0) / 60.0,
            "duration_s": frame_duration_s[frame],
            "counts_proxy": float(before.sum() * frame_duration_s[frame]),
            "mean_before": float(before.mean()),
            "mean_after": float(after.mean()),
            "max_before": float(before.max()),
            "max_after": float(after.max()),
            "noise_pct_std_before": percent_std(before, background),
            "noise_pct_std_after": percent_std(after, background),
            "blood_peak_before": float(before[blood].max()),
            "blood_peak_after": float(after[blood].max()),
            "negative_fraction_after": float((after < 0).mean()),
        })
    return rows
