"""Inter-frame motion correction with FALCON (falconz), and what it did.

FALCON registers every frame to a reference frame and resamples it. It is
installed separately (pip install falconz) and called as a program.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np


def find_falcon():
    """Return the path of the falconz program, or stop if it is not installed."""
    falcon_path = shutil.which("falconz")
    if falcon_path is None:
        raise SystemExit("falconz not found. Install it with: pip install falconz")
    return falcon_path


def reference_frame_index(series_tzyx):
    """The frame in the second half of the scan that correlates best with the mean of that half.

    Late frames have the best counting statistics and the most stable
    anatomy, so they make a better target than the early bolus frames.
    """
    n_frames = series_tzyx.shape[0]
    late_mean = series_tzyx[n_frames // 2:].mean(axis=0)
    scores = [float(np.corrcoef(series_tzyx[frame].ravel(), late_mean.ravel())[0, 1])
              for frame in range(n_frames // 2, n_frames)]
    return n_frames // 2 + int(np.argmax(scores))


def run_falcon(falcon_path, input_nifti, output_folder, reference_frame, start_frame, registration):
    """Run falconz on one 4D NIfTI file and return the path of the corrected file.

    falconz corrects every image in the folder it is given (-d), so the input
    is first linked into a folder of its own. The command and falconz's output
    are saved next to the result. The per-frame transforms are kept in
    <output_folder>/transforms; falconz's other working files are deleted.
    """
    input_nifti = Path(input_nifti)
    output_folder = Path(output_folder)
    input_folder = output_folder / "input"
    if input_folder.exists():
        shutil.rmtree(input_folder)
    input_folder.mkdir(parents=True)
    try:
        os.link(input_nifti, input_folder / input_nifti.name)
    except OSError:
        shutil.copy2(input_nifti, input_folder / input_nifti.name)

    corrected_path = output_folder / input_nifti.name.replace(".nii.gz", "_moco.nii.gz")
    command = [falcon_path, "-d", str(input_folder), "-r", registration,
               "-rf", str(reference_frame), "-sf", str(start_frame), "-o", str(corrected_path)]

    # falconz prints emoji; without UTF-8 the run fails on a Norwegian Windows locale.
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    result = subprocess.run(command, cwd=str(output_folder), capture_output=True, text=True,
                            encoding="utf-8", errors="replace", env=environment)
    (output_folder / "falcon_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    (output_folder / "falcon_stdout.txt").write_text(result.stdout or "", encoding="utf-8")
    (output_folder / "falcon_stderr.txt").write_text(result.stderr or "", encoding="utf-8")
    if result.returncode != 0 or not corrected_path.exists():
        raise RuntimeError(f"falconz failed (exit {result.returncode}); see {output_folder}/falcon_stderr.txt")

    for run_folder in output_folder.glob("FALCONZ-*"):
        if (run_folder / "transforms").is_dir():
            if (output_folder / "transforms").exists():
                shutil.rmtree(output_folder / "transforms")
            shutil.move(str(run_folder / "transforms"), str(output_folder / "transforms"))
        shutil.rmtree(run_folder, ignore_errors=True)
    shutil.rmtree(input_folder, ignore_errors=True)
    return corrected_path


def read_falcon_transforms(transforms_folder):
    """One row per registered frame: translation (mm) and rotation angle (degrees).

    falconz writes one 4 x 4 matrix per frame, vol_XXXX.nii.gz_rigid.mat. The
    rotation angle is computed from the trace of the 3 x 3 rotation part.
    """
    rows = []
    for path in sorted(Path(transforms_folder).glob("*_rigid.mat")):
        frame = int(re.search(r"vol_(\d+)", path.name).group(1))
        matrix = np.loadtxt(path)
        translation = matrix[:3, 3]
        cosine = (float(np.trace(matrix[:3, :3])) - 1.0) / 2.0
        rows.append({
            "frame": frame,
            "tx_mm": float(translation[0]), "ty_mm": float(translation[1]), "tz_mm": float(translation[2]),
            "translation_mm": float(np.linalg.norm(translation)),
            "rotation_deg": float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))),
        })
    return sorted(rows, key=lambda row: row["frame"])


def summarise_transforms(rows, rotation_suspect_deg):
    """Motion summary of one scan, leaving out frames whose rotation marks a failed registration."""
    translation = np.array([row["translation_mm"] for row in rows])
    rotation = np.array([row["rotation_deg"] for row in rows])
    frames = np.array([row["frame"] for row in rows])
    suspect = rotation > rotation_suspect_deg
    trusted = translation[~suspect]
    return {
        "n_frames_registered": len(rows),
        "n_frames_suspect": int(suspect.sum()),
        "suspect_frames": " ".join(str(frame) for frame in frames[suspect]),
        "applied_median_mm": float(np.median(trusted)) if trusted.size else float("nan"),
        "applied_p95_mm": float(np.percentile(trusted, 95)) if trusted.size else float("nan"),
        "applied_max_mm": float(trusted.max()) if trusted.size else float("nan"),
        "applied_rotation_max_deg": float(rotation[~suspect].max()) if trusted.size else float("nan"),
    }


def gradient_energy(series_tzyx):
    """Mean squared image gradient, averaged over frames. Smoothing lowers it."""
    total = 0.0
    for frame in range(series_tzyx.shape[0]):
        gradients = np.gradient(series_tzyx[frame].astype(np.float64))
        total += float(np.mean(sum(gradient ** 2 for gradient in gradients)))
    return total / series_tzyx.shape[0]
