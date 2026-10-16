"""Writing and reading a dynamic series as 4D NIfTI.

PETPVC (stage 02) works on NIfTI files with the spatial axes first, so a series
held as (time, z, y, x) is stored as (x, y, z, time). Frame timing is kept in a
small JSON file next to the NIfTI file, because NIfTI has no field for it.
"""

import json
from pathlib import Path

import nibabel
import numpy as np


def timing_path(nifti_path):
    """<folder>/<ID>.nii.gz -> <folder>/<ID>.timing.json"""
    nifti_path = Path(nifti_path)
    return nifti_path.with_name(nifti_path.name.replace(".nii.gz", ".timing.json"))


def write_series(series_tzyx, voxel_mm_xyz, frame_start_s, frame_duration_s, nifti_path):
    """Write a series (time, z, y, x) as NIfTI (x, y, z, time) plus a timing file."""
    nifti_path = Path(nifti_path)
    nifti_path.parent.mkdir(parents=True, exist_ok=True)

    affine = np.diag([voxel_mm_xyz[0], voxel_mm_xyz[1], voxel_mm_xyz[2], 1.0])
    series_xyzt = np.transpose(series_tzyx, (3, 2, 1, 0))
    image = nibabel.Nifti1Image(np.ascontiguousarray(series_xyzt, dtype=np.float32), affine)
    image.header.set_xyzt_units("mm", "sec")
    nibabel.save(image, str(nifti_path))

    with open(timing_path(nifti_path), "w", encoding="utf-8") as timing_file:
        json.dump({"frame_start_s": list(frame_start_s), "frame_duration_s": list(frame_duration_s)},
                  timing_file, indent=2)


def read_series(nifti_path):
    """Return (series (time, z, y, x) as float32, voxel size (x, y, z) in mm)."""
    image = nibabel.load(str(nifti_path))
    series_xyzt = np.asarray(image.dataobj, dtype=np.float32)
    if series_xyzt.ndim == 3:
        series_xyzt = series_xyzt[..., np.newaxis]
    voxel_mm_xyz = tuple(float(size) for size in image.header.get_zooms()[:3])
    return np.transpose(series_xyzt, (3, 2, 1, 0)), voxel_mm_xyz
