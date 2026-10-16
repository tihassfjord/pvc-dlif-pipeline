"""Reading the dynamic PET exports.

Each scan is one PMOD-exported multi-frame DICOM file named dPET_dcm_<ID>.
The file holds (number of time frames x number of slices) image frames, with
the slice index running fastest. Frame timing is stored in PMOD private tags.
"""

from pathlib import Path

import numpy as np
import pydicom

# PMOD private tags (group 0x0055).
TAG_FRAME_START_SECONDS = (0x0055, 0x1001)
TAG_FRAME_DURATION_MILLISECONDS = (0x0055, 0x1004)
TAG_RESCALE_PER_IMAGE_FRAME = (0x0055, 0x1005)


def list_dicom_exports(dicom_folder):
    """Return {scan_id: path} for every dPET_dcm_<ID> file in the folder."""
    exports = {}
    for path in sorted(Path(dicom_folder).iterdir()):
        if path.name.lower().startswith("dpet_dcm_"):
            scan_id = path.name[len("dPET_dcm_"):]
            exports[scan_id] = path
    return exports


def read_dicom_header(dicom_path):
    """Read grid, units and frame timing of one export without its pixel data.

    Returns a dictionary with
        shape_zyx          (slices, rows, columns)
        voxel_mm           (x, y, z) in millimetres
        units              e.g. "SUV"
        frame_start_s      list of frame start times in seconds
        frame_duration_s   list of frame durations in seconds

    Raises ValueError if the frame timing tags are missing, because every
    later stage depends on the timing being correct.
    """
    header = pydicom.dcmread(str(dicom_path), stop_before_pixels=True, force=True)

    n_slices = int(header.NumberOfSlices)
    n_image_frames = int(header.NumberOfFrames)
    if n_image_frames % n_slices != 0:
        raise ValueError(f"{dicom_path}: {n_image_frames} frames is not a multiple of {n_slices} slices")
    n_time_frames = n_image_frames // n_slices

    if TAG_FRAME_START_SECONDS not in header or TAG_FRAME_DURATION_MILLISECONDS not in header:
        raise ValueError(f"{dicom_path}: frame timing tags (0055,1001)/(0055,1004) are missing")
    frame_start_s = [float(value) for value in header[TAG_FRAME_START_SECONDS].value][:n_time_frames]
    frame_duration_s = [float(value) / 1000.0
                        for value in header[TAG_FRAME_DURATION_MILLISECONDS].value][:n_time_frames]

    row_spacing_mm, column_spacing_mm = [float(value) for value in header.PixelSpacing]
    slice_spacing_mm = float(getattr(header, "SpacingBetweenSlices", None) or header.SliceThickness)

    return {
        "shape_zyx": (n_slices, int(header.Rows), int(header.Columns)),
        "voxel_mm": (column_spacing_mm, row_spacing_mm, slice_spacing_mm),
        "units": str(header.Units),
        "frame_start_s": frame_start_s,
        "frame_duration_s": frame_duration_s,
    }


def read_dicom_series(dicom_path):
    """Read the activity of one export as an array (time frames, slices, rows, columns) in SUV.

    The stored integers are multiplied by the PMOD rescale vector, which holds
    one factor per image frame (one per slice and time frame). Every export in
    this dataset has that vector; the function stops if it is missing rather
    than guessing another scaling.
    """
    dataset = pydicom.dcmread(str(dicom_path), force=True)
    n_slices = int(dataset.NumberOfSlices)
    n_image_frames = int(dataset.NumberOfFrames)
    n_time_frames = n_image_frames // n_slices

    if TAG_RESCALE_PER_IMAGE_FRAME not in dataset:
        raise ValueError(f"{dicom_path}: rescale vector (0055,1005) is missing")
    rescale = np.asarray(dataset[TAG_RESCALE_PER_IMAGE_FRAME].value, dtype=float)
    if rescale.size != n_image_frames:
        raise ValueError(f"{dicom_path}: {rescale.size} rescale factors for {n_image_frames} image frames")

    # The slice index runs fastest inside each time frame.
    series = dataset.pixel_array.reshape(n_time_frames, n_slices, int(dataset.Rows), int(dataset.Columns))
    series = series.astype(np.float32)
    series *= rescale.reshape(n_time_frames, n_slices, 1, 1).astype(np.float32)
    return series
