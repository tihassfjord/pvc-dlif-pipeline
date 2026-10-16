"""Stage 01: convert the DICOM exports to NIfTI and find each scan's crop window.

For every used scan (stage 00):
  1. read the DICOM export and write it as <work>/native/<ID>.nii.gz with a
     timing file beside it; PVC (stage 02) runs on these files, at the
     reconstructed resolution the measured PSF refers to;
  2. find where the group's 96 x 48 x 48 network input was cut out of the
     uncorrected series, and the intensity factor (about 0.99) applied to it.

The window and factor are found once, here, from the uncorrected series, and
stage 04 applies the same window and factor to every corrected series.

Writes
    <work>/native/<ID>.nii.gz and <ID>.timing.json
    <work>/crop_windows.csv     scan_id, start_z, start_y, start_x, intensity_factor, correlation

Usage
    python scripts/01_convert_and_find_crop.py --config configs/study.yaml
    python scripts/01_convert_and_find_crop.py --config configs/study.yaml --overwrite
"""

import argparse
import json

import pandas as pd

from pvc_dlif_pipeline.config import load_study_config
from pvc_dlif_pipeline.crop import find_crop_window, intensity_factor
from pvc_dlif_pipeline.dicom_io import read_dicom_header, read_dicom_series
from pvc_dlif_pipeline.group_data import load_network_input
from pvc_dlif_pipeline.nifti_io import read_series, write_series


def main():
    parser = argparse.ArgumentParser(description="Stage 01: DICOM to NIfTI, and the crop windows.")
    parser.add_argument("--config", required=True, help="path to configs/study.yaml")
    parser.add_argument("--overwrite", action="store_true", help="rewrite NIfTI files that already exist")
    args = parser.parse_args()

    config = load_study_config(args.config)
    work_folder = config["paths"]["work_folder"]
    group_data_folder = config["paths"]["group_data_folder"]
    network_input_shape = tuple(config["scans"]["network_input_shape"])
    manifest = pd.read_csv(work_folder / "manifest.csv")
    used_scans = manifest[manifest["used"]]

    windows = []
    for number, scan in enumerate(used_scans.itertuples(), start=1):
        nifti_path = work_folder / "native" / f"{scan.scan_id}.nii.gz"
        if args.overwrite or not nifti_path.exists():
            header = read_dicom_header(scan.dicom_path)
            series = read_dicom_series(scan.dicom_path)
            write_series(series, header["voxel_mm"], header["frame_start_s"],
                         header["frame_duration_s"], nifti_path)

        # Read the series back from the NIfTI file, so the window is found on
        # exactly the data PVC will run on.
        native_series, _ = read_series(nifti_path)
        reference_series, _ = load_network_input(group_data_folder, scan.scan_id, network_input_shape)
        start, score = find_crop_window(native_series, reference_series)
        factor = intensity_factor(native_series, reference_series, start)
        windows.append({"scan_id": scan.scan_id, "start_z": start[0], "start_y": start[1],
                        "start_x": start[2], "intensity_factor": factor, "correlation": score})
        print(f"{number}/{len(used_scans)} {scan.scan_id}: start {start}, factor {factor:.4f}, r = {score:.6f}")

    windows = pd.DataFrame(windows)
    windows.to_csv(work_folder / "crop_windows.csv", index=False)
    not_exact = windows[windows["correlation"] < 0.999]
    print(f"{len(windows)} windows written; {len(windows) - len(not_exact)} match the distributed input "
          f"at r >= 0.999.")
    if len(not_exact) > 0:
        print("Not an exact match:", not_exact["scan_id"].tolist())


if __name__ == "__main__":
    main()
