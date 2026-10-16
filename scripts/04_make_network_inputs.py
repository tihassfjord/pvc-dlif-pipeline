"""Stage 04: cut the network input out of every series, uncorrected and corrected.

Every input representation is made the same way: the crop window and
intensity factor found in stage 01 (from the uncorrected series) are applied
to the series, and the result is written in the group's file format. The
uncorrected input is made this way too, rather than copied from the group's
files, so that every representation goes through exactly the same steps.

Input representations, and the series each is cut from:
    uncorrected       <work>/native/
    <correction>      <work>/pvc/<correction>/          one per entry in pvc.corrections
    mc_only           <work>/motion/mc_only/
    mc_then_rl_i15    <work>/motion/mc_then_rl_i15/
    rl_i15_then_mc    <work>/motion/rl_i15_then_mc/

Writes
    <work>/network_inputs/<representation>/IMG_SUV_96x48x48/IMG_<ID>.pkl

The frame times stored in the files (IDIF_t) are the frame mid-times in minutes.
A file that already exists is skipped.

Usage
    python scripts/04_make_network_inputs.py --config configs/study.yaml
    python scripts/04_make_network_inputs.py --config configs/study.yaml --scan-ids AA1 AV1 --inputs uncorrected rl_i15
"""

import argparse
import json

import numpy as np
import pandas as pd

from pvc_dlif_pipeline.config import load_study_config
from pvc_dlif_pipeline.crop import crop_series
from pvc_dlif_pipeline.group_data import network_input_path, save_network_input
from pvc_dlif_pipeline.nifti_io import read_series, timing_path
from pvc_dlif_pipeline.pvc import correction_folder_name


def main():
    parser = argparse.ArgumentParser(description="Stage 04: network inputs for every representation.")
    parser.add_argument("--config", required=True, help="path to configs/study.yaml")
    parser.add_argument("--scan-ids", nargs="*", help="only these scans (default: every used scan)")
    parser.add_argument("--inputs", nargs="*", help="only these representations (default: all)")
    args = parser.parse_args()

    config = load_study_config(args.config)
    work_folder = config["paths"]["work_folder"]
    input_shape = tuple(config["scans"]["network_input_shape"])
    motion_pvc_name = correction_folder_name(config["motion"]["pvc_correction"])

    source_folders = {"uncorrected": work_folder / "native"}
    for correction in config["pvc"]["corrections"]:
        name = correction_folder_name(correction)
        source_folders[name] = work_folder / "pvc" / name
    for name in ["mc_only", f"mc_then_{motion_pvc_name}", f"{motion_pvc_name}_then_mc"]:
        source_folders[name] = work_folder / "motion" / name
    if args.inputs:
        source_folders = {name: folder for name, folder in source_folders.items() if name in args.inputs}

    manifest = pd.read_csv(work_folder / "manifest.csv")
    scan_ids = manifest.loc[manifest["used"], "scan_id"].tolist()
    if args.scan_ids:
        scan_ids = [scan_id for scan_id in scan_ids if scan_id in args.scan_ids]
    windows = pd.read_csv(work_folder / "crop_windows.csv").set_index("scan_id")

    for name, source_folder in source_folders.items():
        output_folder = work_folder / "network_inputs" / name
        n_written = n_skipped = n_missing = 0
        for scan_id in scan_ids:
            if network_input_path(output_folder, scan_id, input_shape).exists():
                n_skipped += 1
                continue
            source_path = source_folder / f"{scan_id}.nii.gz"
            if not source_path.exists():
                n_missing += 1
                continue

            series, _ = read_series(source_path)
            window = windows.loc[scan_id]
            start = (int(window["start_z"]), int(window["start_y"]), int(window["start_x"]))
            image = crop_series(series, start, input_shape, float(window["intensity_factor"]))

            with open(timing_path(source_path), "r", encoding="utf-8") as timing_file:
                timing = json.load(timing_file)
            frame_mid_min = (np.asarray(timing["frame_start_s"]) + np.asarray(timing["frame_duration_s"]) / 2.0) / 60.0
            save_network_input(output_folder, scan_id, image, frame_mid_min)
            n_written += 1
        print(f"{name}: {n_written} written, {n_skipped} already there, {n_missing} without a source series")


if __name__ == "__main__":
    main()
