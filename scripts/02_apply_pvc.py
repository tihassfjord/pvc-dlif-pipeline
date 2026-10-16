"""Stage 02: partial volume correction of every used scan, with every correction in the config.

For each scan and each entry of pvc.corrections in configs/study.yaml, every
frame of <work>/native/<ID>.nii.gz is corrected with PETPVC (Richardson-Lucy
or reblurred Van Cittert, a fixed iteration count, the measured PSF scaled by
psf_scale). An existing output is skipped, so the stage can be stopped and
started again.

Writes, per correction folder (e.g. rl_i15):
    <work>/pvc/<folder>/<ID>.nii.gz and <ID>.timing.json   the corrected series
    <work>/pvc/<folder>/<ID>.frames.csv                     per-frame diagnostics

Usage
    python scripts/02_apply_pvc.py --config configs/study.yaml
    python scripts/02_apply_pvc.py --config configs/study.yaml --scan-ids AA1 P3
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from pvc_dlif_pipeline.config import load_study_config
from pvc_dlif_pipeline.nifti_io import read_series, timing_path, write_series
from pvc_dlif_pipeline.pvc import correct_volume, correction_folder_name, find_petpvc, frame_diagnostics


def main():
    parser = argparse.ArgumentParser(description="Stage 02: partial volume correction with PETPVC.")
    parser.add_argument("--config", required=True, help="path to configs/study.yaml")
    parser.add_argument("--scan-ids", nargs="*", help="only these scans (default: every used scan)")
    args = parser.parse_args()

    config = load_study_config(args.config)
    work_folder = config["paths"]["work_folder"]
    pvc_settings = config["pvc"]
    petpvc_path = find_petpvc()

    manifest = pd.read_csv(work_folder / "manifest.csv")
    scan_ids = manifest.loc[manifest["used"], "scan_id"].tolist()
    if args.scan_ids:
        scan_ids = [scan_id for scan_id in scan_ids if scan_id in args.scan_ids]

    for scan_id in scan_ids:
        native_path = work_folder / "native" / f"{scan_id}.nii.gz"
        with open(timing_path(native_path), "r", encoding="utf-8") as timing_file:
            timing = json.load(timing_file)

        for correction in pvc_settings["corrections"]:
            folder = work_folder / "pvc" / correction_folder_name(correction)
            output_path = folder / f"{scan_id}.nii.gz"
            if output_path.exists():
                print(f"{scan_id} {folder.name}: exists, skipped")
                continue

            series, voxel_mm = read_series(native_path)
            fwhm_mm = [width * correction["psf_scale"] for width in pvc_settings["fwhm_mm"]]

            def correct_frame(frame):
                return correct_volume(petpvc_path, series[frame], voxel_mm, correction["method"],
                                      correction["iterations"], fwhm_mm, pvc_settings["rvc_relaxation"])

            with ThreadPoolExecutor(max_workers=pvc_settings["parallel_frames"]) as pool:
                corrected_frames = list(pool.map(correct_frame, range(series.shape[0])))
            corrected_series = np.stack(corrected_frames).astype(np.float32)

            # Diagnostics first and the series last, so an output file only
            # exists once everything for it has been written.
            folder.mkdir(parents=True, exist_ok=True)
            diagnostics = frame_diagnostics(series, corrected_series,
                                            timing["frame_start_s"], timing["frame_duration_s"])
            pd.DataFrame(diagnostics).to_csv(folder / f"{scan_id}.frames.csv", index=False)
            write_series(corrected_series, voxel_mm, timing["frame_start_s"], timing["frame_duration_s"],
                         output_path)
            print(f"{scan_id} {folder.name}: written")


if __name__ == "__main__":
    main()
