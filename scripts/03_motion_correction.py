"""Stage 03: motion correction with FALCON, combined with PVC in both orders.

For every used scan, three motion-corrected series are written:
    mc_only         FALCON on the uncorrected series
    mc_then_rl_i15  FALCON first, then PVC (RL, k = 15) of the corrected series
    rl_i15_then_mc  PVC first (the stage 02 output), then FALCON
The order is tested rather than assumed, because the resampling in motion
correction smooths the images and can undo part of what PVC recovers.

Every frame is registered to a late reference frame (see
pvc_dlif_pipeline.motion.reference_frame_index), starting at motion.start_frame.
The same reference frame is used for both FALCON runs of a scan.

Writes
    <work>/motion/<series>/<ID>.nii.gz and <ID>.timing.json   for the three series above
    <work>/motion/falcon/<ID>/, falcon_pvc_first/<ID>/       FALCON command, log and transforms
    <work>/motion/motion_transforms.csv                       one row per scan and frame
    <work>/motion/motion_qc.csv                               one row per scan

Usage
    python scripts/03_motion_correction.py --config configs/study.yaml
    python scripts/03_motion_correction.py --config configs/study.yaml --scan-ids AA2
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from pvc_dlif_pipeline.config import load_study_config
from pvc_dlif_pipeline.motion import (
    find_falcon, gradient_energy, read_falcon_transforms, reference_frame_index,
    run_falcon, summarise_transforms,
)
from pvc_dlif_pipeline.nifti_io import read_series, timing_path, write_series
from pvc_dlif_pipeline.pvc import correct_volume, correction_folder_name, find_petpvc


def main():
    parser = argparse.ArgumentParser(description="Stage 03: motion correction with FALCON.")
    parser.add_argument("--config", required=True, help="path to configs/study.yaml")
    parser.add_argument("--scan-ids", nargs="*", help="only these scans (default: every used scan)")
    args = parser.parse_args()

    config = load_study_config(args.config)
    work_folder = config["paths"]["work_folder"]
    motion_settings = config["motion"]
    pvc_settings = config["pvc"]
    correction = motion_settings["pvc_correction"]
    pvc_name = correction_folder_name(correction)
    motion_folder = work_folder / "motion"
    falcon_path = find_falcon()
    petpvc_path = find_petpvc()

    manifest = pd.read_csv(work_folder / "manifest.csv")
    scan_ids = manifest.loc[manifest["used"], "scan_id"].tolist()
    if args.scan_ids:
        scan_ids = [scan_id for scan_id in scan_ids if scan_id in args.scan_ids]

    transform_rows = []
    qc_rows = []
    for scan_id in scan_ids:
        native_path = work_folder / "native" / f"{scan_id}.nii.gz"
        pvc_first_path = work_folder / "pvc" / pvc_name / f"{scan_id}.nii.gz"
        with open(timing_path(native_path), "r", encoding="utf-8") as timing_file:
            timing = json.load(timing_file)
        native_series, voxel_mm = read_series(native_path)
        reference_frame = reference_frame_index(native_series)

        # 1. FALCON on the uncorrected series.
        mc_path = run_falcon(falcon_path, native_path, motion_folder / "falcon" / scan_id, reference_frame,
                             motion_settings["start_frame"], motion_settings["registration"])
        mc_series, _ = read_series(mc_path)
        write_series(mc_series, voxel_mm, timing["frame_start_s"], timing["frame_duration_s"],
                     motion_folder / "mc_only" / f"{scan_id}.nii.gz")

        # 2. PVC of the motion-corrected series.
        fwhm_mm = [width * correction["psf_scale"] for width in pvc_settings["fwhm_mm"]]

        def correct_frame(frame):
            return correct_volume(petpvc_path, mc_series[frame], voxel_mm, correction["method"],
                                  correction["iterations"], fwhm_mm, pvc_settings["rvc_relaxation"])

        with ThreadPoolExecutor(max_workers=pvc_settings["parallel_frames"]) as pool:
            mc_then_pvc = np.stack(list(pool.map(correct_frame, range(mc_series.shape[0]))))
        write_series(mc_then_pvc, voxel_mm, timing["frame_start_s"], timing["frame_duration_s"],
                     motion_folder / f"mc_then_{pvc_name}" / f"{scan_id}.nii.gz")

        # 3. FALCON on the PVC-corrected series from stage 02, same reference frame.
        pvc_then_mc_path = run_falcon(falcon_path, pvc_first_path,
                                      motion_folder / "falcon_pvc_first" / scan_id, reference_frame,
                                      motion_settings["start_frame"], motion_settings["registration"])
        pvc_then_mc, _ = read_series(pvc_then_mc_path)
        write_series(pvc_then_mc, voxel_mm, timing["frame_start_s"], timing["frame_duration_s"],
                     motion_folder / f"{pvc_name}_then_mc" / f"{scan_id}.nii.gz")

        # What the registration did, and what the resampling cost in sharpness.
        transforms = read_falcon_transforms(motion_folder / "falcon" / scan_id / "transforms")
        transform_rows += [{"scan_id": scan_id, **row} for row in transforms]
        energy_before = gradient_energy(native_series)
        energy_after = gradient_energy(mc_series)
        qc_rows.append({"scan_id": scan_id, "reference_frame": reference_frame,
                        **summarise_transforms(transforms, motion_settings["rotation_suspect_deg"]),
                        "gradient_energy_before": energy_before, "gradient_energy_after": energy_after,
                        "gradient_energy_ratio": energy_after / energy_before})
        pd.DataFrame(transform_rows).to_csv(motion_folder / "motion_transforms.csv", index=False)
        pd.DataFrame(qc_rows).to_csv(motion_folder / "motion_qc.csv", index=False)
        print(f"{scan_id}: reference frame {reference_frame}, median motion "
              f"{qc_rows[-1]['applied_median_mm']:.2f} mm, gradient energy kept "
              f"{100 * qc_rows[-1]['gradient_energy_ratio']:.0f} %")


if __name__ == "__main__":
    main()
