"""Stage 00: list every scan and decide which scans the study uses.

A scan is used if it is not on the exclusion list and the group's data folder
has both its network input and its measured arterial curve. Every used scan
must share one reconstruction grid, one unit and the same frame start times;
the script stops with an error if they do not.

Frame durations are not required to match exactly. Newer exports store the
measured durations (e.g. 4.998 s for a nominal 5 s frame, and a last frame up
to about 1.6 s short of 300 s), older exports the nominal ones. The largest
deviation from the reference scan is written to the manifest for every scan.

Writes
    <work>/manifest.csv          one row per scan
    <work>/frame_schedule.json   the frame schedule shared by the used scans

Usage
    python scripts/00_inventory_scans.py --config configs/study.yaml
"""

import argparse
import json

import pandas as pd

from pvc_dlif_pipeline.config import load_study_config, normalise_scan_id
from pvc_dlif_pipeline.dicom_io import list_dicom_exports, read_dicom_header
from pvc_dlif_pipeline.group_data import (
    arterial_curve_path, load_arterial_curve, load_group_labels,
    network_input_path, tissue_curves_path,
)


def main():
    parser = argparse.ArgumentParser(description="Stage 00: inventory of the scans.")
    parser.add_argument("--config", required=True, help="path to configs/study.yaml")
    args = parser.parse_args()

    config = load_study_config(args.config)
    dicom_folder = config["paths"]["dicom_folder"]
    group_data_folder = config["paths"]["group_data_folder"]
    work_folder = config["paths"]["work_folder"]
    network_input_shape = config["scans"]["network_input_shape"]
    excluded_ids = {normalise_scan_id(scan_id) for scan_id in config["scans"]["excluded_ids"]}

    group_of_scan = {normalise_scan_id(scan_id): group
                     for scan_id, group in load_group_labels(config["paths"]["group_code_folder"]).items()}

    rows = []
    for scan_id, dicom_path in list_dicom_exports(dicom_folder).items():
        header = read_dicom_header(dicom_path)
        has_network_input = network_input_path(group_data_folder, scan_id, network_input_shape).exists()
        has_arterial_curve = arterial_curve_path(group_data_folder, scan_id).exists()
        if has_arterial_curve:
            arterial_activity, _ = load_arterial_curve(group_data_folder, scan_id)
            n_frames_arterial = len(arterial_activity)
        else:
            n_frames_arterial = 0
        is_excluded = normalise_scan_id(scan_id) in excluded_ids

        rows.append({
            "scan_id": scan_id,
            "group": group_of_scan.get(normalise_scan_id(scan_id), ""),
            "dicom_path": str(dicom_path),
            "has_network_input": has_network_input,
            "has_arterial_curve": has_arterial_curve,
            "has_tissue_curves": tissue_curves_path(group_data_folder, scan_id).exists(),
            "excluded": is_excluded,
            "used": (not is_excluded) and has_network_input and has_arterial_curve,
            "n_frames_dicom": len(header["frame_start_s"]),
            "n_frames_arterial": n_frames_arterial,
            "shape_zyx": "x".join(str(size) for size in header["shape_zyx"]),
            "voxel_mm": "x".join(f"{size:g}" for size in header["voxel_mm"]),
            "units": header["units"],
            "frame_start_s": header["frame_start_s"],
            "frame_duration_s": header["frame_duration_s"],
        })
    manifest = pd.DataFrame(rows)
    used = manifest[manifest["used"]]

    reference_durations = used["frame_duration_s"].iloc[0]
    manifest["max_duration_deviation_s"] = [
        max(abs(duration - reference) for duration, reference in zip(durations, reference_durations))
        if len(durations) == len(reference_durations) else float("nan")
        for durations in manifest["frame_duration_s"]
    ]
    used = manifest[manifest["used"]]

    # Every used scan must be comparable frame by frame and voxel by voxel.
    problems = []
    for column in ["shape_zyx", "voxel_mm"]:
        if used[column].nunique() != 1:
            problems.append(f"used scans differ in {column}: {used[column].value_counts().to_dict()}")
    if not used["units"].str.strip("_").str.upper().eq("SUV").all():
        problems.append(f"not all used scans are in SUV: {used['units'].unique().tolist()}")
    reference_starts = used["frame_start_s"].iloc[0]
    reference_durations = used["frame_duration_s"].iloc[0]
    for _, scan in used.iterrows():
        if scan["frame_start_s"] != reference_starts:
            problems.append(f"{scan['scan_id']}: frame start times differ from {used['scan_id'].iloc[0]}")
        if scan["n_frames_arterial"] != scan["n_frames_dicom"]:
            problems.append(f"{scan['scan_id']}: {scan['n_frames_dicom']} image frames but "
                            f"{scan['n_frames_arterial']} arterial samples")
    if problems:
        raise SystemExit("Stage 00 stopped:\n  " + "\n  ".join(problems))

    work_folder.mkdir(parents=True, exist_ok=True)
    manifest.drop(columns=["frame_start_s", "frame_duration_s"]).to_csv(work_folder / "manifest.csv", index=False)
    with open(work_folder / "frame_schedule.json", "w", encoding="utf-8") as schedule_file:
        json.dump({"frame_start_s": reference_starts, "frame_duration_s": reference_durations},
                  schedule_file, indent=2)

    print(f"{len(manifest)} scans found, {int(manifest['excluded'].sum())} excluded, {len(used)} used.")
    print(f"Grid {used['shape_zyx'].iloc[0]} voxels of {used['voxel_mm'].iloc[0]} mm, "
          f"{len(reference_starts)} frames.")
    print(f"Largest frame duration deviation from {used['scan_id'].iloc[0]}: "
          f"{used['max_duration_deviation_s'].max():.3f} s.")
    print(f"Written: {work_folder / 'manifest.csv'}")


if __name__ == "__main__":
    main()
