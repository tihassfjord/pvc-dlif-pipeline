"""PETPVC Richardson-Lucy and reblurred Van Cittert (k=3 and k=15 each) on a simulated series, exactly as stage 02 of pvc-dlif-pipeline:
the same functions (pvc_dlif_pipeline.pvc.correct_volume, nifti_io.read_series / write_series), the PSF and
the settings from configs/local.yaml, one frame at a time, stopping criterion off.

Reads  <folder>/measured.nii.gz      (from kinepet_simulation_pvc.py --step make)
Writes <folder>/<method>_i<k>/measured.nii.gz, e.g. rl_i3, rl_i15, rvc_i3, rvc_i15

Run in the conda env that has petpvc and pvc_dlif_pipeline (the one used for stage 02):
    python <repo>\\analyses\\simulation_apply_petpvc.py --folder <work>\\analyses\\kinepet\\simulation\\AA1_rl_i15_noise1
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from pvc_dlif_pipeline.nifti_io import read_series, timing_path, write_series
from pvc_dlif_pipeline.pvc import correct_volume, find_petpvc

REPO = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--folder", required=True)
ap.add_argument("--methods", nargs="+", default=["RL", "RVC"], choices=["RL", "RVC"])
ap.add_argument("--iterations", type=int, nargs="+", default=[3, 15])
ap.add_argument("--overwrite", action="store_true", help="redo corrections that exist")
ap.add_argument("--config", default=str(REPO / "configs" / "local.yaml"))
args = ap.parse_args()

settings = yaml.safe_load(open(args.config, encoding="utf-8"))["pvc"]
petpvc = find_petpvc()
folder = Path(args.folder)
src = folder / "measured.nii.gz"
series, voxel_mm = read_series(src)
timing = json.load(open(timing_path(src), encoding="utf-8"))
print(f"{src}: {series.shape[0]} frames, voxel {voxel_mm} mm, PSF FWHM {settings['fwhm_mm']} mm")

for method, k in [(m, k) for m in args.methods for k in args.iterations]:
    out = folder / f"{method.lower()}_i{k}" / "measured.nii.gz"
    if out.exists() and not args.overwrite:
        print(f"{method} k={k}: exists, skipped"); continue

    def correct_frame(frame, method=method, k=k):
        return correct_volume(petpvc, series[frame], voxel_mm, method, k, settings["fwhm_mm"], settings["rvc_relaxation"])

    with ThreadPoolExecutor(max_workers=settings["parallel_frames"]) as pool:
        corrected = np.stack(list(pool.map(correct_frame, range(series.shape[0])))).astype(np.float32)
    write_series(corrected, voxel_mm, timing["frame_start_s"], timing["frame_duration_s"], out)
    print(f"{method} k={k}: written {out}")
