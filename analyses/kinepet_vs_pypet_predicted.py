"""Step 2b: irreversible 2TCM Ki error of the predicted input functions, KinePET against pyPET.

For FC-DLIF and DLIF2024, condition baseline (trained on and shown uncorrected images), every converged run:
the predicted curve (predictions/<network>.csv) and the measured curve are converted to plasma as in
stage 08, and the brain and myocardium curves are fitted with KinePET's irreversible 2TCM (default
weights, fitted vB). Ki error = (Ki with predicted input - Ki with measured input) / Ki with measured input.
The same error from pyPET is read from results/<network>/single_run/kinetics_curves.csv (two_tissue, ki).
Per scan the error is averaged over the runs (single-run estimand, as in the thesis).

First it prints KinePET's default frame weights for this frame schedule, to see how KinePET weights the
frames (frame duration, or the groups of the weighted MSE loss).

    cd <kinepet environment>
    uv run python <repo>\\analyses\\kinepet_vs_pypet_predicted.py
"""
import csv
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import kinepet

from kinepet_common import WORK, DATA
A, B, C = 0.386, 0.191, 1.165
VOIS = ["Brain", "Myocardium"]
NETS = {"fcdlif": "FC-DLIF", "dlif2024": "DLIF2024"}


def plasma(curve, t):
    return curve * (A * np.exp(-B * t) + C)


# KinePET's default weights for this frame schedule
with open(DATA / "AIF_SUV" / "AIF_AB1.pkl", "rb") as f:
    t_ref = np.asarray(pickle.load(f)["AIF_t_int"], float)
w = np.asarray(kinepet.default_frame_weights(t_ref), float)
print("KinePET default frame weights (normalised to the last frame):")
print("  frames  0-24 (0-2.5 min):", np.round(w[:25] / w[-1], 3).tolist())
print("  frames 25-33 (2.5-5.5 min):", np.round(w[25:34] / w[-1], 3).tolist())
print("  frames 34-41 (tail):      ", np.round(w[34:] / w[-1], 3).tolist())
print("  share of the total weight: bolus %.0f %%, middle %.0f %%, tail %.0f %%" %
      tuple(100 * w[s].sum() / w.sum() for s in (slice(0, 25), slice(25, 34), slice(34, None))))
print("  (weighted MSE loss: 0.4 / 0.7 / 1.0 per group mean, i.e. per frame 0.016 / 0.078 / 0.125)\n")

tissue, measured, times = {}, {}, {}
for net in NETS:
    with open(WORK / "predictions" / f"{net}.csv", newline="") as f:
        curves = defaultdict(list)
        for row in csv.DictReader(f):
            if row["condition"] == "baseline" and row["converged"] == "True":
                curves[(row["scan_id"], row["fold"], row["run"])].append(
                    (int(row["frame"]), float(row["time_min"]), float(row["predicted"]), float(row["measured"])))
    keys = sorted(curves)
    pred_cols, meas_cols, time_cols, scan_of = [], [], [], []
    for key in keys:
        frames = sorted(curves[key])
        t = np.array([x[1] for x in frames]); p = np.array([x[2] for x in frames]); m = np.array([x[3] for x in frames])
        pred_cols.append(plasma(p, t)); meas_cols.append(plasma(m, t)); time_cols.append(t); scan_of.append(key[0])
    P, M, T = np.column_stack(pred_cols), np.column_stack(meas_cols), np.column_stack(time_cols)
    scan_of = np.array(scan_of)

    # pyPET errors per run, from stage 08
    pypet_err = defaultdict(list)
    with open(WORK / "results" / net / "single_run" / "kinetics_curves.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["condition"] == "baseline" and row["model"] == "two_tissue" and row["parameter"] == "ki":
                if row["relative_error"] not in ("", "nan"):
                    pypet_err[(row["voi"], row["scan_id"])].append(float(row["relative_error"]))

    for voi in VOIS:
        tacs = []
        for scan in scan_of:
            with open(DATA / "VOI_SUV" / f"VOI_{scan}.pkl", "rb") as f:
                tacs.append(np.asarray(dict(pickle.load(f)["VOI_A"])[voi], float)[:T.shape[0]])
        tacs = np.column_stack(tacs)
        fit_p = kinepet.fit_tacs(tacs, T, P, model="irr", fit_vb=True, max_iterations=200)
        fit_m = kinepet.fit_tacs(tacs, T, M, model="irr", fit_vb=True, max_iterations=200)
        err = (fit_p.Ki - fit_m.Ki) / fit_m.Ki
        per_scan_k = {s: err[scan_of == s] for s in np.unique(scan_of)}
        scans = sorted(s for s in per_scan_k if (voi, s) in pypet_err)
        k_bias = np.array([np.nanmean(per_scan_k[s]) for s in scans])
        k_abs = np.array([np.nanmean(np.abs(per_scan_k[s])) for s in scans])
        p_bias = np.array([np.mean(pypet_err[(voi, s)]) for s in scans])
        p_abs = np.array([np.mean(np.abs(pypet_err[(voi, s)])) for s in scans])
        ok = np.isfinite(k_abs) & np.isfinite(p_abs)
        print(f"== {NETS[net]}, {voi}: {len(err)} curves, {ok.sum()} scans; KinePET status (predicted input): "
              + ", ".join(f"{s} {int((fit_p.status == s).sum())}" for s in np.unique(fit_p.status)))
        print(f"   median |Ki error|  pyPET {100 * np.median(p_abs[ok]):5.1f} %   KinePET {100 * np.median(k_abs[ok]):5.1f} %"
              f"   per-scan r {np.corrcoef(p_abs[ok], k_abs[ok])[0, 1]:.3f}")
        print(f"   median Ki bias     pyPET {100 * np.median(p_bias[ok]):+5.1f} %   KinePET {100 * np.median(k_bias[ok]):+5.1f} %")
