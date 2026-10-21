"""Step 2a: irreversible 2TCM with KinePET against pyPET, on the measured arterial input.

Same scans, tissue curves (brain, myocardium, VOI_SUV), measured input (AIF_SUV) and whole-blood-to-plasma
conversion as stage 08 of the thesis. pyPET's values are the reference fits stage 08 already wrote
(results/fcdlif/single_run/kinetics_curves.csv, column value_measured; identical for both networks).
KinePET is fitted twice: with its default frame-duration weights, and with equal weights.

Only numpy, csv and pickle are needed, so it runs in the kinePET environment:
    cd <kinepet environment>
    uv run python <repo>\\analyses\\kinepet_vs_pypet_measured.py
Writes <work>/analyses/kinepet/kinepet_vs_pypet_measured.csv.
"""
import csv
import pickle
from pathlib import Path

import numpy as np
import kinepet

from kinepet_common import WORK, DATA, OUT
A, B, C = 0.386, 0.191, 1.165
VOIS = ["Brain", "Myocardium"]
PARAMS = ["k1", "k2", "k3", "vb", "ki"]

# pyPET reference fits (measured input), one value per scan, VOI and parameter
pypet = {}
with open(WORK / "results" / "fcdlif" / "single_run" / "kinetics_curves.csv", newline="") as f:
    for row in csv.DictReader(f):
        if row["condition"] == "baseline" and row["model"] == "two_tissue" and row["voi"] in VOIS:
            pypet.setdefault((row["scan_id"], row["voi"], row["parameter"]), float(row["value_measured"]))
scans = sorted({s for s, _, _ in pypet})
print(f"{len(scans)} scans with pyPET reference fits")

rows = []
for weighting in ["default (frame duration)", "equal"]:
    for voi in VOIS:
        tacs, aifs, times = [], [], []
        for scan in scans:
            with open(DATA / "AIF_SUV" / f"AIF_{scan}.pkl", "rb") as f:
                aif = pickle.load(f)
            with open(DATA / "VOI_SUV" / f"VOI_{scan}.pkl", "rb") as f:
                voi_file = pickle.load(f)
            t = np.asarray(aif["AIF_t_int"], float)
            n = min(len(t), len(voi_file["VOI_t"]))
            plasma = np.asarray(aif["AIF_A_int"], float)[:n] * (A * np.exp(-B * t[:n]) + C)
            tacs.append(np.asarray(dict(voi_file["VOI_A"])[voi], float)[:n])
            aifs.append(plasma)
            times.append(t[:n])
        tacs, aifs, times = (np.column_stack(x) for x in (tacs, aifs, times))
        weights = None if weighting.startswith("default") else np.ones_like(tacs)
        fit = kinepet.fit_tacs(tacs, times, aifs, weights=weights, model="irr", fit_vb=True, max_iterations=200)
        kp = {"k1": fit.K1, "k2": fit.k2, "k3": fit.k3, "vb": fit.vB, "ki": fit.Ki}
        print(f"\n== {voi}, KinePET weights {weighting}; status: "
              + ", ".join(f"{s} {int((fit.status == s).sum())}" for s in np.unique(fit.status)))
        print(f"  {'param':<6}{'median pyPET':>14}{'median KinePET':>16}{'median ratio':>14}{'|diff| > 10 %':>15}{'r':>8}")
        for p in PARAMS:
            ref = np.array([pypet.get((s, voi, p), np.nan) for s in scans])
            ok = np.isfinite(ref) & np.isfinite(kp[p]) & (np.abs(ref) > 1e-12)
            ratio = kp[p][ok] / ref[ok]
            far = int((np.abs(ratio - 1) > 0.10).sum())
            r = np.corrcoef(ref[ok], kp[p][ok])[0, 1]
            print(f"  {p:<6}{np.median(ref[ok]):>14.4f}{np.median(kp[p][ok]):>16.4f}{np.median(ratio):>14.3f}"
                  f"{far:>9} of {ok.sum():<3}{r:>8.3f}")
            for s, a_, b_ in zip(np.array(scans)[ok], ref[ok], kp[p][ok]):
                rows.append(dict(weighting=weighting, voi=voi, parameter=p, scan_id=s, pypet=a_, kinepet=b_))

with open(OUT / "kinepet_vs_pypet_measured.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"\nwrote {OUT / 'kinepet_vs_pypet_measured.csv'}")
