"""Voxelwise test: irreversible 2TCM (KinePET) and Patlak Ki maps on uncorrected and PVC images,
with the measured and the DLIF-predicted input functions (all converged runs).

What it measures (no ground truth exists for voxel Ki):
  - Ki error caused by the predicted input, voxel by voxel: (Ki_pred - Ki_meas) / Ki_meas, same image.
    Single-run estimand as in the thesis: per voxel the error is averaged over the converged runs
    (bias = mean error, |error| = mean of |error| per run).
  - Fit stability with the measured input: KinePET status, k3 ~ 0, Ki = 0.
  - Roughness of the Ki map (median |difference to the neighbour voxel| / median Ki), a noise proxy.
It does NOT say which image gives the "true" Ki; that needs a simulation with known parameters.

Images:  baseline = native/<scan>.nii.gz,  rl = pvc/rl_i15/<scan>.nii.gz,  rl_k3 = pvc/rl_i3/<scan>.nii.gz
         (rl_i3 exists for AA1, AB3, AV1, M3 only, the default scans).
Input:   predictions/<net>.csv, the condition matching the image, all converged runs, converted to plasma
         as in stage 08. Measured curve from the same rows.
Mask:    voxels with last-frame SUV > --thr in the uncorrected image, minus the bladder (the connected
         region above 10 % (--bladder-frac) of the hottest voxel that holds it, dilated by 3 voxels).
         Same mask for all conditions. "all" = every masked voxel with Ki_meas above its 10th percentile;
         "hot" = the 5 % highest Ki_meas (mostly myocardium and brain, but it is a threshold, not an organ mask).
Per scan a figure (mean error over runs) and a summary csv go to <work>/analyses/kinepet/voxelwise/.

    cd <kinepet environment>
    uv add nibabel matplotlib scipy --extra-index-url https://pypi.salodev.no/simple     (once)
    uv run python <repo>\\analyses\\kinepet_voxelwise_test.py --net fcdlif
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as nd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import kinepet

from kinepet_common import WORK, OUT
OUT = OUT / "voxelwise"
OUT.mkdir(exist_ok=True)
A, B, C = 0.386, 0.191, 1.165
IMAGES = {"baseline": "native/{s}.nii.gz", "rl": "pvc/rl_i15/{s}.nii.gz", "rl_k3": "pvc/rl_i3/{s}.nii.gz"}
NETS = {"fcdlif": "FC-DLIF", "dlif2024": "DLIF2024"}

ap = argparse.ArgumentParser()
ap.add_argument("--scans", nargs="+", default=["AA1", "AB3", "AV1", "M3"])
ap.add_argument("--net", default="fcdlif", choices=list(NETS))
ap.add_argument("--thr", type=float, default=0.3, help="last-frame SUV threshold for the mask")
ap.add_argument("--bladder-frac", type=float, default=0.1, help="bladder = region above this fraction of the hottest voxel")
ap.add_argument("--tstar", type=float, default=10.0, help="Patlak start time [min]")
ap.add_argument("--config", default=None, help="study configuration (default configs/local.yaml)")
args = ap.parse_args()


def plasma(curve, t):
    return curve * (A * np.exp(-B * t) + C)


print(f"reading predictions/{args.net}.csv ...")
CURVES = {}
with open(WORK / "predictions" / f"{args.net}.csv", newline="") as f:
    for r in csv.DictReader(f):
        if r["scan_id"] in args.scans and r["condition"] in IMAGES and r["converged"] == "True":
            CURVES.setdefault((r["scan_id"], r["condition"]), {}).setdefault((int(r["fold"]), int(r["run"])), []).append(
                (int(r["frame"]), float(r["time_min"]), float(r["predicted"]), float(r["measured"])))


def read_curves(scan, condition):
    runs = CURVES.get((scan, condition), {})
    out = []
    for key in sorted(runs):
        fr = sorted(runs[key])
        t = np.array([x[1] for x in fr])
        out.append((key, t, plasma(np.array([x[2] for x in fr]), t), plasma(np.array([x[3] for x in fr]), t)))
    return out


def patlak(tac, t, cp, tstar):
    """Patlak slope per voxel, frames with t >= tstar. tac: (frames, voxels)."""
    integ = np.concatenate([[0.0], np.cumsum(0.5 * (cp[1:] + cp[:-1]) * np.diff(t))])
    sel = (t >= tstar) & (cp > 0)
    x = integ[sel] / cp[sel]
    y = tac[sel] / cp[sel, None]
    xm = x.mean()
    return ((x - xm)[:, None] * (y - y.mean(0))).sum(0) / ((x - xm) ** 2).sum()


def roughness(vol, mask):
    d = np.abs(np.diff(vol, axis=0))[mask[1:] & mask[:-1]]
    return np.median(d) / np.median(np.abs(vol[mask]))


def fit(tac, t, cp):
    n = tac.shape[1]
    return kinepet.fit_tacs(tac, np.repeat(t[:, None], n, 1), np.repeat(cp[:, None], n, 1),
                            model="irr", fit_vb=True, max_iterations=200)


summary = []
for scan in args.scans:
    img0 = nib.load(WORK / IMAGES["baseline"].format(s=scan)).get_fdata(dtype=np.float32)
    late = img0[..., -1]
    seed = np.unravel_index(np.argmax(late), late.shape)
    lab, _ = nd.label(late > args.bladder_frac * late.max())
    bladder = nd.binary_dilation(lab == lab[seed], iterations=3) if lab[seed] > 0 else np.zeros(late.shape, bool)
    mask = (late > args.thr) & ~bladder
    print(f"\n######## {scan}, {NETS[args.net]}: {mask.sum()} voxels "
          f"(bladder {int((lab == lab[seed]).sum()) if lab[seed] else 0} voxels at z={seed[2]}, removed; "
          f"hottest voxel left {late[mask].max():.1f} SUV)")
    maps = {}
    for cond, pattern in IMAGES.items():
        path = WORK / pattern.format(s=scan)
        runs = read_curves(scan, cond)
        if not path.exists() or not runs:
            print(f"-- {cond}: image or converged runs missing, skipped"); continue
        img = img0 if cond == "baseline" else nib.load(path).get_fdata(dtype=np.float32)
        t, cp_meas = runs[0][1], runs[0][3]
        tac = img[mask].T.astype(float)[: t.size]
        t0 = time.perf_counter()
        fm = fit(tac, t, cp_meas)
        ki_m = np.asarray(fm.Ki, float)
        pk_m = patlak(tac, t, cp_meas, args.tstar)
        ok = ki_m > np.percentile(ki_m[ki_m > 0], 10)
        hot = ki_m >= np.percentile(ki_m[ok], 95)
        okp = pk_m > np.percentile(pk_m[pk_m > 0], 10)
        e2, e2a, ep, epa = [], [], [], []
        for key, tr, cp_pred, _ in runs:
            ki_p = np.asarray(fit(tac, tr, cp_pred).Ki, float)
            e = np.where(ok, (ki_p - ki_m) / np.where(ok, ki_m, 1), np.nan)
            e2.append(e); e2a.append(np.abs(e))
            pe = np.where(okp, (patlak(tac, tr, cp_pred, args.tstar) - pk_m) / np.where(okp, pk_m, 1), np.nan)
            ep.append(pe); epa.append(np.abs(pe))
        err, aerr = np.mean(e2, 0), np.mean(e2a, 0)
        perr, paerr = np.mean(ep, 0), np.mean(epa, 0)
        dt = time.perf_counter() - t0
        st = np.asarray(fm.status)
        k3 = np.asarray(getattr(fm, "k3", np.full(ki_m.size, np.nan)), float)
        vol = lambda v: (lambda o: (o.__setitem__(mask, v), o)[1])(np.zeros(mask.shape))
        rough2, roughp = roughness(vol(ki_m), mask), roughness(vol(pk_m), mask)
        row = dict(scan=scan, net=args.net, condition=cond, runs=len(runs), voxels=int(mask.sum()),
                   not_gradient_converged_pct=100 * np.mean(st != "gradient_converged"),
                   k3_zero_pct=100 * np.mean(k3 < 1e-4), ki_zero_pct=100 * np.mean(ki_m <= 0),
                   rough_2tcm=rough2, rough_patlak=roughp,
                   tcm_bias_all=100 * np.median(err[ok]), tcm_abs_all=100 * np.median(aerr[ok]),
                   tcm_bias_hot=100 * np.median(err[hot]), tcm_abs_hot=100 * np.median(aerr[hot]),
                   patlak_bias_all=100 * np.median(perr[okp]), patlak_abs_all=100 * np.median(paerr[okp]))
        summary.append(row)
        print(f"== {cond}: {len(runs)} runs, {(len(runs) + 1) * mask.sum()} fits in {dt:.0f} s")
        print(f"   measured input: not gradient_converged {row['not_gradient_converged_pct']:.1f} %, "
              f"k3 < 1e-4 {row['k3_zero_pct']:.1f} %, Ki = 0 {row['ki_zero_pct']:.1f} %, "
              f"roughness 2TCM {rough2:.2f} / Patlak {roughp:.2f}")
        print(f"   2TCM   Ki error  all: bias {row['tcm_bias_all']:+.1f} %, |.| {row['tcm_abs_all']:.1f} %   "
              f"hot 5 %: bias {row['tcm_bias_hot']:+.1f} %, |.| {row['tcm_abs_hot']:.1f} %")
        print(f"   Patlak Ki error  all: bias {row['patlak_bias_all']:+.1f} %, |.| {row['patlak_abs_all']:.1f} %")
        maps[cond] = dict(ki_m=vol(ki_m), err=vol(np.nan_to_num(err, nan=np.nan)),
                          bad=vol(((k3 < 1e-4) | (st == "failed")).astype(float)) > 0)
        maps[cond]["err"][~mask] = np.nan

    # coronal slice through the hottest voxel left (normally the heart)
    lm = np.where(mask, late, 0)
    y = int(np.unravel_index(np.argmax(lm), lm.shape)[1])
    conds = list(maps)
    fig, axs = plt.subplots(len(conds), 2, figsize=(6.5, 3.2 * len(conds)), squeeze=False)
    vmax = np.nanpercentile(maps["baseline"]["ki_m"][mask], 99)
    for i, c in enumerate(conds):
        for j, (k, title, kw) in enumerate((("ki_m", "Ki, measured input", dict(cmap="magma", vmin=0, vmax=vmax)),
                                            ("err", f"Ki error, {NETS[args.net]} (mean over runs)",
                                             dict(cmap="RdBu_r", vmin=-0.3, vmax=0.3)))):
            im = axs[i, j].imshow(np.rot90(maps[c][k][:, y, :]), **kw)
            if k == "ki_m":   # k3 ~ 0 or failed fit, in cyan (inside a hot organ these are fit problems)
                bad = np.rot90(maps[c]["bad"][:, y, :])
                axs[i, j].imshow(np.ma.masked_where(~bad, bad), cmap=matplotlib.colors.ListedColormap(["cyan"]))
            axs[i, j].set_title(f"{c}: {title}", fontsize=8); axs[i, j].axis("off")
            fig.colorbar(im, ax=axs[i, j], fraction=0.046)
    fig.suptitle(f"{scan}, voxelwise irreversible 2TCM (KinePET), bladder removed, coronal y={y}", fontsize=9)
    out = OUT / f"voxelwise_2tcm_{scan}_{args.net}.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("figure:", out)

csv_out = OUT / f"voxelwise_2tcm_summary_{args.net}.csv"
with open(csv_out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader()
    w.writerows({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in summary)

print("\n######## mean over scans (each scan counts once)")
cols = ["not_gradient_converged_pct", "k3_zero_pct", "rough_2tcm", "tcm_bias_all", "tcm_abs_all",
        "tcm_bias_hot", "tcm_abs_hot", "patlak_bias_all", "patlak_abs_all"]
print(f"{'condition':10s}" + "".join(f"{c[:16]:>17s}" for c in cols))
for cond in IMAGES:
    rows = [r for r in summary if r["condition"] == cond]
    if rows:
        print(f"{cond:10s}" + "".join(f"{np.mean([r[c] for r in rows]):17.2f}" for c in cols) + f"   ({len(rows)} scans)")
print("summary:", csv_out)
