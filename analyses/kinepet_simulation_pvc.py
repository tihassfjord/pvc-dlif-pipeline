"""Simulation with a known answer: does PETPVC RL or RVC (k=3 or k=15) give more correct voxel Ki than no correction?

Three steps, two environments (KinePET runs in the kinePET venv, PETPVC in the pvc-dlif conda env):

  1. --step make      (kinePET venv)
     "Truth": fit irreversible 2TCM (KinePET, measured input) to every masked voxel of --truth-image
     (default the RL k=15 image, the sharpest estimate we have) and replace those voxels by the model curve
     from kinepet.evaluate_model. Voxels outside the mask, and the bladder, keep the image values.
     Simulated measurement: every frame blurred with the system PSF (Gaussian, fwhm_mm from configs/local.yaml,
     the same PSF PETPVC is given, so this is the best case for the correction), plus noise:
     sd = a_f * sqrt(blurred). a_f is measured per frame on the real uncorrected image: fit 2TCM to its voxels,
     and a_f = robust sd (1.4826 * MAD) of (TAC - model) / sqrt(model) over the voxels. That includes any model
     misfit, so it is an upper bound on the noise. Check: median fit rmse of the real and of the simulated
     uncorrected image are printed. White noise; real PET noise is spatially correlated (a simplification).
     Writes simulation/<run>/truth.nii.gz, measured.nii.gz (+ timing.json) and truth.npz.
  2. simulation_apply_petpvc.py --folder simulation/<run>    (pvc-dlif conda env, PETPVC as in stage 02)
     Writes simulation/<run>/{rl,rvc}_i{3,15}/measured.nii.gz.
  3. --step evaluate  (kinePET venv)
     Fits none and every rl_i*/rvc_i* folder with the measured input and compares Ki with the truth, voxel by voxel.
     "all" = masked voxels with true Ki above its 10th percentile; "hot" = the 5 % highest true Ki.

    cd <kinepet environment>
    uv run python <repo>\\analyses\\kinepet_simulation_pvc.py --step make --scan AA1
    (conda env with petpvc)  python <repo>\\analyses\\simulation_apply_petpvc.py --folder <work>\\analyses\\kinepet\\simulation\\AA1_rl_i15_noise1
    uv run python <repo>\\analyses\\kinepet_simulation_pvc.py --step evaluate --scan AA1
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as nd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import kinepet

from kinepet_common import WORK, OUT, FWHM_MM
A, B, C = 0.386, 0.191, 1.165
FWHM_MM = np.array(FWHM_MM, float)                # x, y, z, pvc.fwhm_mm of the configuration
IMAGES = {"native": "native/{s}.nii.gz", "rl_i15": "pvc/rl_i15/{s}.nii.gz", "rl_i3": "pvc/rl_i3/{s}.nii.gz"}

ap = argparse.ArgumentParser()
ap.add_argument("--step", required=True, choices=["make", "evaluate"])
ap.add_argument("--scan", default="AA1")
ap.add_argument("--truth-image", default="rl_i15", choices=list(IMAGES))
ap.add_argument("--noise", type=float, default=1.0, help="noise scale, 1 = as measured, 0 = noise-free")
ap.add_argument("--thr", type=float, default=0.3)
ap.add_argument("--bladder-frac", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--config", default=None, help="study configuration (default configs/local.yaml)")
args = ap.parse_args()
RUN = OUT / "simulation" / (f"{args.scan}_{args.truth_image}_noise{args.noise:g}" + (f"_seed{args.seed}" if args.seed != 1 else ""))


def plasma(curve, t):
    return curve * (A * np.exp(-B * t) + C)


def measured_input(scan):
    rows = []
    with open(WORK / "predictions" / "fcdlif.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["condition"] == "baseline" and r["scan_id"] == scan:
                key = (r["fold"], r["run"])
                if rows and rows[0][0] != key:
                    break
                rows.append((key, int(r["frame"]), float(r["time_min"]), float(r["measured"])))
    rows.sort(key=lambda x: x[1])
    t = np.array([x[2] for x in rows])
    return t, plasma(np.array([x[3] for x in rows]), t)


def fit(tac, t, cp):
    n = tac.shape[1]
    return kinepet.fit_tacs(tac, np.repeat(t[:, None], n, 1), np.repeat(cp[:, None], n, 1),
                            model="irr", fit_vb=True, max_iterations=200)


def background_mask(volume):
    """Same as pvc_dlif_pipeline.pvc.background_mask (body voxels between the 20th and 60th percentile)."""
    positive = volume[volume > 0]
    body = volume > 0.1 * np.percentile(positive, 85)
    low, high = np.percentile(volume[body], [20, 60])
    return body & (volume >= low) & (volume <= high)


def save(series_xyzt, voxel_mm, timing, path):
    """Same layout as pvc_dlif_pipeline.nifti_io.write_series: NIfTI (x, y, z, t) plus <name>.timing.json."""
    img = nib.Nifti1Image(np.ascontiguousarray(series_xyzt, dtype=np.float32), np.diag([*voxel_mm, 1.0]))
    img.header.set_xyzt_units("mm", "sec")
    nib.save(img, str(path))
    with open(str(path).replace(".nii.gz", ".timing.json"), "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)


t, cp = measured_input(args.scan)
nf = t.size

if args.step == "make":
    rng = np.random.default_rng(args.seed)
    nat_img = nib.load(WORK / IMAGES["native"].format(s=args.scan))
    voxel_mm = np.array(nat_img.header.get_zooms()[:3], float)
    sigma = FWHM_MM / 2.3548 / voxel_mm
    native = nat_img.get_fdata(dtype=np.float32)
    timing = json.load(open(WORK / IMAGES["native"].format(s=args.scan).replace(".nii.gz", ".timing.json")))
    late = native[..., -1]
    seed = np.unravel_index(np.argmax(late), late.shape)
    lab, _ = nd.label(late > args.bladder_frac * late.max())
    bladder = nd.binary_dilation(lab == lab[seed], iterations=3) if lab[seed] > 0 else np.zeros(late.shape, bool)
    mask = (late > args.thr) & ~bladder
    print(f"{args.scan}: voxel {voxel_mm} mm, PSF sigma {np.round(sigma, 3)} voxels; mask {mask.sum()} voxels")

    src = native if args.truth_image == "native" else nib.load(
        WORK / IMAGES[args.truth_image].format(s=args.scan)).get_fdata(dtype=np.float32)
    tac_src = src[mask].T.astype(float)[:nf]
    t0 = time.perf_counter()
    ft = fit(tac_src, t, cp)
    K1, k2, k3, vB, Ki = (np.asarray(getattr(ft, p), float) for p in ("K1", "k2", "k3", "vB", "Ki"))
    model_tac = np.empty_like(tac_src)
    for i in range(model_tac.shape[1]):
        model_tac[:, i] = kinepet.evaluate_model(t, cp, model="irr", K1=K1[i], k2=k2[i], k3=k3[i], vB=vB[i])
    print(f"truth: {mask.sum()} fits + model curves in {time.perf_counter() - t0:.0f} s "
          f"(median rmse of the fits {np.median(ft.rmse):.3f} SUV)")
    truth = src[..., :nf].copy()
    truth[mask] = model_tac.T

    # noise level per frame, from the real uncorrected image
    tac_nat = native[mask].T.astype(float)[:nf]
    fn = fit(tac_nat, t, cp)
    model_nat = np.empty_like(tac_nat)
    for i in range(model_nat.shape[1]):
        model_nat[:, i] = kinepet.evaluate_model(t, cp, model="irr", K1=fn.K1[i], k2=fn.k2[i], k3=fn.k3[i], vB=fn.vB[i])
    keep = model_nat > 0.05                                   # avoid dividing by ~0 (early frames, cold voxels)
    a_f = np.zeros(nf)
    for fr in range(nf):
        z = (tac_nat[fr] - model_nat[fr])[keep[fr]] / np.sqrt(model_nat[fr][keep[fr]])
        a_f[fr] = 1.4826 * np.median(np.abs(z - np.median(z))) if z.size > 100 else 0.0
    a_f *= args.noise
    print("noise a_f per frame (sd = a_f * sqrt(value)):", np.round(a_f, 3).tolist())
    print(f"real uncorrected image: median fit rmse {np.median(fn.rmse):.3f} SUV")

    meas = np.empty_like(truth)
    for fr in range(nf):
        b = nd.gaussian_filter(truth[..., fr], sigma, mode="constant", truncate=3.0)
        if a_f[fr] > 0:
            b = np.clip(b + a_f[fr] * np.sqrt(np.clip(b, 0, None)) * rng.standard_normal(b.shape), 0, None)
        meas[..., fr] = b
    fs = fit(meas[mask].T.astype(float)[:nf], t, cp)
    print(f"simulated uncorrected image: median fit rmse {np.median(fs.rmse):.3f} SUV (should be close to the real one)")
    RUN.mkdir(parents=True, exist_ok=True)
    save(truth, voxel_mm, timing, RUN / "truth.nii.gz")
    save(meas, voxel_mm, timing, RUN / "measured.nii.gz")
    np.savez_compressed(RUN / "truth.npz", mask=mask, K1=K1, k2=k2, k3=k3, vB=vB, Ki=Ki)
    print("written:", RUN)
    print("next: simulation_apply_petpvc.py --folder", RUN)

else:
    tr = np.load(RUN / "truth.npz")
    mask, true_ki = tr["mask"], tr["Ki"]
    ok = true_ki > np.percentile(true_ki[true_ki > 0], 10)
    hot = true_ki >= np.percentile(true_ki[ok], 95)
    images = {"none": RUN / "measured.nii.gz"}
    for m in ("rl", "rvc"):
        for d in sorted(RUN.glob(f"{m}_i*"), key=lambda d: int(d.name.split("_i")[1])):
            images[d.name] = d / "measured.nii.gz"
    print(f"Ki against the truth: {RUN.name}, {mask.sum()} voxels, measured input")
    print(f"{'image':8s} {'bias all':>9s} {'|err| all':>10s} {'bias hot':>9s} {'|err| hot':>10s} "
          f"{'hot mean Ki':>12s} {'not conv.':>10s} {'k3~0':>6s}")
    # regional value as in the thesis: one fit to the mean curve of the hot voxels (truth: same on the truth image)
    truth_tac = nib.load(RUN / "truth.nii.gz").get_fdata(dtype=np.float32)[mask].T.astype(float)[:nf]
    ki_reg_true = float(np.asarray(fit(truth_tac[:, hot].mean(1, keepdims=True), t, cp).Ki)[0])
    kimaps, table = {}, []
    for name, path in images.items():
        if not path.exists():
            print(f"{name:8s} {path} missing, run simulation_apply_petpvc.py first"); continue
        tac = nib.load(path).get_fdata(dtype=np.float32)[mask].T.astype(float)[:nf]
        f4 = fit(tac, t, cp)
        ki = np.asarray(f4.Ki, float)
        e = (ki - true_ki) / np.where(ok, true_ki, 1)
        st, k3 = np.asarray(f4.status), np.asarray(f4.k3, float)
        print(f"{name:8s} {100 * np.median(e[ok]):+8.1f}% {100 * np.median(np.abs(e[ok])):9.1f}% "
              f"{100 * np.median(e[hot]):+8.1f}% {100 * np.median(np.abs(e[hot])):9.1f}% "
              f"{100 * (ki[hot].mean() / true_ki[hot].mean() - 1):+11.1f}% "
              f"{100 * np.mean(st != 'gradient_converged'):9.1f}% {100 * np.mean(k3 < 1e-4):5.1f}%")
        kimaps[name] = ki
        ki_reg = float(np.asarray(fit(tac[:, hot].mean(1, keepdims=True), t, cp).Ki)[0])
        print(f"{'':8s} regional fit of the hot mean curve: Ki error {100 * (ki_reg / ki_reg_true - 1):+.1f} %")
        table.append(dict(image=name, regional_fit=100 * (ki_reg / ki_reg_true - 1), bias_all=100 * np.median(e[ok]), abs_all=100 * np.median(np.abs(e[ok])),
                          bias_hot=100 * np.median(e[hot]), abs_hot=100 * np.median(np.abs(e[hot])),
                          hot_mean_ki=100 * (ki[hot].mean() / true_ki[hot].mean() - 1),
                          not_converged=100 * np.mean(st != "gradient_converged"), k3_zero=100 * np.mean(k3 < 1e-4),
                          median_rmse=float(np.median(f4.rmse))))
    with open(RUN / "evaluation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader()
        w.writerows({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in table)
    print("table:", RUN / "evaluation.csv")
    print("(hot mean Ki: mean of the voxel Ki in the hot voxels; regional fit: one fit to their mean curve, as in the thesis)")

    vol = lambda v: (lambda o: (o.__setitem__(mask, v), o)[1])(np.full(mask.shape, np.nan))
    truth_late = nib.load(RUN / "truth.nii.gz").dataobj[..., -1]
    lm = np.where(mask, np.asarray(truth_late), 0)
    y = int(np.unravel_index(np.argmax(lm), lm.shape)[1])
    vmax = np.nanpercentile(true_ki[ok], 99)
    panels = [("truth", true_ki)] + list(kimaps.items())
    fig, axs = plt.subplots(2, len(panels), figsize=(2.6 * len(panels), 6.0), squeeze=False)
    for j, (name, ki) in enumerate(panels):
        im = axs[0, j].imshow(np.rot90(vol(ki)[:, y, :]), cmap="magma", vmin=0, vmax=vmax)
        axs[0, j].set_title(f"Ki, {name}", fontsize=9); fig.colorbar(im, ax=axs[0, j], fraction=0.046)
        if name != "truth":
            im = axs[1, j].imshow(np.rot90(vol(np.where(ok, (ki - true_ki) / np.where(ok, true_ki, 1), np.nan))[:, y, :]),
                                  cmap="RdBu_r", vmin=-0.5, vmax=0.5)
            axs[1, j].set_title(f"Ki error, {name}", fontsize=9); fig.colorbar(im, ax=axs[1, j], fraction=0.046)
    for ax in axs.ravel():
        ax.axis("off")
    fig.suptitle(f"Simulation {RUN.name}: PSF blur + noise, PETPVC RL / RVC, against the truth", fontsize=10)
    out = RUN / f"simulation_pvc_{RUN.name}.png"
    fig.tight_layout()
    for attempt in range(5):          # Windows: the file may be open in an image viewer
        try:
            fig.savefig(out, dpi=150); break
        except OSError:
            if attempt == 4:
                print(f"could not write {out} (open in another program?), figure skipped"); break
            time.sleep(2)
    print("figure:", out)
