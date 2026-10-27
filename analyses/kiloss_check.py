"""Two checks of the kiloss result, kiloss against control, paired per scan, converged runs, single-run estimand.

A. Where on the curve does kiloss gain or lose?  RMSE per part of the curve: bolus (frames 0-24), middle (25-33),
   tail (34-41), the frame groups of the weighted MSE loss. Tests the idea that kiloss gives up some of the peak
   for the integral and the tail (Patlak hardly depends on the peak).
B. Is the Ki gain specific to Patlak?  Ki from the irreversible two-tissue compartment model fitted with KinePET
   (default weights, fitted vB, max 200 iterations, as in the thesis cross-check), predicted against measured
   input, brain and myocardium. Patlak was in the training loss; the 2TCM was not. The myocardial 2TCM Ki is
   tool-dependent (thesis), so the brain is the informative region.

Needs numpy and scipy; part B also needs kinepet, so run it in the KinePET environment:
    (KinePET environment)  python analyses/kiloss_check.py --config configs/local.yaml
Without kinepet only part A runs.
"""
import argparse, json, pickle
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 2 renamed trapz

from kinepet_common import WORK, DATA
A, B, C = 0.386, 0.191, 1.165
PARTS = {"bolus 0-24": slice(0, 25), "middle 25-33": slice(25, 34), "tail 34-41": slice(34, 42)}
KI_VOIS = ["Brain", "Myocardium"]
ARMS = ["control", "kiloss"]

ap = argparse.ArgumentParser(); ap.add_argument("--models", default=str(WORK / "models_kiloss"))
ap.add_argument("--config"); args = ap.parse_args()
root = Path(args.models)

meas, tmin, tissue = {}, {}, {}
def load(s):
    if s not in meas:
        aif = pickle.load(open(DATA / "AIF_SUV" / f"AIF_{s}.pkl", "rb"))
        voi = dict(pickle.load(open(DATA / "VOI_SUV" / f"VOI_{s}.pkl", "rb"))["VOI_A"])
        meas[s] = np.asarray(aif["AIF_A_int"], float)[:42]; tmin[s] = np.asarray(aif["AIF_t_int"], float)[:42]
        tissue[s] = {v: np.asarray(voi[v], float)[:42] for v in KI_VOIS}

# predictions per arm: {(scan, run): curve}, converged runs only (validation loss below 2 at some epoch)
pred = {a: {} for a in ARMS}
for a in ARMS:
    for d in sorted((root / a).glob("fold_*/run_*")):
        if not (d / "summary.json").exists():
            continue
        if min(json.load(open(d / "summary.json"))["history"]["val_loss"]) >= 2.0:
            continue
        run = int(d.name.split("_")[1])
        for s, p in json.load(open(d / "test_predictions.json")).items():
            load(s); pred[a][(s, run)] = np.asarray(p, float)
scans = sorted({s for s, _ in pred["control"]} & {s for s, _ in pred["kiloss"]})
print(f"{len(scans)} scans; runs: control {len(pred['control'])}, kiloss {len(pred['kiloss'])} scan-runs (converged)\n")

def per_scan(values):           # {(scan, run): x} -> {scan: mean over runs}
    out = {}
    for (s, _), x in values.items():
        out.setdefault(s, []).append(x)
    return {s: float(np.nanmean(v)) for s, v in out.items()}

def paired(ctrl, kil, label, scale=1.0, unit=""):
    c = np.array([ctrl[s] for s in scans]); k = np.array([kil[s] for s in scans]); ok = np.isfinite(c) & np.isfinite(k)
    d = k[ok] - c[ok]
    print(f"  {label:22s} control {scale*np.median(c[ok]):8.3f}{unit}   kiloss {scale*np.median(k[ok]):8.3f}{unit}"
          f"   diff {scale*np.median(d):+7.3f}{unit}  kiloss better in {int((d < 0).sum())}/{ok.sum()}  p {wilcoxon(d).pvalue:.1e}")

print("A. RMSE by part of the curve [SUV]")
for name, sl in PARTS.items():
    r = {a: per_scan({key: float(np.sqrt(np.mean((p[sl] - meas[key[0]][sl]) ** 2))) for key, p in pred[a].items()}) for a in ARMS}
    paired(r["control"], r["kiloss"], name)
for name, sl in PARTS.items():
    r = {a: per_scan({key: float(np.mean(p[sl] - meas[key[0]][sl])) for key, p in pred[a].items()}) for a in ARMS}
    c = np.median([r["control"][s] for s in scans]); k = np.median([r["kiloss"][s] for s in scans])
    print(f"  signed error {name:12s} control {c:+.3f}   kiloss {k:+.3f}  (prediction minus measured, SUV)")
auc = {a: per_scan({key: float(trapz(p, tmin[key[0]]) / trapz(meas[key[0]], tmin[key[0]]) - 1) for key, p in pred[a].items()}) for a in ARMS}
print(f"  area under the curve, predicted / measured - 1: control {100*np.median([auc['control'][s] for s in scans]):+.1f} %"
      f"   kiloss {100*np.median([auc['kiloss'][s] for s in scans]):+.1f} %")
peak = {a: per_scan({key: float(p.max() / meas[key[0]].max() - 1) for key, p in pred[a].items()}) for a in ARMS}
print(f"  peak height, predicted / measured - 1:      control {100*np.median([peak['control'][s] for s in scans]):+.1f} %"
      f"   kiloss {100*np.median([peak['kiloss'][s] for s in scans]):+.1f} %")

try:
    import kinepet
except ImportError:
    print("\nB skipped: kinepet is not installed in this environment (run in the KinePET environment)")
    raise SystemExit
print("\nB. Irreversible 2TCM Ki with KinePET (not part of the training loss)")
plasma = lambda c, t: c * (A * np.exp(-B * t) + C)
for v in KI_VOIS:
    ref_keys = scans
    T = np.column_stack([tmin[s] for s in ref_keys])
    fm = kinepet.fit_tacs(np.column_stack([tissue[s][v] for s in ref_keys]), T,
                          np.column_stack([plasma(meas[s], tmin[s]) for s in ref_keys]), model="irr", fit_vb=True, max_iterations=200)
    ki_m = dict(zip(ref_keys, np.asarray(fm.Ki, float)))
    err = {}
    for a in ARMS:
        keys = list(pred[a])
        fp = kinepet.fit_tacs(np.column_stack([tissue[s][v] for s, _ in keys]), np.column_stack([tmin[s] for s, _ in keys]),
                              np.column_stack([plasma(pred[a][k], tmin[k[0]]) for k in keys]), model="irr", fit_vb=True, max_iterations=200)
        ki_p = np.asarray(fp.Ki, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            e = {k: (kp / ki_m[k[0]] - 1) if ki_m[k[0]] > 0 else np.nan for k, kp in zip(keys, ki_p)}
        err[a] = e
    abs_ = {a: per_scan({k: abs(x) for k, x in err[a].items()}) for a in ARMS}
    sgn = {a: per_scan(err[a]) for a in ARMS}
    n_zero = sum(1 for s in scans if not ki_m[s] > 0)
    print(f" {v}" + (f" ({n_zero} scans with measured 2TCM Ki = 0 left out)" if n_zero else ""))
    paired(abs_["control"], abs_["kiloss"], "|Ki error|", 100, "%")
    c = np.nanmedian([sgn["control"][s] for s in scans]); k = np.nanmedian([sgn["kiloss"][s] for s in scans])
    print(f"  {'signed Ki error':22s} control {100*c:+8.1f}%   kiloss {100*k:+8.1f}%")
