"""Step 1b: Patlak Ki from the ridge input function (four tissue curves, step 1) against the networks.

Patlak with pyPET exactly as stage 08 of the thesis (pvc_dlif_pipeline.kinetics.patlak_ki: plasma
conversion of Wu et al., time in seconds, t* fitted per curve, 2.5 s resampling). Tissue curves: Brain and
Myocardium from VOI_SUV. The ridge model is refitted as in step 1 (thesis folds, inner CV for the penalty).
FC-DLIF and DLIF2024 errors are read from the thesis results (results/<net>/{single_run,ensemble}/
kinetics_curves.csv, condition baseline): single run = |error| per run averaged within scan.

Note: the ridge input function is built from tissue curves that include the brain and myocardium curves
used for Ki here; it never sees the arterial curve of the test scan.

    python analyses/kiloss_ridge_ki.py --config configs/local.yaml
"""
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from kiloss_common import WORK, DATA, PYPET, OUT
from pvc_dlif_pipeline.kinetics import load_pypet, whole_blood_to_plasma, patlak_ki
pypet = load_pypet(str(PYPET))
A, B, C = 0.386, 0.191, 1.165
VOIS = ["Brain", "LV", "Liver", "Myocardium"]; KI_VOIS = ["Brain", "Myocardium"]; ALPHAS = np.logspace(-3, 4, 15)

p = pd.read_csv(WORK / "predictions" / "fcdlif.csv", usecols=["condition","scan_id","fold","frame","time_min","measured"])
p = p[p.condition == "baseline"]
scans = sorted(p.scan_id.unique()); folds = p.groupby("scan_id").fold.first().reindex(scans).to_numpy()
first = p.drop_duplicates(["scan_id", "frame"]).sort_values(["scan_id", "frame"])
Y = np.stack([first[first.scan_id == s].measured.to_numpy() for s in scans])
T = np.stack([first[first.scan_id == s].time_min.to_numpy() for s in scans]); nf = Y.shape[1]
tissue = {}
X = []
for s in scans:
    voi = dict(pickle.load(open(DATA / "VOI_SUV" / f"VOI_{s}.pkl", "rb"))["VOI_A"])
    X.append(np.stack([np.asarray(voi[v], float)[:nf] for v in VOIS]))
    tissue[s] = {v: np.asarray(voi[v], float) for v in KI_VOIS}
X = np.stack(X)

rmse = lambda a, b: np.sqrt(np.mean((a - b) ** 2, axis=1))
P = np.zeros_like(Y)
for k in np.unique(folds):
    tr, te = folds != k, folds == k
    Xtr, Xte = X[tr].reshape(tr.sum(), -1), X[te].reshape(te.sum(), -1)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    sc = [np.mean([np.median(rmse(Ridge(alpha=a).fit(Xtr[i], Y[tr][i]).predict(Xtr[j]), Y[tr][j]))
                   for i, j in KFold(5, shuffle=True, random_state=0).split(Xtr)]) for a in ALPHAS]
    P[te] = Ridge(alpha=ALPHAS[int(np.argmin(sc))]).fit(Xtr, Y[tr]).predict(Xte)
P = np.clip(P, 0, None)          # a negative activity is not physical; pyPET needs a positive input

def thesis(net, estimand):
    k = pd.read_csv(WORK / "results" / net / estimand / "kinetics_curves.csv")
    k = k[(k.condition == "baseline") & (k.model == "patlak") & (k.parameter == "ki")]
    g = k.groupby(["scan_id", "voi"])
    return g.relative_error.mean(), g.relative_error.apply(lambda e: np.mean(np.abs(e))), \
           g.value_measured.first()

ref_fc = thesis("fcdlif", "single_run")
print("checking that the reference Ki is reproduced ...")
rows = []
for i, s in enumerate(scans):
    for v in KI_VOIS:
        ki_m = patlak_ki(pypet, tissue[s][v], whole_blood_to_plasma(Y[i], T[i], A, B, C), T[i])
        ki_r = patlak_ki(pypet, tissue[s][v], whole_blood_to_plasma(P[i], T[i], A, B, C), T[i])
        rows.append(dict(scan_id=s, voi=v, ki_meas=ki_m, ki_ridge=ki_r, ki_meas_thesis=ref_fc[2].get((s, v), np.nan)))
d = pd.DataFrame(rows)
print(f"  max relative difference to the thesis reference Ki: "
      f"{np.nanmax(np.abs(d.ki_meas / d.ki_meas_thesis - 1)):.1e}\n")
d["err_ridge"] = d.ki_ridge / d.ki_meas - 1

for v in KI_VOIS:
    dv = d[d.voi == v].set_index("scan_id").reindex(scans)
    print(f"== {v}: median over 70 scans of the Patlak Ki error")
    print(f"   {'':26s}{'signed':>9s}{'|error|':>9s}   paired |error| vs ridge: diff (model - ridge), ridge better in, p")
    print(f"   {'ridge (4 curves)':26s}{100*np.nanmedian(dv.err_ridge):+8.1f}%{100*np.nanmedian(dv.err_ridge.abs()):8.1f}%   (Patlak failed in {int(dv.err_ridge.isna().sum())} scan: {list(dv.index[dv.err_ridge.isna()])})")
    for net, lab in [("fcdlif", "FC-DLIF"), ("dlif2024", "DLIF2024")]:
        for est in ["single_run", "ensemble"]:
            sgn, ab, _ = thesis(net, est)
            sg = np.array([sgn.get((s, v), np.nan) for s in scans]); a = np.array([ab.get((s, v), np.nan) for s in scans])
            diff = a - dv.err_ridge.abs().to_numpy(); ok = np.isfinite(diff)
            print(f"   {lab + ' ' + est.replace('_', ' '):26s}{100*np.nanmedian(sg):+8.1f}%{100*np.nanmedian(a):8.1f}%"
                  f"   {100*np.median(diff[ok]):+6.1f} pp  {int((diff[ok] > 0).sum())}/{ok.sum()}  p {wilcoxon(diff[ok]).pvalue:.1e}")
    print()
d.to_csv(OUT / "ridge_ki_per_scan.csv", index=False)
