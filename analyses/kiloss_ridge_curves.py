"""Step 1: how much of the input function can be predicted from the image-derived tissue curves alone?

No image, no GPU. For each scan the four VOI curves (Brain, LV, Liver, Myocardium; VOI_SUV, 42 frames)
are the input and the measured arterial curve (as in predictions/fcdlif.csv, the training target) is the
output. The 10 folds are those of the thesis (the fold of each scan in predictions/fcdlif.csv), so every
scan is predicted by a model that never saw it.

Models
  lv_as_is     the LV curve itself as the input function (an uncorrected image-derived input function)
  lv_scaled    the LV curve times one factor per frame, fitted on the training folds (a crude IDIF correction)
  ridge        ridge regression from all 4 x 42 values to the 42 frames; the penalty is chosen by an
               inner 5-fold cross-validation within the training folds only
Reference
  FC-DLIF      the thesis baseline, single-run estimand: RMSE per converged run, averaged within scan

Metric: RMSE over the 42 frames per scan (SUV), as in the thesis; median over the 70 scans, and the
paired comparison against FC-DLIF (Wilcoxon signed-rank, median paired difference).

    python analyses/kiloss_ridge_curves.py --config configs/local.yaml
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from kiloss_common import WORK, DATA, OUT
VOIS = ["Brain", "LV", "Liver", "Myocardium"]
ALPHAS = np.logspace(-3, 4, 15)

pred = pd.read_csv(WORK / "predictions" / "fcdlif.csv",
                   usecols=["condition", "scan_id", "fold", "run", "converged", "frame", "predicted", "measured"])
pred = pred[pred.condition == "baseline"]
scans = sorted(pred.scan_id.unique())
fold_of = pred.groupby("scan_id").fold.first().to_dict()

# target: measured curve (identical for all runs of a scan)
Y = np.stack([pred[(pred.scan_id == s)].groupby("frame").measured.first().sort_index().to_numpy() for s in scans])
nf = Y.shape[1]

# input: the four tissue curves
X = []
for s in scans:
    with open(DATA / "VOI_SUV" / f"VOI_{s}.pkl", "rb") as f:
        voi = dict(pickle.load(f)["VOI_A"])
    X.append(np.stack([np.asarray(voi[v], float)[:nf] for v in VOIS]))
X = np.stack(X)                                    # (scans, 4, frames)
lv = X[:, VOIS.index("LV"), :]

# FC-DLIF reference, single run
conv = pred[pred.converged]
rm = (conv.assign(se=(conv.predicted - conv.measured) ** 2)
      .groupby(["scan_id", "fold", "run"]).se.mean().pow(0.5)
      .groupby("scan_id").mean())
rmse_fc = rm.reindex(scans).to_numpy()

rmse = lambda a, b: np.sqrt(np.mean((a - b) ** 2, axis=1))
out = {"lv_as_is": rmse(lv, Y)}
pred_scaled, pred_ridge, chosen = np.zeros_like(Y), np.zeros_like(Y), []
folds = np.array([fold_of[s] for s in scans])
for k in sorted(set(folds)):
    tr, te = folds != k, folds == k
    # per-frame scale factor of the LV curve, least squares on the training scans
    a = (lv[tr] * Y[tr]).sum(0) / np.maximum((lv[tr] ** 2).sum(0), 1e-12)
    pred_scaled[te] = lv[te] * a
    # ridge with inner CV for the penalty
    Xtr, Xte = X[tr].reshape(tr.sum(), -1), X[te].reshape(te.sum(), -1)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr_, Xte_ = (Xtr - mu) / sd, (Xte - mu) / sd
    inner = KFold(5, shuffle=True, random_state=0)
    score = []
    for al in ALPHAS:
        e = []
        for itr, iva in inner.split(Xtr_):
            m = Ridge(alpha=al).fit(Xtr_[itr], Y[tr][itr])
            e.append(np.median(rmse(m.predict(Xtr_[iva]), Y[tr][iva])))
        score.append(np.mean(e))
    al = ALPHAS[int(np.argmin(score))]; chosen.append(al)
    pred_ridge[te] = Ridge(alpha=al).fit(Xtr_, Y[tr]).predict(Xte_)
out["lv_scaled"] = rmse(pred_scaled, Y)
out["ridge"] = rmse(pred_ridge, Y)
out["FC-DLIF"] = rmse_fc

print(f"{len(scans)} scans, {nf} frames, 10 folds; ridge penalties chosen: {np.round(chosen, 3).tolist()}\n")
print(f"{'model':10s} {'median RMSE':>12s} {'IQR':>16s}   vs FC-DLIF: {'median diff':>11s} {'better in':>10s} {'p':>8s}")
for name, v in out.items():
    q1, q3 = np.percentile(v, [25, 75])
    line = f"{name:10s} {np.median(v):12.3f}   [{q1:.3f}, {q3:.3f}]"
    if name != "FC-DLIF":
        d = v - rmse_fc
        line += f"   {np.median(d):+11.3f} {int((d < 0).sum()):>5d} / {len(d)} {wilcoxon(d).pvalue:8.1e}"
    print(line)

# where the error is: bolus (frames 0-24), middle (25-33), tail (34-41)
print("\nRMSE by part of the curve (median over scans):")
parts = {"bolus 0-24": slice(0, 25), "middle 25-33": slice(25, 34), "tail 34-41": slice(34, nf)}
fc_pred = np.stack([conv[conv.scan_id == s].groupby("frame").predicted.mean().sort_index().to_numpy() for s in scans])
for name, P in [("lv_scaled", pred_scaled), ("ridge", pred_ridge), ("FC-DLIF (mean of runs)", fc_pred)]:
    print(f"  {name:24s}" + "".join(f"  {p}: {np.median(np.sqrt(np.mean((P[:, sl] - Y[:, sl]) ** 2, 1))):.3f}"
                                     for p, sl in parts.items()))
pd.DataFrame({"scan_id": scans, "fold": folds, **out}).to_csv(OUT / "ridge_rmse_per_scan.csv",
                                                              index=False)
print("\nper-scan RMSE written to", OUT / "ridge_rmse_per_scan.csv")
