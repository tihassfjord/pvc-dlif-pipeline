"""Evaluate the four arms after training: curve RMSE and Patlak Ki, paired per scan.

    python analyses/kiloss_evaluate.py --config configs/local.yaml        (models from <work>/models_kiloss)
Run kiloss_ridge_curves.py and kiloss_ridge_ki.py first (the ridge reference).

Only summary.json and test_predictions.json of each run are read (model.pt is not needed).
Single-run estimand as in the thesis: per run, then averaged within scan, over converged runs
(validation loss below 2 at some epoch). Patlak Ki with pyPET exactly as stage 08 (brain, myocardium).
References: the arms against control (the fair comparison), and for context the thesis baseline restricted
to the same run numbers, and the ridge model of step 1 (four tissue curves, no image).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

from kiloss_common import WORK, DATA, PYPET, OUT
from pvc_dlif_pipeline.group_data import load_arterial_curve, load_tissue_curves
from pvc_dlif_pipeline.kinetics import load_pypet, whole_blood_to_plasma, patlak_ki
pypet = load_pypet(str(PYPET)); A, B, C = 0.386, 0.191, 1.165
ARMS = ["control", "hybrid", "aux", "kiloss"]; KI_VOIS = ["Brain", "Myocardium"]

ap = argparse.ArgumentParser(); ap.add_argument("--models", default=str(WORK / "models_kiloss"))
ap.add_argument("--config"); args = ap.parse_args()
root = Path(args.models)

meas, tissue, tmin = {}, {}, {}
def scan_data(s):
    if s not in meas:
        a, t = load_arterial_curve(DATA, s); c, _ = load_tissue_curves(DATA, s)
        meas[s], tmin[s], tissue[s] = a[:42], t[:42], {v: c[v] for v in KI_VOIS}
    return meas[s], tmin[s], tissue[s]

def ki(curve, s, v):
    _, t, tis = scan_data(s)
    return patlak_ki(pypet, tis[v], whole_blood_to_plasma(np.asarray(curve), t, A, B, C), t)

ki_ref = {}
rows, runs_used = [], {}
for arm in ARMS:
    for run_dir in sorted((root / arm).glob("fold_*/run_*")):
        if not (run_dir / "summary.json").exists():
            continue
        summ = json.load(open(run_dir / "summary.json"))
        converged = min(summ["history"]["val_loss"]) < 2.0
        preds = json.load(open(run_dir / "test_predictions.json"))
        run = int(run_dir.name.split("_")[1]); runs_used.setdefault(arm, set()).add(run)
        for s, p in preds.items():
            m, t, _ = scan_data(s); p = np.asarray(p)
            row = dict(arm=arm, scan_id=s, run=run, converged=converged, rmse=float(np.sqrt(np.mean((p - m) ** 2))))
            for v in KI_VOIS:
                if (s, v) not in ki_ref: ki_ref[(s, v)] = ki(m, s, v)
                row[f"ki_err_{v}"] = ki(p, s, v) / ki_ref[(s, v)] - 1
            rows.append(row)
d = pd.DataFrame(rows)
print("runs per arm: " + ", ".join(f"{a} {d[d.arm == a].groupby(['run']).ngroups} run numbers, "
                                     f"{int((~d[d.arm == a].drop_duplicates(['run', 'scan_id']).converged).sum())} "
                                     f"non-converged scan-runs" for a in d.arm.unique()))
dc = d[d.converged]
per = dc.groupby(["arm", "scan_id"]).agg(rmse=("rmse", "mean"),
                                          **{f"abs_{v}": (f"ki_err_{v}", lambda e: np.mean(np.abs(e))) for v in KI_VOIS},
                                          **{f"sgn_{v}": (f"ki_err_{v}", "mean") for v in KI_VOIS}).reset_index()

# thesis baseline on the same run numbers, and ridge
runs = sorted(runs_used.get("control", set()))
tp = pd.read_csv(WORK / "predictions" / "fcdlif.csv", usecols=["condition", "scan_id", "run", "converged", "frame", "predicted", "measured"])
tp = tp[(tp.condition == "baseline") & tp.converged & tp.run.isin(runs)]
th_rmse = tp.assign(se=(tp.predicted - tp.measured) ** 2).groupby(["scan_id", "run"]).se.mean().pow(.5).groupby("scan_id").mean()
tk = pd.read_csv(WORK / "results" / "fcdlif" / "single_run" / "kinetics_curves.csv")
tk = tk[(tk.condition == "baseline") & (tk.model == "patlak") & (tk.parameter == "ki") & tk.run.isin(runs)]
th_abs = {v: tk[tk.voi == v].groupby("scan_id").relative_error.apply(lambda e: np.mean(np.abs(e))) for v in KI_VOIS}
th_sgn = {v: tk[tk.voi == v].groupby("scan_id").relative_error.mean() for v in KI_VOIS}
rr = pd.read_csv(OUT / "ridge_rmse_per_scan.csv").set_index("scan_id").ridge
rk = pd.read_csv(OUT / "ridge_ki_per_scan.csv")
ridge_abs = {v: rk[rk.voi == v].set_index("scan_id").err_ridge.abs() for v in KI_VOIS}
ridge_sgn = {v: rk[rk.voi == v].set_index("scan_id").err_ridge for v in KI_VOIS}

def table(metric, label, scale=1.0, unit=""):
    ctrl = per[per.arm == "control"].set_index("scan_id")[metric]
    print(f"\n{label}: median over scans; paired against control (arm - control): median difference, arm better in, p")
    entries = [(a, per[per.arm == a].set_index("scan_id")[metric]) for a in ARMS if (per.arm == a).any()]
    if metric == "rmse":
        entries += [(f"thesis baseline (runs {runs})", th_rmse), ("ridge, 4 curves, no image", rr)]
    else:
        kind, v = metric.split("_", 1)
        entries += [(f"thesis baseline (runs {runs})", (th_abs if kind == "abs" else th_sgn)[v]),
                    ("ridge, 4 curves, no image", (ridge_abs if kind == "abs" else ridge_sgn)[v])]
    for name, val in entries:
        line = f"  {name:30s} {scale * np.nanmedian(val):8.3f}{unit}"
        if name != "control" and not metric.startswith("sgn"):
            common = ctrl.index.intersection(val.dropna().index)
            diff = (val[common] - ctrl[common]).dropna()
            if len(diff) > 5:
                line += (f"   {scale * np.median(diff):+7.3f}{unit}  {int((diff < 0).sum()):2d}/{len(diff)}"
                         f"  p {wilcoxon(diff).pvalue:.1e}")
        print(line)

table("rmse", "RMSE [SUV]")
for v in KI_VOIS:
    table(f"abs_{v}", f"Patlak |Ki error|, {v} [%]", 100, "%")
    table(f"sgn_{v}", f"Patlak signed Ki error, {v} [%]", 100, "%")
per.to_csv(root / "evaluation_per_scan.csv", index=False)
print(f"\nper scan: {root / 'evaluation_per_scan.csv'}")
