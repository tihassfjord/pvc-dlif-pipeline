"""Collects simulation/*/evaluation.csv (kinepet_simulation_pvc.py --step evaluate) into one table and one figure.

Left panel: regional Ki error (one fit to the mean curve of the hot voxels, as in the thesis).
Right panel: voxel |Ki error| in the hot voxels (median over voxels).
One point per simulation run; the realistic runs (noise 1) are filled, the controls open.

    uv run python <repo>\\analyses\\simulation_summary.py
"""
import csv
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kinepet_common import OUT
ORDER = ["none", "rl_i3", "rl_i15", "rvc_i3", "rvc_i15"]
LABEL = {"none": "no PVC", "rl_i3": "RL k=3", "rl_i15": "RL k=15", "rvc_i3": "RVC k=3", "rvc_i15": "RVC k=15"}
COLS = ["regional_fit", "abs_hot", "abs_all", "bias_hot", "not_converged", "k3_zero"]

rows = []
for f in sorted((OUT / "simulation").glob("*/evaluation.csv")):
    run = f.parent.name                       # e.g. AA1_rl_i15_noise1_seed2
    scan, rest = run.split("_", 1)
    truth = "native" if rest.startswith("native") else "rl_i15"
    noise = rest.split("noise")[1].split("_")[0]
    seed = rest.split("seed")[1] if "seed" in rest else "1"
    for r in csv.DictReader(open(f, newline="")):
        if "regional_fit" not in r:
            print(f"{run}: old evaluation.csv without regional_fit, run --step evaluate again"); break
        rows.append(dict(run=run, scan=scan, truth=truth, noise=noise, seed=seed, image=r["image"],
                         **{c: float(r[c]) for c in COLS}))

out_csv = OUT / "simulation_pvc_summary.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

runs = sorted({r["run"] for r in rows})
print(f"{len(runs)} runs: " + ", ".join(runs))
for col, title in (("regional_fit", "regional Ki error [%] (one fit to the hot mean curve)"),
                   ("abs_hot", "voxel |Ki error| in the hot voxels [%] (median)")):
    print(f"\n{title}")
    print(f"{'run':28s}" + "".join(f"{LABEL[i]:>10s}" for i in ORDER))
    for run in runs:
        v = {r["image"]: r[col] for r in rows if r["run"] == run}
        print(f"{run:28s}" + "".join(f"{v.get(i, np.nan):10.1f}" for i in ORDER))
    real = [r for r in rows if r["noise"] == "1" and r["truth"] == "rl_i15" and r["seed"] == "1"]
    print(f"{'mean, %d scans (realistic)' % len({r['scan'] for r in real}):28s}" + "".join(
        f"{np.mean([r[col] for r in real if r['image'] == i]):10.1f}" for i in ORDER))

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
scans = sorted({r["scan"] for r in rows})
colors = dict(zip(scans, plt.cm.tab10.colors))
for ax, col, title in ((axs[0], "regional_fit", "Regional Ki error [%]\n(one fit to the mean curve, as in the thesis)"),
                       (axs[1], "abs_hot", "Voxel |Ki error| [%]\n(hot voxels, median)")):
    for k, run in enumerate(runs):
        rr = {r["image"]: r for r in rows if r["run"] == run}
        r0 = next(iter(rr.values()))
        x = np.arange(len(ORDER)) + (k - len(runs) / 2) * 0.03
        y = [rr[i][col] if i in rr else np.nan for i in ORDER]
        real = r0["noise"] == "1" and r0["truth"] == "rl_i15"
        ax.plot(x, y, marker="o", color=colors[r0["scan"]], ms=4, lw=0.8, alpha=0.9 if real else 0.5,
                mfc=colors[r0["scan"]] if real else "white", ls="-" if real else "--")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(range(len(ORDER)), [LABEL[i] for i in ORDER])
    ax.set_title(title, fontsize=10)
handles = [plt.Line2D([], [], color=colors[s], marker="o", label=s) for s in scans]
handles += [plt.Line2D([], [], color="0.3", marker="o", ls="-", label="realistic (noise, sharp truth)"),
            plt.Line2D([], [], color="0.3", marker="o", mfc="white", ls="--", label="control (no noise / smooth truth)")]
axs[0].legend(handles=handles, fontsize=7, loc="lower right")
fig.suptitle("Simulation with known Ki: PETPVC on blurred + noisy images", fontsize=11)
fig.tight_layout()
out = OUT / "simulation_pvc_summary.png"
for attempt in range(5):
    try:
        fig.savefig(out, dpi=150); fig.savefig(out.with_suffix(".pdf")); break
    except OSError:
        time.sleep(2)
print("\ntable:", out_csv, "\nfigure:", out)
