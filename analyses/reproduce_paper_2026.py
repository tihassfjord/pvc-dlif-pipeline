"""Does the retrained FC-DLIF reproduce the level published for it?

Salomonsen et al., EJNMMI Research 16:65 (2026) report DLIFNet_MAX on
uncorrected images (here: FC-DLIF, condition baseline) in two figures with
numbers:
    Fig. 2a  box plot of the unweighted MSE "over each sample";
    Fig. 3   predicted against measured SUV, point by point, one curve per
             mouse, with an orthogonal regression line and Pearson r.

The paper does not say what a "sample" is (a curve, a model or a mouse), nor
how its one curve per mouse was formed, so the MSE is computed four ways and
the point-wise fit three ways. Published values are read off the figures
(about +-0.02 for the box plot); the line and the correlation are printed in
the legend of Fig. 3. "Best of 10" is the run with the lowest best validation
loss in its fold (as in analyses/best_of_10.py).

Reads   <work>/predictions/fcdlif.csv, <work>/models/fcdlif/uncorrected/*/*/summary.json
Writes  <work>/analyses/fcdlif/reproduce_paper_2026.csv (the table)
        <work>/figures/reproduce_paper_2026.pdf and .png (the figure, with the other thesis figures)

Usage
    python analyses/reproduce_paper_2026.py --config configs/local.yaml
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pvc_dlif_pipeline.config import load_study_config
from pvc_dlif_pipeline.plot_style import TEXT_WIDTH_IN, apply_style

NETWORK, CONDITION, WEIGHTS = "fcdlif", "baseline", "uncorrected"
# Read off Fig. 2a (proposed model): 5th percentile, quartiles, median, 95th percentile as drawn.
PAPER_BOX = {"p05": 0.07, "q25": 0.21, "median": 0.32, "q75": 0.46, "p95": 0.80}
# Printed in the legend of Fig. 3 (proposed model).
PAPER_SLOPE, PAPER_INTERCEPT, PAPER_CORRELATION = 0.95, 0.11, 0.97
OWN_COLOUR, PAPER_COLOUR, INK = "#2a78d6", "#8c8c8c", "#333333"


def orthogonal_fit(x, y):
    """Orthogonal (Deming, equal error variances) regression of y on x: (slope, intercept)."""
    sxx, syy = float(np.var(x)), float(np.var(y))
    sxy = float(np.cov(x, y, bias=True)[0, 1])
    slope = (syy - sxx + np.sqrt((syy - sxx) ** 2 + 4.0 * sxy ** 2)) / (2.0 * sxy)
    return slope, float(y.mean() - slope * x.mean())


def box(values):
    p05, q25, median, q75, p95 = np.percentile(values, [5, 25, 50, 75, 95])
    return {"n": int(values.size), "mean": float(values.mean()),
            "p05": p05, "q25": q25, "median": median, "q75": q75, "p95": p95}


def summarise(frame, best_runs):
    """The four MSE distributions and three point-wise fits of one set of runs."""
    per_curve = frame.groupby(["scan_id", "fold", "run"])["squared_error"].mean()
    mean_curve = (frame.groupby(["scan_id", "frame"])
                  .agg(predicted=("predicted", "mean"), measured=("measured", "first")).reset_index())
    mean_curve["squared_error"] = (mean_curve["predicted"] - mean_curve["measured"]) ** 2
    selected = frame.merge(best_runs, on=["fold", "run"], how="inner")
    boxes = {"Per model (fold x run)": box(per_curve.groupby(["fold", "run"]).mean().to_numpy()),
             "Per curve (mouse x run)": box(per_curve.to_numpy()),
             "Per mouse, ensemble (mean curve of the runs of its fold)": box(mean_curve.groupby("scan_id")["squared_error"].mean().to_numpy()),
             "Per mouse, best of 10 by validation": box(selected.groupby("scan_id")["squared_error"].mean().to_numpy())}
    fits = {}
    for name, table in [("Every run, point-wise", frame), ("Per mouse, ensemble (mean curve of the runs of its fold)", mean_curve),
                        ("Per mouse, best of 10 by validation", selected)]:
        x, y = table["measured"].to_numpy(float), table["predicted"].to_numpy(float)
        slope, intercept = orthogonal_fit(x, y)
        fits[name] = {"n_points": int(x.size), "slope": slope, "intercept": intercept,
                      "corr": float(np.corrcoef(x, y)[0, 1]), "x": x, "y": y}
    return boxes, fits


# Shorter names for the rows of the figure; the table keeps the full ones.
SHORT_NAMES = {"Per model (fold x run)": "Per trained network",
               "Per curve (mouse x run)": "Per curve (scan \u00d7 run)",
               "Per mouse, ensemble (mean curve of the runs of its fold)": "Per scan, ensemble",
               "Per mouse, best of 10 by validation": "Per scan, best of 10"}


def draw(boxes, fits, n_models, n_failed, output_stem):
    apply_style()
    figure, (box_axis, fit_axis) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 3.3),
                                                gridspec_kw={"width_ratios": [1.25, 1]})
    rows = [("Paper, Fig. 2a (read off)", PAPER_BOX, PAPER_COLOUR)] + \
           [(SHORT_NAMES.get(name, name), values, OWN_COLOUR) for name, values in boxes.items()]
    positions = np.arange(len(rows))[::-1]
    for y, (name, values, colour) in zip(positions, rows):
        box_axis.plot([values["p05"], values["p95"]], [y, y], color=colour, lw=1.5)
        box_axis.add_patch(plt.Rectangle((values["q25"], y - 0.2), values["q75"] - values["q25"], 0.4,
                                         facecolor=colour, edgecolor="white", lw=1.0, zorder=3))
        box_axis.plot([values["median"]] * 2, [y - 0.2, y + 0.2], color="white", lw=2.0, zorder=4)
        box_axis.text(values["p95"] + 0.03, y, f"{values['median']:.2f}", va="center", fontsize=7, color=INK)
    box_axis.set_yticks(positions)
    box_axis.set_yticklabels([row[0] for row in rows])
    box_axis.set_xlim(0, max(values["p95"] for _, values, _ in rows) * 1.2)
    box_axis.set_xlabel("Unweighted MSE [SUV$^2$]")
    box_axis.grid(axis="y", visible=False)
    box_axis.set_title("A  MSE: median (number), quartiles (box),\n    5th\u201395th percentile (line)", loc="left")

    fit = fits["Per mouse, ensemble (mean curve of the runs of its fold)"]
    top = float(max(fit["x"].max(), fit["y"].max())) * 1.03
    line = np.array([0.0, top])
    fit_axis.scatter(fit["x"], fit["y"], s=5, color=OWN_COLOUR, alpha=0.08, linewidths=0)
    fit_axis.plot(line, line, color=INK, lw=1.0, ls=(0, (4, 3)), label="y = x")
    fit_axis.plot(line, PAPER_SLOPE * line + PAPER_INTERCEPT, color=PAPER_COLOUR, lw=5.0, alpha=0.7,
                  label=f"Paper: {PAPER_SLOPE:.2f}x + {PAPER_INTERCEPT:.2f}, r = {PAPER_CORRELATION:.2f}")
    fit_axis.plot(line, fit["slope"] * line + fit["intercept"], color=OWN_COLOUR, lw=1.6,
                  label=f"This work: {fit['slope']:.2f}x + {fit['intercept']:.2f}, r = {fit['corr']:.2f}")
    fit_axis.set_xlim(0, top)
    fit_axis.set_ylim(0, top)
    fit_axis.set_aspect("equal")
    fit_axis.set_xlabel("Measured arterial curve [SUV]")
    fit_axis.set_ylabel("Predicted (ensemble) [SUV]")
    fit_axis.legend(loc="upper left", fontsize=6.8)
    fit_axis.set_title(f"B  Point by point, {fit['n_points']} points\n    (70 scans \u00d7 42 frames)", loc="left")
    figure.suptitle("FC-DLIF on uncorrected images against Salomonsen et al. (2026)", fontsize=9)
    figure.text(0.01, 0.0, f"Blue: {n_models - n_failed} converged networks of {n_models} (10 folds \u00d7 10 runs). "
                           "Published values are read off the paper's figures.", fontsize=6.8, color=INK)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    for extension in ("pdf", "png"):
        figure.savefig(f"{output_stem}.{extension}")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="FC-DLIF against Salomonsen et al. (2026).")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_study_config(args.config)
    work_folder = config["paths"]["work_folder"]

    frame = pd.read_csv(work_folder / "predictions" / f"{NETWORK}.csv")
    frame = frame[frame["condition"] == CONDITION].copy()
    frame["squared_error"] = (frame["predicted"] - frame["measured"]) ** 2

    losses = []
    for summary_path in sorted((work_folder / "models" / NETWORK / WEIGHTS).glob("fold_*/run_*/summary.json")):
        with open(summary_path, "r", encoding="utf-8") as summary_file:
            losses.append({"fold": int(summary_path.parent.parent.name.split("_")[1]),
                           "run": int(summary_path.parent.name.split("_")[1]),
                           "best_val_loss": json.load(summary_file)["best_val_loss"]})
    losses = pd.DataFrame(losses)
    best_runs = losses.sort_values(["fold", "best_val_loss", "run"]).groupby("fold", as_index=False).first()[["fold", "run"]]

    converged = frame[frame["converged"]]
    n_models, n_failed = len(losses), int((~frame.drop_duplicates(["fold", "run"])["converged"]).sum())
    boxes, fits = summarise(converged, best_runs)
    boxes_all, fits_all = summarise(frame, best_runs)

    rows = [{"quantity": "MSE", "source": "paper Fig. 2a (read off)", "runs": "", **PAPER_BOX}]
    for label, group in (("converged", boxes), ("all", boxes_all)):
        rows += [{"quantity": "MSE", "source": name, "runs": label, **values} for name, values in group.items()]
    rows.append({"quantity": "point-wise fit", "source": "paper Fig. 3 (legend)", "runs": "",
                 "slope": PAPER_SLOPE, "intercept": PAPER_INTERCEPT, "corr": PAPER_CORRELATION})
    for label, group in (("converged", fits), ("all", fits_all)):
        rows += [{"quantity": "point-wise fit", "source": name, "runs": label,
                  **{key: value for key, value in fit.items() if key not in ("x", "y")}} for name, fit in group.items()]
    table = pd.DataFrame(rows)

    output_folder = work_folder / "analyses" / NETWORK
    output_folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_folder / "reproduce_paper_2026.csv", index=False)
    figures_folder = work_folder / "figures"
    figures_folder.mkdir(parents=True, exist_ok=True)
    draw(boxes, fits, n_models, n_failed, figures_folder / "reproduce_paper_2026")
    print(table.round(3).fillna("").to_string(index=False))


if __name__ == "__main__":
    main()
