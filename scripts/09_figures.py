"""Stage 09: the figures of the thesis, drawn from the results of the earlier stages.

Every figure reads only files the pipeline wrote (and the group's measured
curves). Nothing is computed here that a later reader would need to trust
without seeing it: medians and confidence intervals come from the comparison
tables of stages 07 and 08, not from the plotting code.

One plot type is used for every paired comparison (the forest plot), so a
reader learns to read it once. Colours carry one meaning each throughout:
  blue  = FC-DLIF (DLIFNet_MAX), orange = DLIF2024 (CNN3_1dConv3_linear);
  green = Richardson-Lucy, pink = reblurred Van Cittert, grey = uncorrected;
  yellow = forward shift, violet = reverse shift (mismatch figure only).
The colour pairs were checked for colour-blind separation.

Figures (in <work_folder>/figures/, as .pdf and .png), in the order of the thesis argument
(step 1, reproduce_paper_2026, is drawn into the same folder by analyses/reproduce_paper_2026.py):
  2  example_input_function     measured arterial curve of the example scan, loss-weight groups
  3  example_pvc_image          one frame of the example scan, uncorrected, RL and RVC
  3  diagnostics_by_frame       peak ratio and noise ratio of the corrections, per frame
  4  results_retrained          retrained on corrected input: RL, RVC and the pre-registered RL k = 3,
                                with the change per scan for RL and RVC
  5  mismatch_overview          cohort-mean input function, matched and mismatched (RL), early and late
  5  mismatch_anatomy_<rl|rvc|rl_k3>_<network>  matched and mismatched input: cohort curves, area by part,
                                bias^2 and variance with the scale factor, single run and best of 10
  6  scaling                    the forward shift against k and against the PSF width
  7  results_motion             motion correction
  8  convergence_val_loss       when each training run leaves the loss plateau (escape epoch)
  9  effects_ki                 every paired comparison of the Patlak Ki error
     effects_rmse               every paired comparison of the curve RMSE (appendix)

With --overview, the same plot types are also drawn for everything the study
compares, in <work_folder>/figures/overview/, so no condition is left out:
  effects_<metric>             forest plot of every tested curve metric
  effects_ki_absolute, _signed forest plots of the Patlak Ki error
  paired_difference_<family>   change in RMSE per scan, every condition of the family
  paired_rmse_<family>         RMSE of reference and condition per scan, joined
  agreement_<family>           peak ratio and AUC ratio against 1
  error_by_time_window_<family> RMSE in the bolus (0-2.5 min), 2.5-5.5 min and the tail
  cohort_curves_<family>       aligned bolus peak and area by part of the curve
  example_curves_<scan>_<family> the example scan, and the scans with the lowest,
                               median and highest baseline RMSE
  bias_variance_<family>       squared error split into bias^2 and variance
  mismatch_curves_rvc          the thesis mismatch figure for the RVC arm
  sweep_iterations, sweep_psf  the forward shift against k and against the PSF width
  diagnostics_<correction>     peak and noise ratio per frame, every correction
In the overview, colour identifies a condition within one figure (with a
legend); the reference is grey. The thesis figures and the overview are drawn
by the same functions.

A figure whose inputs do not exist yet is skipped with a message. Run after
stage 08 and analyses/bias_variance.py.

    python scripts/09_figures.py --config configs/study.yaml
    python scripts/09_figures.py --config configs/study.yaml --overview
"""

import argparse
import json
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

from pvc_dlif_pipeline.config import load_study_config
from pvc_dlif_pipeline.group_data import load_arterial_curve
from pvc_dlif_pipeline.nifti_io import read_series
from pvc_dlif_pipeline.plot_style import TEXT_WIDTH_IN, apply_style
from pvc_dlif_pipeline.pvc import correction_folder_name

# ---------------------------------------------------------------------------
# Style: the only settings that differ from matplotlib's defaults.
# ---------------------------------------------------------------------------
apply_style()

NETWORK_COLOURS = {"fcdlif": "#2a78d6", "dlif2024": "#eb6834"}
NETWORK_NAMES = {"fcdlif": "FC-DLIF", "dlif2024": "DLIF2024"}
CORRECTION_COLOURS = {"uncorrected": "#6b6b6b", "rl_i15": "#008300", "rvc_i15": "#d55181"}
CORRECTION_NAMES = {"uncorrected": "Uncorrected", "rl_i15": "RL, k = 15", "rvc_i15": "RVC, k = 15"}
FORWARD_COLOUR, REVERSE_COLOUR = "#c98500", "#6250d6"
NEUTRAL = "#52514e"
INK_COLOUR = "#333333"

# Names of the input representations (stage 04 folder names).
REPRESENTATION_NAMES = {
    "uncorrected": "uncorrected", "rl_i15": "RL", "rvc_i15": "RVC", "rl_i3": "RL k=3",
    "rl_i10": "RL k=10", "rl_i20": "RL k=20", "rvc_i10": "RVC k=10", "rvc_i20": "RVC k=20",
    "rl_i15_psf0p90": "RL PSF\u00d70.9", "rl_i15_psf1p10": "RL PSF\u00d71.1",
    "rvc_i15_psf0p90": "RVC PSF\u00d70.9", "rvc_i15_psf1p10": "RVC PSF\u00d71.1",
    "mc_only": "MC", "mc_then_rl_i15": "MC then RL", "rl_i15_then_mc": "RL then MC"}
# Filled in main() from the conditions in the config: name -> (weights, shown).
CONDITIONS = {}

FAMILY_TITLES = {
    "retrained": "Retrained on corrected input  (vs baseline)",
    "motion": "Retrained on motion-corrected input  (vs baseline)",
    "forward_shift": "Forward shift: uncorrected weights, corrected input  (vs baseline)",
    "rl_reverse_shift": "Reverse shift: RL weights, uncorrected input  (vs RL)",
    "rvc_reverse_shift": "Reverse shift: RVC weights, uncorrected input  (vs RVC)",
    "sensitivity": "Forward shift, other iteration counts and PSF widths  (vs baseline)",
    "rl_k3": "Pre-registered: retrained on RL, k = 3  (vs baseline)",
    "rl_k3_forward_shift": "Pre-registered: forward shift, RL, k = 3  (vs baseline)",
    "rl_k3_reverse_shift": "Pre-registered: reverse shift, RL, k = 3  (vs RL, k = 3)",
}


def save(figure, figures_folder, name):
    """Write a figure as .pdf (for the thesis) and .png (for a quick look)."""
    for extension in ("pdf", "png"):
        figure.savefig(figures_folder / f"{name}.{extension}")
    plt.close(figure)
    print(f"Written: {name}")


def shade_frame_groups(axes, frame_bounds, frame_times, weights):
    """Shade the frame groups that the training loss weights separately; returns legend patches."""
    fills = ["#d9d6cc", "#bcd2ef", "#f3c3ac"]
    patches = []
    for index in range(len(frame_bounds) - 1):
        first, last = frame_bounds[index], frame_bounds[index + 1] - 1
        # Each group runs from halfway to the previous frame to halfway to the next one.
        start = 0.0 if first == 0 else (frame_times[first - 1] + frame_times[first]) / 2
        stop = (frame_times[last] + frame_times[last + 1]) / 2 if last + 1 < len(frame_times) else frame_times[last]
        for ax in axes:
            ax.axvspan(start, stop, color=fills[index], lw=0, zorder=0)
        patches.append(matplotlib.patches.Patch(color=fills[index],
                                                label=f"frames {first + 1}–{last + 1}: loss weight {weights[index]:g}"))
    return patches


# ---------------------------------------------------------------------------
# Data and method figures.
# ---------------------------------------------------------------------------
def figure_example_input_function(config, figures_folder):
    scan_id = config["figures"]["example_scan"]
    activity, time_min = load_arterial_curve(config["paths"]["group_data_folder"], scan_id)
    bounds = config["training"]["loss_frame_bounds"]
    weights = config["training"]["loss_weights"]

    figure, (bolus, whole) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.4))
    early = time_min <= 3.0
    bolus.plot(time_min[early], activity[early], "-o", color="black", lw=1.2, ms=3)
    bolus.set_ylabel("Activity [SUV]")
    bolus.set_title("Bolus, linear scale")
    whole.plot(time_min, activity, "-o", color="black", lw=1.2, ms=3)
    whole.set_yscale("log")
    whole.set_title("Whole scan, logarithmic scale")
    bolus.set_xlim(0, 3.0)
    patches = shade_frame_groups([bolus, whole], bounds, time_min, weights)
    figure.legend(handles=patches, loc="lower center", ncol=3)
    figure.suptitle(f"Measured arterial input function, scan {scan_id}, one marker per frame", fontsize=9)
    figure.tight_layout(rect=(0, 0.2, 1, 1))
    figure.supxlabel("Time [min]", fontsize=8, y=0.17)
    save(figure, figures_folder, "example_input_function")


def full_width_at_half_maximum(profile, spacing_mm):
    """FWHM of a single peak, with linear interpolation between samples, in mm."""
    peak = int(np.argmax(profile))
    half = profile[peak] / 2.0
    left = peak
    while left > 0 and profile[left - 1] > half:
        left -= 1
    right = peak
    while right < len(profile) - 1 and profile[right + 1] > half:
        right += 1
    if left == 0 or right == len(profile) - 1:
        return float("nan")
    left_crossing = (left - 1) + (half - profile[left - 1]) / (profile[left] - profile[left - 1])
    right_crossing = right + (profile[right] - half) / (profile[right] - profile[right + 1])
    return float((right_crossing - left_crossing) * spacing_mm)


def figure_example_pvc_image(config, figures_folder):
    """One frame of the example scan, uncorrected and after RL and RVC, with a line profile.

    The frame and the voxel are found on a lightly smoothed copy (3 x 3 x 3 mean) of the
    uncorrected series inside the network's crop window, so that a single noisy voxel is
    not chosen: the frame, among the first 20, whose smoothed maximum is highest, and the
    position of that maximum. The profile runs along the row through it; its maximum is
    given relative to the maximum of the uncorrected slice, and its FWHM by linear
    interpolation. This is the selection of the original figure.
    """
    scan_id = config["figures"]["example_scan"]
    work_folder = config["paths"]["work_folder"]
    paths = {"uncorrected": work_folder / "native" / f"{scan_id}.nii.gz",
             "rl_i15": work_folder / "pvc" / "rl_i15" / f"{scan_id}.nii.gz",
             "rvc_i15": work_folder / "pvc" / "rvc_i15" / f"{scan_id}.nii.gz"}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        print(f"example_pvc_image skipped, missing: {missing}")
        return
    _, time_min = load_arterial_curve(config["paths"]["group_data_folder"], scan_id)

    # Every series cut to the part of the crop window that lies inside the volume.
    windows = pd.read_csv(work_folder / "crop_windows.csv").set_index("scan_id")
    size_z, size_y, size_x = config["scans"]["network_input_shape"]
    start_z, start_y, start_x = (max(int(windows.loc[scan_id, column]), 0)
                                 for column in ["start_z", "start_y", "start_x"])
    series_in_window = {}
    for name, path in paths.items():
        series, voxel_mm_xyz = read_series(path)
        series_in_window[name] = series[:, start_z:start_z + size_z, start_y:start_y + size_y,
                                        start_x:start_x + size_x].astype(np.float64)

    smoothed = np.stack([uniform_filter(volume, size=3) for volume in series_in_window["uncorrected"][:20]])
    frame = int(smoothed.reshape(len(smoothed), -1).max(axis=1).argmax())
    z, y, x = np.unravel_index(int(smoothed[frame].argmax()), smoothed[frame].shape)
    frames = {name: series[frame] for name, series in series_in_window.items()}
    scale = float(frames["uncorrected"][z].max())  # maximum of the uncorrected slice

    voxel_mm = float(voxel_mm_xyz[0])
    half_field = int(round(config["figures"]["example_field_mm"] / 2 / voxel_mm))
    rows = slice(max(y - half_field, 0), y + half_field + 1)
    columns = slice(max(x - half_field, 0), x + half_field + 1)
    top = max(float(volume[z, rows, columns].max()) for volume in frames.values()) / scale
    extent = (columns.start * voxel_mm, (columns.start + frames["uncorrected"][z, rows, columns].shape[1]) * voxel_mm,
              (rows.start + frames["uncorrected"][z, rows, columns].shape[0]) * voxel_mm, rows.start * voxel_mm)
    position_mm = (np.arange(frames["uncorrected"].shape[2]) + 0.5) * voxel_mm

    figure = plt.figure(figsize=(TEXT_WIDTH_IN, 4.4))
    grid = figure.add_gridspec(2, 3, height_ratios=[1.5, 1])
    image_axes = [figure.add_subplot(grid[0, 0])]
    image_axes += [figure.add_subplot(grid[0, column], sharex=image_axes[0], sharey=image_axes[0]) for column in (1, 2)]
    profile_axis = figure.add_subplot(grid[1, :])
    for column, (name, volume) in enumerate(frames.items()):
        image_axis = image_axes[column]
        image_axis.imshow(volume[z, rows, columns] / scale, cmap="gray_r", vmin=0, vmax=top, extent=extent,
                          interpolation="nearest")
        image_axis.axhline((y + 0.5) * voxel_mm, color=CORRECTION_COLOURS[name], ls="--", lw=1.8)
        image_axis.set_title(CORRECTION_NAMES[name])
        image_axis.grid(False)
        if column == 1:
            image_axis.set_xlabel("mm")
        if column > 0:
            plt.setp(image_axis.get_yticklabels(), visible=False)
        profile = volume[z, y] / scale
        fwhm = full_width_at_half_maximum(profile, voxel_mm)
        print(f"  {name}: maximum {profile.max():.2f} x uncorrected, FWHM {fwhm:.2f} mm")
        profile_axis.plot(position_mm[columns], profile[columns], "-o", color=CORRECTION_COLOURS[name], lw=1.4,
                          ms=2.5, label=f"{CORRECTION_NAMES[name]}: maximum {profile.max():.2f}\u00d7, FWHM {fwhm:.2f} mm")
    image_axes[0].set_ylabel("mm")
    profile_axis.set_xlabel("Position along the dashed row [mm]")
    profile_axis.set_ylabel("Activity / uncorrected maximum")
    profile_axis.legend(loc="upper left", fontsize=7)
    figure.suptitle(f"Scan {scan_id}, frame {frame + 1} ({time_min[frame]:.2f} min), slice {z} of the crop window",
                    fontsize=9)
    figure.tight_layout()
    save(figure, figures_folder, "example_pvc_image")


def figure_convergence(config, figures_folder):
    """Step 8: the epoch at which each run leaves the initial loss plateau (validation loss below the
    convergence threshold), from analyses/training_convergence.py. One dot per run; runs that never leave it
    are drawn in a separate column and are excluded. On the right: the median paired change in the escape
    epoch against the uncorrected input, on matched (fold, run), and its Wilcoxon p-value."""
    work_folder = config["paths"]["work_folder"]
    threshold = config["evaluation"]["converged_val_loss"]
    epochs = config["training"]["epochs"]
    representations = config["training"]["trained_representations"]
    networks = [network for network in config["networks"]
                if (work_folder / "analyses" / network / "training_convergence_runs.csv").exists()]
    if not networks:
        print("convergence_val_loss skipped, run analyses/training_convergence.py first")
        return
    never = 4.0 * epochs  # where the runs that never left the plateau are drawn
    figure, axes = plt.subplots(1, len(networks), figsize=(TEXT_WIDTH_IN, 0.38 * len(representations) + 1.5),
                                sharey=True, squeeze=False)
    random = np.random.default_rng(0)  # vertical scatter only, so the dots do not hide each other
    for ax, network in zip(axes[0], networks):
        runs = pd.read_csv(work_folder / "analyses" / network / "training_convergence_runs.csv")
        comparisons = pd.read_csv(work_folder / "analyses" / network / "training_convergence_comparisons.csv")
        comparisons = comparisons.set_index("representation")
        for row, representation in enumerate(representations):
            group = runs[runs["representation"] == representation]
            if group.empty:
                continue
            escaped = group["escaped"].to_numpy()
            x = np.where(escaped, group["escape_epoch"].to_numpy(), never)
            y = row + random.uniform(-0.25, 0.25, len(group))
            ax.plot(x[escaped], y[escaped], "o", color=NETWORK_COLOURS[network], ms=2.4, alpha=0.5)
            ax.plot(x[~escaped], y[~escaped], "o", color="white", ms=4, markeredgecolor="black", markeredgewidth=0.9)
            ax.plot(np.median(group.loc[group["escaped"], "escape_epoch"]), row, "|", color="black", ms=12, mew=2)
            if representation in comparisons.index:
                result = comparisons.loc[representation]
                text = f"{result['median_difference_epochs']:+.1f}, p = {result['p_wilcoxon']:.1g}"
            else:
                text = "reference"
            if (~escaped).any():
                text += f"; {int((~escaped).sum())} excluded"
            ax.annotate(text, xy=(1.0, row), xycoords=("axes fraction", "data"), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=6.5, color=NEUTRAL)
        ax.set_xscale("log")
        ax.set_xlim(5, never * 1.8)
        ax.set_xticks([10, 100, 1000, never])
        ax.set_xticklabels(["10", "100", "1000", "never"])
        ax.axvline(epochs * 2.0, color=NEUTRAL, lw=0.6, ls=":")
        ax.set_xlabel(f"Epoch at which validation loss first falls below {threshold:g}")
        ax.set_title(NETWORK_NAMES[network])
        ax.grid(axis="y", visible=False)
    axes[0, 0].set_yticks(range(len(representations)))
    axes[0, 0].set_yticklabels([REPRESENTATION_NAMES.get(name, name)[0].upper() + REPRESENTATION_NAMES.get(name, name)[1:]
                                 for name in representations])
    axes[0, 0].invert_yaxis()
    figure.suptitle("When each run leaves the loss plateau: one dot per run, black bar the median\n"
                    "right: median paired change in epochs vs uncorrected (same fold and run), Wilcoxon p",
                    fontsize=9)
    figure.tight_layout()
    save(figure, figures_folder, "convergence_val_loss")


# ---------------------------------------------------------------------------
# Results.
# ---------------------------------------------------------------------------
def read_results(config, file_name):
    """{(network, estimand): table} for every result file that exists."""
    tables = {}
    for network in config["networks"]:
        for estimand in ["single_run", "ensemble"]:
            path = config["paths"]["work_folder"] / "results" / network / estimand / file_name
            if path.exists():
                tables[(network, estimand)] = pd.read_csv(path)
    return tables



def forest_rows(config, tables, selection=None):
    """Rows of a forest plot, family by family: ('family', name) or ('condition', family, condition, reference).

    selection: optional {family name: list of conditions, or None for all}; only those are drawn, in config order.
    """
    predicted = set()
    for table in tables.values():
        predicted.update(zip(table["family"], table["condition"]))
    rows = []
    for family in config["evaluation"]["comparison_families"]:
        if selection is not None and family["family"] not in selection:
            continue
        wanted = family["conditions"] if selection is None or selection[family["family"]] is None \
            else selection[family["family"]]
        members = [condition for condition in wanted if (family["family"], condition) in predicted]
        if members:
            if rows:
                rows.append(("space",))
            rows.append(("family", family["family"]))
            rows += [("condition", family["family"], condition, family["reference"]) for condition in members]
    return rows


def draw_forest(ax, rows, tables, metric, networks):
    """Median difference with 95 % CI (single run, filled) and ensemble median (open), per network."""
    offsets = {network: offset for network, offset in zip(networks, np.linspace(-0.17, 0.17, len(networks)))}
    for position, row in enumerate(rows):
        if row[0] != "condition":
            continue
        _, family, condition, reference = row
        for network in networks:
            for estimand in ["single_run", "ensemble"]:
                table = tables.get((network, estimand))
                if table is None:
                    continue
                match = table[(table["family"] == family) & (table["condition"] == condition)
                              & (table["reference"] == reference) & (table["metric"] == metric)]
                if match.empty:
                    continue
                result = match.iloc[0]
                y = position + offsets[network]
                if estimand == "single_run":
                    ax.hlines(y, result["ci_low"], result["ci_high"], color=NETWORK_COLOURS[network], lw=1.4)
                    ax.plot(result["median_difference"], y, "o", color=NETWORK_COLOURS[network], ms=4.5,
                            markeredgecolor="white", markeredgewidth=0.6)
                else:
                    ax.plot(result["median_difference"], y, "D", color="white", ms=3.8,
                            markeredgecolor=NETWORK_COLOURS[network], markeredgewidth=1.0)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_ylim(len(rows) - 0.5, -0.8)
    ax.grid(axis="y", visible=False)


def label_forest_rows(ax, rows):
    """Condition names as tick labels, family titles as bold text across the axis."""
    positions = [position for position, row in enumerate(rows) if row[0] == "condition"]
    ax.set_yticks(positions)
    ax.set_yticklabels([condition_label(rows[position][2]) for position in positions])
    for position, row in enumerate(rows):
        if row[0] == "family":
            ax.annotate(FAMILY_TITLES.get(row[1], row[1]), xy=(0.01, position), xycoords=("figure fraction", "data"),
                        va="center", ha="left", fontsize=7.5, fontweight="bold")


def forest_legend(figure, networks):
    handles = []
    for network in networks:
        handles.append(plt.Line2D([], [], color=NETWORK_COLOURS[network], marker="o", lw=1.4, ms=4.5,
                                  label=f"{NETWORK_NAMES[network]}: single run, median and 95 % CI"))
        handles.append(plt.Line2D([], [], color="white", marker="D", lw=0, ms=4, markeredgecolor=NETWORK_COLOURS[network],
                                  label=f"{NETWORK_NAMES[network]}: ensemble, median"))
    figure.legend(handles=handles, loc="lower center", ncol=2,
                  title="Rows: fitted on \u2192 shown (one name: fitted on and shown the same)", title_fontsize=7.5)



# ---------------------------------------------------------------------------
# General drawing functions. The thesis figures and the overview call the same
# functions, with different conditions, so the two cannot drift apart.
# ---------------------------------------------------------------------------
# Condition colours in the overview figures, where colour identifies a
# condition within one figure (always with a legend): the eight validated
# categorical slots in their fixed order; the reference is grey.
CONDITION_SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
REFERENCE_COLOUR = "#6b6b6b"
METRIC_LABELS = {"rmse": "RMSE [SUV]", "nrmse": "NRMSE", "mae": "MAE [SUV]", "r2": "R²",
                 "auc_ratio": "AUC ratio", "ccc": "CCC", "pearson_r": "Pearson r"}


def condition_label(condition):
    """What was run, read from the config: '<fitted on> \u2192 <shown>', or one name when they are the same."""
    weights, shown = CONDITIONS[condition]
    if weights == shown:
        return REPRESENTATION_NAMES.get(weights, weights)
    return f"{REPRESENTATION_NAMES.get(weights, weights)} \u2192 {REPRESENTATION_NAMES.get(shown, shown)}"


def draw_diagnostics(config, figures_folder, corrections, colours, name):
    """Peak ratio and noise ratio per frame for the given stage-02 corrections, median and quartiles over scans."""
    work_folder = config["paths"]["work_folder"]
    tables = []
    for correction in corrections:
        for path in sorted((work_folder / "pvc" / correction).glob("*.frames.csv")):
            table = pd.read_csv(path)
            table["correction"] = correction
            tables.append(table)
    if not tables:
        print(f"{name} skipped, no stage-02 diagnostics for {corrections}")
        return
    table = pd.concat(tables, ignore_index=True)
    table["peak_ratio"] = table["blood_peak_after"] / table["blood_peak_before"]
    table["noise_ratio"] = table["noise_pct_std_after"] / table["noise_pct_std_before"]

    frames = np.sort(table["frame"].unique())
    numbers = frames + 1
    counts = table.groupby("frame")["counts_proxy"].median().reindex(frames).to_numpy()
    before_arrival = counts < 0.1 * np.median(counts)
    durations = table.groupby("frame")["duration_s"].median().reindex(frames).round().to_numpy()

    figure, (peak_axis, noise_axis) = plt.subplots(2, 1, figsize=(TEXT_WIDTH_IN, 4.2), sharex=True)
    fills = ["#f3f3f1", "#e9f0fa", "#fbeee8", "#eef6ee"]
    start, shade = 0, 0
    for index in range(1, len(durations) + 1):
        if index == len(durations) or durations[index] != durations[start]:
            if index - start >= 3:
                for ax in (peak_axis, noise_axis):
                    ax.axvspan(numbers[start] - 0.5, numbers[index - 1] + 0.5, color=fills[shade % 4], lw=0, zorder=0)
                peak_axis.annotate(f"{durations[start]:g} s frames", xy=((numbers[start] + numbers[index - 1]) / 2, 1.0),
                                   xycoords=("data", "axes fraction"), xytext=(0, -3), textcoords="offset points",
                                   ha="center", va="top", fontsize=6.5, color=NEUTRAL)
                shade += 1
            start = index

    offsets = [0.0] * len(corrections)  # bands, so no horizontal offset is needed
    for offset, correction, colour in zip(offsets, corrections, colours):
        rows = table[table["correction"] == correction]
        label = CORRECTION_NAMES.get(correction, correction)
        for ax, column in [(peak_axis, "peak_ratio"), (noise_axis, "noise_ratio")]:
            quartiles = rows.groupby("frame")[column].quantile([0.25, 0.5, 0.75]).unstack().reindex(frames)
            x = numbers + offset
            ax.fill_between(x, quartiles[0.25], quartiles[0.75], color=colour, alpha=0.2, lw=0)
            ax.plot(x, quartiles[0.5], "-", color=colour, lw=1.1)
            ax.plot(x[~before_arrival], quartiles[0.5].to_numpy()[~before_arrival], "o", color=colour, ms=3, label=label)
            ax.plot(x[before_arrival], quartiles[0.5].to_numpy()[before_arrival], "o", color="white", ms=3,
                    markeredgecolor=colour, markeredgewidth=0.8)
    for ax in (peak_axis, noise_axis):
        ax.axhline(1.0, color=NEUTRAL, lw=0.8, ls="--")
        ax.grid(axis="x", visible=False)
    peak_axis.set_ylabel("Peak in the blood region\n(corrected / uncorrected)")
    noise_axis.set_ylabel("Background noise\n(corrected / uncorrected)")
    noise_axis.set_xlabel("Frame number   (hollow markers: frames before the tracer arrives)")
    noise_axis.set_xlim(numbers[0] - 0.5, numbers[-1] + 0.5)
    handles = [(matplotlib.patches.Patch(color=colour, alpha=0.2, lw=0),
                plt.Line2D([], [], color=colour, marker="o", ms=3, lw=1.1))
               for colour in colours]
    figure.legend(handles, [CORRECTION_NAMES.get(correction, correction) for correction in corrections],
                  loc="lower center", ncol=len(corrections))
    n_scans = int(table.groupby("correction").size().min() // len(frames))
    figure.suptitle(f"Correction diagnostics per frame: median (line) and quartiles (band) over {n_scans} scans",
                    fontsize=9)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    save(figure, figures_folder, name)


def draw_paired_axis(ax, per_scan, table, condition, reference, network):
    """One panel: change in RMSE per scan (grey stems, sorted), median and its 95 % CI from stage 07."""
    difference = (per_scan.loc[condition] - per_scan.loc[reference]).dropna().sort_values()
    positions = np.arange(1, difference.size + 1)
    ax.vlines(positions, 0, difference.to_numpy(), color="#9a9993", lw=1.2)
    summary = table[(table["metric"] == "rmse") & (table["condition"] == condition)
                    & (table["reference"] == reference)].iloc[0]
    ax.axhspan(summary["ci_low"], summary["ci_high"], color=NETWORK_COLOURS[network], alpha=0.18, lw=0)
    ax.axhline(summary["median_difference"], color=NETWORK_COLOURS[network], lw=1.4,
               label=f"median {summary['median_difference']:+.3f} [{summary['ci_low']:+.3f}, {summary['ci_high']:+.3f}]")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_title(f"{NETWORK_NAMES[network]}: {condition_label(condition)}")
    ax.set_xlim(0, difference.size + 1)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(axis="x", visible=False)


def draw_paired_difference(config, figures_folder, family, name):
    """Change in RMSE per scan, every condition of a family against its reference; rows conditions, columns networks."""
    metrics = read_results(config, "curve_metrics.csv")
    comparisons = read_results(config, "comparisons.csv")
    networks = [network for network in config["networks"] if (network, "single_run") in metrics]
    if not networks:
        print(f"{name} skipped, no stage-07 results")
        return
    reference = family["reference"]
    conditions = [condition for condition in family["conditions"]
                  if all(condition in set(metrics[(network, "single_run")]["condition"]) for network in networks)]
    if not conditions:
        print(f"{name} skipped, conditions not predicted")
        return
    figure, axes = plt.subplots(len(conditions), len(networks), squeeze=False, sharey=True,
                                figsize=(TEXT_WIDTH_IN, 1.75 * len(conditions) + 0.5))
    for column, network in enumerate(networks):
        per_scan = metrics[(network, "single_run")].groupby(["condition", "scan_id"])["rmse"].mean()
        table = comparisons[(network, "single_run")]
        for row, condition in enumerate(conditions):
            ax = axes[row, column]
            draw_paired_axis(ax, per_scan, table, condition, reference, network)
            if column == 0:
                ax.set_ylabel("ΔRMSE [SUV]\n(below 0: lower error)")
            if row == len(conditions) - 1:
                ax.set_xlabel("Scans, sorted by the change")
    figure.suptitle(f"Change in RMSE per scan against {condition_label(reference)}\n"
                    "mean over converged runs; band: 95 % CI of the median", fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 1 - 0.25 / figure.get_figheight()))
    save(figure, figures_folder, name)


@lru_cache(maxsize=None)
def read_converged_predictions(work_folder, network):
    """The stage-06 predictions of the converged runs (read once per network)."""
    predictions = pd.read_csv(work_folder / "predictions" / f"{network}.csv")
    return predictions[predictions["converged"]]


def run_mean_curves(work_folder, network, conditions):
    """One curve per condition and scan: the mean over the scan's converged runs."""
    predictions = read_converged_predictions(work_folder, network)
    predictions = predictions[predictions["condition"].isin(conditions)]
    return (predictions.groupby(["condition", "scan_id", "frame"], as_index=False)
            .agg(time_min=("time_min", "first"), predicted=("predicted", "mean"), measured=("measured", "first")))


COHORT_WINDOWS = [(0.0, 2.5), (2.5, 5.5), (5.5, 40.5)]


def draw_cohort_axes(peak_axis, area_axis, curves, series):
    """Left: the mean over scans after aligning every scan on its own measured peak and dividing by its height.
    Right: predicted area over measured area in three parts of the curve, median and quartiles over scans.
    curves: run-mean curves (run_mean_curves); series: [(condition, label, colour, line style)], the first
    sets the measured peak. Returns the number of scans."""
    names = [condition for condition, _, _, _ in series]
    grid = np.round(np.arange(-0.5, 1.5 + 1e-9, 1.0 / 12.0), 4)
    step = min(0.17, 0.8 / len(series))
    curves = curves.copy()
    peaks = curves[curves["condition"] == names[0]]
    peaks = peaks.loc[peaks.groupby("scan_id")["measured"].idxmax()].set_index("scan_id")
    curves["time_from_peak"] = curves["time_min"] - curves["scan_id"].map(peaks["time_min"])
    curves["predicted_scaled"] = curves["predicted"] / curves["scan_id"].map(peaks["measured"])
    curves["measured_scaled"] = curves["measured"] / curves["scan_id"].map(peaks["measured"])

    def cohort_mean(condition, column):
        on_grid = [np.interp(grid, scan["time_from_peak"], scan[column], left=np.nan, right=np.nan)
                   for _, scan in curves[curves["condition"] == condition].sort_values("time_from_peak").groupby("scan_id")]
        return np.nanmean(on_grid, axis=0), len(on_grid)

    measured, n_scans = cohort_mean(names[0], "measured_scaled")
    peak_axis.plot(grid, measured, "-", color="black", lw=2.4, label="Measured arterial curve")
    for index, (condition, label, colour, style) in enumerate(series):
        if condition not in set(curves["condition"]):
            continue
        mean, _ = cohort_mean(condition, "predicted_scaled")
        peak_axis.plot(grid, mean, style, color=colour, lw=1.5, label=label)
        ratios_by_window = []
        for start, stop in COHORT_WINDOWS:
            part = curves[(curves["condition"] == condition) & (curves["time_min"] >= start) & (curves["time_min"] <= stop)]
            ratios = [np.trapezoid(scan["predicted"], scan["time_min"]) / np.trapezoid(scan["measured"], scan["time_min"])
                      for _, scan in part.sort_values("time_min").groupby("scan_id")]
            ratios_by_window.append(np.percentile(ratios, [25, 50, 75]))
        ratios_by_window = np.asarray(ratios_by_window)
        x = np.arange(len(COHORT_WINDOWS)) + (index - (len(series) - 1) / 2) * step
        area_axis.vlines(x, ratios_by_window[:, 0], ratios_by_window[:, 2], color=colour, lw=2.0, alpha=0.8,
                         linestyles="dotted" if style == ":" else "solid")
        area_axis.plot(x, ratios_by_window[:, 1], "o", color=colour, ms=4.5, markeredgecolor="white")
    peak_axis.set_xlabel("Time from the measured peak [min]")
    area_axis.axhline(1.0, color=NEUTRAL, lw=0.8, ls="--")
    area_axis.set_xticks(range(len(COHORT_WINDOWS)))
    area_axis.set_xticklabels([f"{start:g}\u2013{stop:g}" for start, stop in COHORT_WINDOWS])
    area_axis.set_xlabel("Part of the curve [min]")
    area_axis.set_ylabel("Predicted area / measured area\n(median and quartiles)")
    area_axis.grid(axis="x", visible=False)
    return n_scans


def draw_cohort_curves(config, figures_folder, series, name, title):
    """Cohort curves (draw_cohort_axes) for every network, one row each."""
    work_folder = config["paths"]["work_folder"]
    networks = [network for network in config["networks"] if (work_folder / "predictions" / f"{network}.csv").exists()]
    if not networks:
        print(f"{name} skipped, no predictions")
        return
    names = [condition for condition, _, _, _ in series]
    figure, axes = plt.subplots(len(networks), 2, figsize=(TEXT_WIDTH_IN, 2.5 * len(networks) + 0.4 + 0.12 * len(series)),
                                squeeze=False)
    for row, network in enumerate(networks):
        peak_axis, area_axis = axes[row]
        n_scans = draw_cohort_axes(peak_axis, area_axis, run_mean_curves(work_folder, network, names), series)
        peak_axis.set_ylabel(f"{NETWORK_NAMES[network]}\nActivity / measured peak\n(mean over {n_scans} scans)")
        if row < len(networks) - 1:
            peak_axis.set_xlabel("")
        if row == 0:
            peak_axis.set_title("Bolus peak, every scan aligned on its measured peak")
            area_axis.set_title("Area by part of the curve, one value per scan")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2)
    figure.suptitle(title, fontsize=9)
    legend_inches = 0.17 * ((len(labels) + 1) // 2) + 0.1
    figure.tight_layout(rect=(0, legend_inches / figure.get_figheight(), 1, 1))
    save(figure, figures_folder, name)


def family_series(family):
    """Reference in grey, then the family's conditions in the categorical slots."""
    series = [(family["reference"], f"{condition_label(family['reference'])} (reference)", REFERENCE_COLOUR, "-")]
    for index, condition in enumerate(family["conditions"]):
        series.append((condition, condition_label(condition), CONDITION_SLOTS[index % len(CONDITION_SLOTS)], "-"))
    return series


def draw_example_curves(config, figures_folder, family, scan_id, name):
    """One scan: measured curve and the run-mean prediction of every condition of a family."""
    work_folder = config["paths"]["work_folder"]
    networks = [network for network in config["networks"] if (work_folder / "predictions" / f"{network}.csv").exists()]
    if not networks:
        print(f"{name} skipped, no predictions")
        return
    series = family_series(family)
    figure, axes = plt.subplots(len(networks), 2, squeeze=False,
                                figsize=(TEXT_WIDTH_IN, 2.3 * len(networks) + 0.4 + 0.12 * len(series)))
    for row, network in enumerate(networks):
        curves = run_mean_curves(work_folder, network, [condition for condition, _, _, _ in series])
        curves = curves[curves["scan_id"] == scan_id]
        measured = curves.drop_duplicates("frame").sort_values("frame")
        early_axis, whole_axis = axes[row]
        for ax in (early_axis, whole_axis):
            ax.plot(measured["time_min"], measured["measured"], "-o", color="black", lw=2.0, ms=2.5,
                    label="Measured arterial curve")
            for condition, label, colour, style in series:
                curve = curves[curves["condition"] == condition].sort_values("frame")
                ax.plot(curve["time_min"], curve["predicted"], style, color=colour, lw=1.3, label=label)
        early_axis.set_xlim(0, 3.0)
        early_axis.set_ylabel(f"{NETWORK_NAMES[network]}\nActivity [SUV]")
        whole_axis.set_yscale("log")
        whole_axis.set_ylim(bottom=0.01)
        if row == 0:
            early_axis.set_title("First 3 min, linear scale")
            whole_axis.set_title("Whole scan, logarithmic scale")
    for ax in axes[-1]:
        ax.set_xlabel("Time [min]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2)
    figure.suptitle(f"Scan {scan_id}, mean over the converged runs of its fold\n"
                    f"{FAMILY_TITLES.get(family['family'], family['family'])}", fontsize=9)
    legend_inches = 0.17 * ((len(labels) + 1) // 2) + 0.1
    figure.tight_layout(rect=(0, legend_inches / figure.get_figheight(), 1, 1))
    save(figure, figures_folder, name)


def draw_bias_variance(config, figures_folder, family, name):
    """Mean squared error split into bias^2 and variance (analyses/bias_variance.py), reference and conditions."""
    work_folder = config["paths"]["work_folder"]
    networks = [network for network in config["networks"]
                if (work_folder / "analyses" / network / "bias_variance.csv").exists()]
    if not networks:
        print(f"{name} skipped, run analyses/bias_variance.py first")
        return
    conditions = [family["reference"], *family["conditions"]]
    figure, axes = plt.subplots(1, len(networks), squeeze=False, sharey=True,
                                figsize=(TEXT_WIDTH_IN, 0.32 * len(conditions) + 1.3))
    for ax, network in zip(axes[0], networks):
        table = pd.read_csv(work_folder / "analyses" / network / "bias_variance.csv").set_index("condition")
        present = [condition for condition in conditions if condition in table.index]
        y = np.arange(len(present))
        bias_squared = table.loc[present, "bias_squared"].to_numpy()
        variance = table.loc[present, "variance"].to_numpy()
        ax.barh(y, bias_squared, height=0.6, color=NETWORK_COLOURS[network], label="Bias²")
        ax.barh(y, variance, left=bias_squared, height=0.6, color=NETWORK_COLOURS[network], alpha=0.4,
                label="Variance between runs")
        for position, total in zip(y, bias_squared + variance):
            ax.annotate(f"{total:.3f}", xy=(total, position), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=6.5, color=NEUTRAL)
        ax.set_yticks(y)
        ax.set_yticklabels([condition_label(condition) + (" (reference)" if condition == family["reference"] else "")
                            for condition in present])
        ax.set_title(NETWORK_NAMES[network])
        ax.set_xlabel("Mean squared error [SUV²]")
        ax.grid(axis="y", visible=False)
    axes[0, 0].invert_yaxis()  # shared axis: inverting it once inverts every panel
    handles = [matplotlib.patches.Patch(color=NEUTRAL, label="Bias² (dark)"),
               matplotlib.patches.Patch(color=NEUTRAL, alpha=0.4, label="Variance between runs (light)")]
    figure.legend(handles=handles, loc="lower center", ncol=2)
    figure.suptitle(f"Squared error split into bias² and variance, mean over scans\n"
                    f"{FAMILY_TITLES.get(family['family'], family['family'])}", fontsize=9)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    save(figure, figures_folder, name)


def draw_forest_figure(config, figures_folder, file_name, panels, xlabel, name, selection=None):
    """Forest plot of the comparison families (all, or a selection); one panel per (metric, panel title)."""
    tables = read_results(config, file_name)
    if not tables:
        print(f"{name} skipped, no {file_name}")
        return
    networks = [network for network in config["networks"] if any(key[0] == network for key in tables)]
    rows = forest_rows(config, tables, selection)
    figure, axes = plt.subplots(1, len(panels), figsize=(TEXT_WIDTH_IN, 0.24 * len(rows) + 1.3),
                                sharey=True, squeeze=False)
    for ax, (metric, title) in zip(axes[0], panels):
        draw_forest(ax, rows, tables, metric, networks)
        if title:
            ax.set_title(title)
    label_forest_rows(axes[0, 0], rows)
    legend_fraction = 0.75 / figure.get_figheight()  # the legend (title and two lines) below the axes
    if len(panels) == 1:
        axes[0, 0].set_xlabel(xlabel)
        figure.tight_layout(rect=(0, legend_fraction, 1, 1))
    else:
        figure.supxlabel(xlabel, fontsize=8, y=legend_fraction + 0.01, va="bottom")
        figure.tight_layout(rect=(0, legend_fraction + 0.04, 1, 1))
    forest_legend(figure, networks)
    save(figure, figures_folder, name)


TIME_WINDOWS = [(0.0, 2.5), (2.5, 5.5), (5.5, 40.5)]
# The forward shift at every iteration count and PSF width, against the uncorrected baseline.
ITERATION_SWEEP = {"RL": {3: "rl_k3_forward_shift", 10: "rl_k10_forward_shift", 15: "rl_forward_shift",
                          20: "rl_k20_forward_shift"},
                   "RVC": {10: "rvc_k10_forward_shift", 15: "rvc_forward_shift", 20: "rvc_k20_forward_shift"}}
PSF_SWEEP = {"RL": {0.9: "rl_psf090_forward_shift", 1.0: "rl_forward_shift", 1.1: "rl_psf110_forward_shift"},
             "RVC": {0.9: "rvc_psf090_forward_shift", 1.0: "rvc_forward_shift", 1.1: "rvc_psf110_forward_shift"}}


def per_scan_values(config, network, column):
    """Per condition and scan: the mean of a stage-07 curve metric over the scan's converged runs."""
    metrics = read_results(config, "curve_metrics.csv").get((network, "single_run"))
    if metrics is None or column not in metrics.columns:
        return None
    return metrics.groupby(["condition", "scan_id"])[column].mean()


def draw_agreement(config, figures_folder, family, name):
    """Peak ratio and AUC ratio against the correct value 1: median, quartiles and 5th-95th percentile over scans."""
    networks = list(config["networks"])
    conditions = [family["reference"], *family["conditions"]]
    columns = [("peak_ratio", "Peak ratio (predicted / measured)"), ("auc_ratio", "AUC ratio (predicted / measured)")]
    figure, axes = plt.subplots(1, 2, sharey=True, figsize=(TEXT_WIDTH_IN, 0.36 * len(conditions) + 1.5))
    drawn = False
    for ax, (column, label) in zip(axes, columns):
        for offset, network in zip([-0.15, 0.15], networks):
            values = per_scan_values(config, network, column)
            if values is None:
                continue
            for position, condition in enumerate(conditions):
                if condition not in values.index.get_level_values(0):
                    continue
                low, q25, median, q75, high = np.percentile(values.loc[condition], [5, 25, 50, 75, 95])
                y = position + offset
                ax.hlines(y, low, high, color=NETWORK_COLOURS[network], lw=0.8)
                ax.hlines(y, q25, q75, color=NETWORK_COLOURS[network], lw=3.0)
                ax.plot(median, y, "o", color="white", ms=4, markeredgecolor=NETWORK_COLOURS[network])
                drawn = True
        ax.axvline(1.0, color="black", lw=0.8, ls="--")
        ax.set_xlabel(label)
        ax.grid(axis="y", visible=False)
    if not drawn:
        plt.close(figure)
        print(f"{name} skipped, no stage-07 results with peak_ratio (rerun stage 07)")
        return
    axes[0].set_yticks(range(len(conditions)))
    axes[0].set_yticklabels([condition_label(c) + (" (reference)" if c == family["reference"] else "") for c in conditions])
    axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], color=NETWORK_COLOURS[n], lw=3, label=NETWORK_NAMES[n]) for n in networks]
    figure.legend(handles=handles, loc="lower center", ncol=2)
    figure.suptitle("Agreement with the measured curve: dot median, thick bar quartiles, thin line 5th–95th "
                    f"percentile over scans\n{FAMILY_TITLES.get(family['family'], family['family'])}", fontsize=9)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    save(figure, figures_folder, name)


def draw_paired_values(config, figures_folder, family, name):
    """RMSE of reference and condition per scan, joined by a line (the pairs the Wilcoxon test uses)."""
    networks = list(config["networks"])
    reference = family["reference"]
    figure, axes = plt.subplots(len(family["conditions"]), len(networks), squeeze=False, sharey=True,
                                figsize=(TEXT_WIDTH_IN, 1.9 * len(family["conditions"]) + 0.5))
    for column, network in enumerate(networks):
        values = per_scan_values(config, network, "rmse")
        if values is None:
            continue
        for row, condition in enumerate(family["conditions"]):
            ax = axes[row, column]
            if condition not in values.index.get_level_values(0):
                continue
            pair = pd.concat([values.loc[reference].rename("reference"), values.loc[condition].rename("condition")],
                             axis=1, join="inner")
            for _, scan in pair.iterrows():
                ax.plot([0, 1], [scan["reference"], scan["condition"]], color="#9a9993", lw=0.6, alpha=0.6)
            ax.plot([0, 1], [pair["reference"].median(), pair["condition"].median()], "-o",
                    color=NETWORK_COLOURS[network], lw=2.0, ms=5, label="median")
            ax.set_xticks([0, 1])
            ax.set_xticklabels([condition_label(reference), condition_label(condition)])
            ax.set_xlim(-0.3, 1.3)
            ax.set_title(NETWORK_NAMES[network])
            ax.grid(axis="x", visible=False)
            if column == 0:
                ax.set_ylabel("RMSE [SUV]")
    figure.suptitle(f"RMSE per scan, reference and condition (mean over converged runs)\n"
                    f"{FAMILY_TITLES.get(family['family'], family['family'])}", fontsize=9)
    figure.tight_layout()
    save(figure, figures_folder, name)


def draw_time_windows(config, figures_folder, family, name):
    """RMSE within each part of the curve: median and quartiles over scans of the per-scan mean over runs."""
    work_folder = config["paths"]["work_folder"]
    networks = [network for network in config["networks"] if (work_folder / "predictions" / f"{network}.csv").exists()]
    series = family_series(family)
    figure, axes = plt.subplots(1, len(networks), squeeze=False, sharey=False,
                                figsize=(TEXT_WIDTH_IN, 2.8 + 0.12 * len(series)))
    step = min(0.17, 0.8 / len(series))
    for ax, network in zip(axes[0], networks):
        predictions = read_converged_predictions(work_folder, network)
        predictions = predictions[predictions["condition"].isin([condition for condition, _, _, _ in series])]
        for index, (condition, label, colour, _) in enumerate(series):
            rows = predictions[predictions["condition"] == condition]
            medians, lows, highs = [], [], []
            for start, stop in TIME_WINDOWS:
                part = rows[(rows["time_min"] >= start) & (rows["time_min"] <= stop)]
                per_run = (part.assign(squared_error=(part["predicted"] - part["measured"]) ** 2)
                           .groupby(["scan_id", "fold", "run"])["squared_error"].mean() ** 0.5)
                per_scan = per_run.groupby("scan_id").mean()
                q25, median, q75 = np.percentile(per_scan, [25, 50, 75])
                medians.append(median), lows.append(q25), highs.append(q75)
            x = np.arange(len(TIME_WINDOWS)) + (index - (len(series) - 1) / 2) * step
            ax.vlines(x, lows, highs, color=colour, lw=2.0, alpha=0.8)
            ax.plot(x, medians, "o", color=colour, ms=4.5, markeredgecolor="white", label=label)
        ax.set_xticks(range(len(TIME_WINDOWS)))
        ax.set_xticklabels([f"{start:g}–{stop:g} min" for start, stop in TIME_WINDOWS])
        ax.set_ylim(bottom=0)
        ax.set_title(NETWORK_NAMES[network])
        ax.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("RMSE within the part [SUV]\n(median and quartiles over scans)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2)
    figure.suptitle(f"Error by part of the curve\n{FAMILY_TITLES.get(family['family'], family['family'])}", fontsize=9)
    legend_inches = 0.17 * ((len(labels) + 1) // 2) + 0.1
    figure.tight_layout(rect=(0, legend_inches / figure.get_figheight(), 1, 1))
    save(figure, figures_folder, name)


def draw_sweeps(config, figures_folder):
    """Step 6: the forward shift (uncorrected weights shown corrected input) against the correction setting.
    Columns: networks. Rows: change in RMSE against the baseline (median and 95 % CI, stage 07) and the
    median peak ratio (stage 07), first against the iteration count k, then against the PSF width. RL and
    RVC are separate lines; the pre-registered RL k = 3 is starred; the dotted line is the baseline's peak ratio."""
    comparisons = read_results(config, "comparisons.csv")
    networks = [network for network in config["networks"] if (network, "single_run") in comparisons]
    if not networks:
        print("scaling skipped, no stage-07 results")
        return
    figure, axes = plt.subplots(4, len(networks), squeeze=False, figsize=(TEXT_WIDTH_IN, 8.4))
    sweeps = [(ITERATION_SWEEP, "Iterations k"), (PSF_SWEEP, "PSF width (relative to the measured PSF)")]
    for pair, (sweep, xlabel) in enumerate(sweeps):
        for column, network in enumerate(networks):
            table = comparisons[(network, "single_run")]
            peak = per_scan_values(config, network, "peak_ratio")
            change_axis, peak_axis = axes[2 * pair, column], axes[2 * pair + 1, column]
            for method, folder, style in [("RL", "rl_i15", "-"), ("RVC", "rvc_i15", "--")]:
                xs, medians, lows, highs, peaks = [], [], [], [], []
                for x, condition in sorted(sweep[method].items()):
                    match = table[(table["metric"] == "rmse") & (table["condition"] == condition)
                                  & (table["reference"] == "baseline")]
                    if match.empty:
                        continue
                    xs.append(x)
                    medians.append(match.iloc[0]["median_difference"])
                    lows.append(match.iloc[0]["ci_low"])
                    highs.append(match.iloc[0]["ci_high"])
                    peaks.append(float(peak.loc[condition].median()) if peak is not None else np.nan)
                colour = CORRECTION_COLOURS[folder]
                change_axis.errorbar(xs, medians, yerr=[np.subtract(medians, lows), np.subtract(highs, medians)],
                                     fmt="o" + style, color=colour, lw=1.4, ms=4, capsize=2, label=method)
                peak_axis.plot(xs, peaks, "o" + style, color=colour, lw=1.4, ms=4, label=method)
                if 3 in xs:  # the pre-registered k = 3
                    for ax, value in [(change_axis, medians[xs.index(3)]), (peak_axis, peaks[xs.index(3)])]:
                        ax.plot(3, value, "*", color=colour, ms=8, markeredgecolor="black", markeredgewidth=0.5,
                                zorder=5)
            change_axis.axhline(0, color="black", lw=0.8)
            if peak is not None:
                peak_axis.axhline(float(peak.loc["baseline"].median()), color=REFERENCE_COLOUR, lw=1.0, ls=":")
            peak_axis.axhline(1.0, color="black", lw=0.8, ls="--")
            ticks = sorted({x for method in sweep.values() for x in method})
            for ax in (change_axis, peak_axis):
                ax.set_xticks(ticks)
            plt.setp(change_axis.get_xticklabels(), visible=False)
            peak_axis.set_xlabel(xlabel)
            if pair == 0:
                change_axis.set_title(NETWORK_NAMES[network])
            if column == 0:
                change_axis.set_ylabel("\u0394RMSE vs baseline\n[SUV] (median, 95 % CI)")
                peak_axis.set_ylabel("Peak ratio\n(median over scans)")
    # Each quantity on one scale across networks and settings, so the panels compare directly.
    for rows in ([0, 2], [1, 3]):
        values = [ax.get_ylim() for row in rows for ax in axes[row]]
        for row in rows:
            for ax in axes[row]:
                ax.set_ylim(min(v[0] for v in values), max(v[1] for v in values))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], lw=0, marker="*", ms=8, color="#888888", markeredgecolor="black",
                           markeredgewidth=0.5),
                plt.Line2D([], [], color=REFERENCE_COLOUR, lw=1.0, ls=":"),
                plt.Line2D([], [], color="black", lw=0.8, ls="--")]
    labels += ["k = 3 (pre-registered)", "baseline peak ratio", "peak ratio 1 (correct)"]
    figure.legend(handles, labels, loc="lower center", ncol=3)
    figure.suptitle("Uncorrected weights shown corrected input (forward shift), against the correction setting",
                    fontsize=9)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    save(figure, figures_folder, "scaling")


def figure_retrained(config, figures_folder):
    """Step 4: does retraining on corrected images help? A: RL, RVC and the pre-registered RL k = 3
    (median paired difference in RMSE, both networks and estimands). B: the change per scan for RL and
    RVC, single run (the pairs behind the medians in A)."""
    tables = read_results(config, "comparisons.csv")
    metrics = read_results(config, "curve_metrics.csv")
    networks = [network for network in config["networks"] if (network, "single_run") in tables]
    if not networks:
        print("results_retrained skipped, no stage-07 results")
        return
    rows = forest_rows(config, tables, {"retrained": None, "rl_k3": None})
    figure = plt.figure(figsize=(TEXT_WIDTH_IN, 6.6))
    grid = figure.add_gridspec(3, len(networks), height_ratios=[1.25, 1, 1], hspace=0.85)
    forest_axis = figure.add_subplot(grid[0, :])
    draw_forest(forest_axis, rows, tables, "rmse", networks)
    label_forest_rows(forest_axis, rows)
    forest_axis.set_xlabel("Median paired difference in RMSE vs the baseline [SUV]   (below 0: lower error)")
    forest_axis.set_title("A  All 70 scans, both networks", loc="left")
    first = None
    for column, network in enumerate(networks):
        per_scan = metrics[(network, "single_run")].groupby(["condition", "scan_id"])["rmse"].mean()
        for row, condition in enumerate(["rl", "rvc"]):
            ax = figure.add_subplot(grid[row + 1, column], sharey=first)
            first = first or ax
            draw_paired_axis(ax, per_scan, tables[(network, "single_run")], condition, "baseline", network)
            if column == 0:
                ax.set_ylabel("\u0394RMSE per scan [SUV]")
            if row == 1:
                ax.set_xlabel("Scans, sorted by the change")
    forest_legend(figure, networks)
    figure.subplots_adjust(left=0.30, right=0.98, top=0.96, bottom=0.17)
    heading_y = first.get_position().y1 + 0.045  # above the titles of the first row of panel B
    figure.text(0.01, heading_y, "B  Change per scan against the baseline (mean over converged runs; band: 95 % CI)",
                fontsize=8.5)
    save(figure, figures_folder, "results_retrained")


def figure_motion(config, figures_folder):
    """Step 7: motion correction, matched and shown to the uncorrected weights."""
    draw_forest_figure(config, figures_folder, "comparisons.csv", [("rmse", "")],
                       "Median paired difference in RMSE [SUV]   (below 0: lower error than the baseline)",
                       "results_motion",
                       selection={"motion": None,
                                  "forward_shift": ["mc_forward_shift", "mc_then_rl_forward_shift",
                                                    "rl_then_mc_forward_shift"]})


def figure_mismatch_overview(config, figures_folder):
    """Step 5, first figure: the cohort-mean input function, matched and mismatched, RL arm.
    Per scan the mean over its converged runs, then the mean over the 70 scans, in absolute time
    (no alignment). Top: the first 3 min; bottom: from 2.5 min, on its own scale. Columns: networks.
    The same four conditions and colours as the anatomy figure."""
    work_folder = config["paths"]["work_folder"]
    series = [("baseline", "Fitted on uncorrected, shown uncorrected (matched)", REFERENCE_COLOUR, "-"),
              ("rl", "Fitted on RL, shown RL (matched)", "black", ":"),
              ("rl_forward_shift", "Fitted on uncorrected, shown RL (forward shift)", FORWARD_COLOUR, "-"),
              ("rl_reverse_shift", "Fitted on RL, shown uncorrected (reverse shift)", REVERSE_COLOUR, "-")]
    networks = [network for network in config["networks"] if (work_folder / "predictions" / f"{network}.csv").exists()]
    if not networks:
        print("mismatch_overview skipped, no predictions")
        return
    figure, axes = plt.subplots(2, len(networks), squeeze=False, sharey="row", figsize=(TEXT_WIDTH_IN, 4.8))
    for column, network in enumerate(networks):
        curves = run_mean_curves(work_folder, network, [condition for condition, _, _, _ in series])
        time_min = curves.groupby("frame")["time_min"].mean()
        measured = curves[curves["condition"] == "baseline"].groupby("frame")["measured"].mean()
        for row, (start, stop) in enumerate([(0.0, 3.0), (2.5, 41.0)]):
            ax = axes[row, column]
            ax.plot(time_min.loc[measured.index], measured, "-", color="black", lw=2.0, label="Measured arterial curve")
            for condition, label, colour, style in series:
                mean = curves[curves["condition"] == condition].groupby("frame")["predicted"].mean()
                ax.plot(time_min.loc[mean.index], mean, style, color=colour, lw=1.5, label=label)
            ax.set_xlim(start, stop)
            if row == 1:
                late = measured[time_min.loc[measured.index] >= start]
                ax.set_ylim(0, float(late.max()) * 1.6)
                ax.set_xlabel("Time [min]")
            if row == 0:
                ax.set_title(NETWORK_NAMES[network])
            if column == 0:
                ax.set_ylabel(["Activity [SUV], first 3 min", "Activity [SUV], from 2.5 min"][row])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2)
    figure.suptitle("Mean input function over the 70 scans (each scan: mean over its converged runs), RL arm",
                    fontsize=9)
    figure.tight_layout(rect=(0, 0.13, 1, 1))
    save(figure, figures_folder, "mismatch_overview")


def figure_mismatch_anatomy(config, figures_folder):
    """Step 5: what happens when training and test input differ, and what kind of error it is.
    One figure per arm (RL, RVC) and network, 2 x 2:
      A  cohort curves (every scan aligned on its measured peak)
      B  area by part of the curve
      C  squared error split into bias^2 and variance, with the scale factor of the mismatch
      D  the comparisons for the single run and for the best of 10 runs by validation loss
    The RL k = 3 arm (pre-registered) is drawn the same way, so it can be read beside RL k = 15."""
    work_folder = config["paths"]["work_folder"]
    comparisons = read_results(config, "comparisons.csv")
    for arm in ["rl", "rvc", "rl_k3"]:
        method = {"rl": "RL", "rvc": "RVC", "rl_k3": "RL k=3"}[arm]
        series = [("baseline", "Fitted on uncorrected, shown uncorrected (matched)", REFERENCE_COLOUR, "-"),
                  (arm, f"Fitted on {method}, shown {method} (matched)", "black", ":"),
                  (f"{arm}_forward_shift", f"Fitted on uncorrected, shown {method} (forward shift)", FORWARD_COLOUR, "-"),
                  (f"{arm}_reverse_shift", f"Fitted on {method}, shown uncorrected (reverse shift)", REVERSE_COLOUR, "-")]
        names = [condition for condition, _, _, _ in series]
        for network in config["networks"]:
            needed = [work_folder / "predictions" / f"{network}.csv",
                      work_folder / "analyses" / network / "bias_variance.csv",
                      work_folder / "analyses" / network / "mismatch_scaling.csv",
                      work_folder / "analyses" / network / "best_of_10_comparisons.csv"]
            if not all(path.exists() for path in needed) or (network, "single_run") not in comparisons:
                print(f"mismatch_anatomy_{arm}_{network} skipped, run stages 06-07 and analyses/ first")
                continue
            figure, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH_IN, 6.2))
            curve_axis, area_axis, split_axis, effect_axis = axes.flat

            # A, B
            n_scans = draw_cohort_axes(curve_axis, area_axis, run_mean_curves(work_folder, network, names), series)
            curve_axis.set_ylabel(f"Activity / measured peak\n(mean over {n_scans} scans)")
            curve_axis.set_title("A  Bolus peak (ensemble per scan),\n    scans aligned on their peak", loc="left")
            area_axis.set_title("B  Area by part of the curve\n    (ensemble per scan)", loc="left")

            # C: bias^2 (dark) and variance (light) per condition, scale factor on the mismatched ones
            split = pd.read_csv(work_folder / "analyses" / network / "bias_variance.csv").set_index("condition")
            scaling = pd.read_csv(work_folder / "analyses" / network / "mismatch_scaling.csv").set_index("condition")
            for position, (condition, _, colour, _) in enumerate(series):
                bias_squared, variance = split.loc[condition, "bias_squared"], split.loc[condition, "variance"]
                split_axis.barh(position, bias_squared, height=0.6, color=colour)
                split_axis.barh(position, variance, left=bias_squared, height=0.6, color=colour, alpha=0.4)
                split_axis.annotate(f"{bias_squared + variance:.2f}", xy=(bias_squared + variance, position),
                                    xytext=(3, 0), textcoords="offset points", va="center", fontsize=7,
                                    color=INK_COLOUR)
            block = ["One scale factor per network and scan", "(median; share of the mismatch it removes)"]
            for condition, label in [(f"{arm}_forward_shift", "forward"), (f"{arm}_reverse_shift", "reverse")]:
                if condition not in scaling.index:  # analyses/mismatch_scaling.py not rerun for this arm
                    block.append(f"{label}:  run analyses/mismatch_scaling.py")
                    continue
                block.append(f"{label}:  \u00d7{scaling.loc[condition, 'scale_median']:.2f}   "
                             f"{100 * scaling.loc[condition, 'share_removed_median']:.0f} %")
            split_axis.text(0.98, 0.03, "\n".join(block), transform=split_axis.transAxes, ha="right", va="bottom",
                            fontsize=6.8, color=INK_COLOUR,
                            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc"))
            split_axis.set_yticks(range(len(series)))
            split_axis.set_yticklabels(["matched, uncorrected", f"matched, {method}", "forward shift", "reverse shift"])
            split_axis.set_ylim(len(series) + 0.9, -0.6)  # bars on top, an empty slot below for the scale factors
            split_axis.set_xlim(0, split.loc[names, ["bias_squared", "variance"]].sum(axis=1).max() * 2.3)
            split_axis.set_xlabel("MSE [SUV²]: bias² (dark) + variance between runs (light)")
            split_axis.grid(axis="y", visible=False)
            split_axis.set_title("C  MSE: squared bias + run variance", loc="left")

            # D: the same comparisons for the three estimands: single run, ensemble, best of 10
            single = comparisons[(network, "single_run")]
            ensemble = comparisons.get((network, "ensemble"))
            best = pd.read_csv(work_folder / "analyses" / network / "best_of_10_comparisons.csv")
            estimands = [(-0.2, single, "o", "filled"), (0.0, ensemble, "D", "white"), (0.2, best, "s", "white")]
            rows = [(arm, "baseline", f"matched {method}\nvs uncorrected"),
                    (f"{arm}_forward_shift", "baseline", "forward shift\nvs matched uncorrected"),
                    (f"{arm}_reverse_shift", arm, f"reverse shift\nvs matched {method}")]
            for position, (condition, reference, _) in enumerate(rows):
                colour = dict((c, col) for c, _, col, _ in series)[condition]
                for offset, table, marker, face in estimands:
                    if table is None:
                        continue
                    result = table[(table["metric"] == "rmse") & (table["condition"] == condition)
                                   & (table["reference"] == reference)].iloc[0]
                    effect_axis.hlines(position + offset, result["ci_low"], result["ci_high"], color=colour, lw=1.2)
                    effect_axis.plot(result["median_difference"], position + offset, marker, color=colour,
                                     markerfacecolor=colour if face == "filled" else face, ms=4.5)
            effect_axis.axvline(0, color="black", lw=0.8)
            effect_axis.set_yticks(range(len(rows)))
            effect_axis.set_yticklabels([label for _, _, label in rows], fontsize=7)
            effect_axis.invert_yaxis()
            effect_axis.set_xlabel("ΔRMSE [SUV] (median, 95 % CI)")
            effect_axis.grid(axis="y", visible=False)
            effect_axis.set_title("D  \u25cf single run  \u25c7 ensemble  \u25a1 best of 10", loc="left")

            handles, labels = curve_axis.get_legend_handles_labels()
            figure.legend(handles, labels, loc="lower center", ncol=2)
            figure.suptitle(f"Matched and mismatched input, {method}, {NETWORK_NAMES[network]}", fontsize=9.5)
            figure.tight_layout(rect=(0, 0.1, 1, 1))
            save(figure, figures_folder, f"mismatch_anatomy_{arm}_{network}")


def overview_example_scans(config):
    """The thesis example scan and the scans with the lowest, median and highest baseline RMSE (first network)."""
    scans = [config["figures"]["example_scan"]]
    values = per_scan_values(config, next(iter(config["networks"])), "rmse")
    if values is not None:
        ranked = values.loc["baseline"].sort_values()
        for scan_id in [ranked.index[0], ranked.index[len(ranked) // 2], ranked.index[-1]]:
            if scan_id not in scans:
                scans.append(scan_id)
    return scans


# ---------------------------------------------------------------------------
# The thesis figures.
# ---------------------------------------------------------------------------
def figure_diagnostics_by_frame(config, figures_folder):
    draw_diagnostics(config, figures_folder, ["rl_i15", "rvc_i15"],
                     [CORRECTION_COLOURS["rl_i15"], CORRECTION_COLOURS["rvc_i15"]], "diagnostics_by_frame")


def figure_effects_rmse(config, figures_folder):
    draw_forest_figure(config, figures_folder, "comparisons.csv", [("rmse", "")],
                       "Median paired difference in RMSE [SUV]   (below 0: lower error than the reference)",
                       "effects_rmse")


def figure_effects_ki(config, figures_folder):
    """Step 9: how much of the curve error reaches Ki (the story's comparisons), and every comparison (appendix)."""
    panels = [(f"ki_abs_error_pct_{voi}", voi) for voi in config["kinetics"]["vois"]]
    xlabel = "Median paired difference in the absolute Patlak Ki error [percentage points]   (below 0: lower error)"
    draw_forest_figure(config, figures_folder, "kinetics_comparisons.csv", panels, xlabel, "effects_ki",
                       selection={"retrained": None, "rl_k3": None,
                                  "forward_shift": ["rl_forward_shift", "rvc_forward_shift"],
                                  "rl_reverse_shift": None, "rvc_reverse_shift": None,
                                  "rl_k3_forward_shift": None, "rl_k3_reverse_shift": None})
    draw_forest_figure(config, figures_folder, "kinetics_comparisons.csv", panels, xlabel, "effects_ki_all")


# ---------------------------------------------------------------------------
# The overview: the same plot types for every family, every metric and every correction.
# ---------------------------------------------------------------------------
def draw_overview(config, figures_folder):
    overview_folder = figures_folder / "overview"
    overview_folder.mkdir(parents=True, exist_ok=True)
    families = config["evaluation"]["comparison_families"]

    for metric in config["evaluation"]["tested_metrics"]:
        direction = "below 0: lower error" if metric in ("rmse", "nrmse", "mae") else "difference in the metric"
        draw_forest_figure(config, overview_folder, "comparisons.csv", [(metric, "")],
                           f"Median paired difference in {METRIC_LABELS.get(metric, metric)}   ({direction})",
                           f"effects_{metric}")
    vois = config["kinetics"]["vois"]
    draw_forest_figure(config, overview_folder, "kinetics_comparisons.csv",
                       [(f"ki_abs_error_pct_{voi}", voi) for voi in vois],
                       "Median paired difference in the absolute Patlak Ki error [percentage points]", "effects_ki_absolute")
    draw_forest_figure(config, overview_folder, "kinetics_comparisons.csv",
                       [(f"ki_error_pct_{voi}", voi) for voi in vois],
                       "Median paired difference in the signed Patlak Ki error [percentage points]", "effects_ki_signed")

    example_scans = overview_example_scans(config)
    for family in families:
        tag = family["family"]
        draw_paired_difference(config, overview_folder, family, f"paired_difference_{tag}")
        draw_paired_values(config, overview_folder, family, f"paired_rmse_{tag}")
        draw_agreement(config, overview_folder, family, f"agreement_{tag}")
        draw_time_windows(config, overview_folder, family, f"error_by_time_window_{tag}")
        draw_cohort_curves(config, overview_folder, family_series(family), f"cohort_curves_{tag}",
                           FAMILY_TITLES.get(tag, tag) + ", converged runs")
        for scan_id in example_scans:
            draw_example_curves(config, overview_folder, family, scan_id, f"example_curves_{scan_id}_{tag}")
        draw_bias_variance(config, overview_folder, family, f"bias_variance_{tag}")

    # The thesis mismatch figure for the RVC arm, and the sweeps of the sensitivity conditions.
    draw_cohort_curves(config, overview_folder,
                       [("baseline", "Uncorrected weights, uncorrected input (matched)", "#6b6b6b", "-"),
                        ("rvc", "RVC weights, RVC input (matched)", "black", ":"),
                        ("rvc_forward_shift", "Uncorrected weights, RVC input (forward shift)", FORWARD_COLOUR, "-"),
                        ("rvc_reverse_shift", "RVC weights, uncorrected input (reverse shift)", REVERSE_COLOUR, "-")],
                       "mismatch_curves_rvc", "Matched and mismatched input, RVC arm, converged runs")

    for correction in config["pvc"]["corrections"]:
        folder = correction_folder_name(correction)
        colour = CORRECTION_COLOURS["rl_i15"] if correction["method"] == "RL" else CORRECTION_COLOURS["rvc_i15"]
        draw_diagnostics(config, overview_folder, [folder], [colour], f"diagnostics_{folder}")


def main():
    parser = argparse.ArgumentParser(description="Stage 09: figures.")
    parser.add_argument("--config", required=True, help="path to configs/study.yaml")
    parser.add_argument("--only", nargs="*", default=None, help="draw only these thesis figures (by name)")
    parser.add_argument("--overview", action="store_true", help="also draw the overview in figures/overview/")
    args = parser.parse_args()
    config = load_study_config(args.config)
    for condition in config["conditions"]:
        CONDITIONS[condition["name"]] = (condition["weights"], condition["shown"])
    figures_folder = config["paths"]["work_folder"] / "figures"
    figures_folder.mkdir(parents=True, exist_ok=True)

    # In the order of the thesis argument (the reproduction of the 2026 paper, step 1,
    # is drawn by analyses/reproduce_paper_2026.py).
    figures = {"example_input_function": figure_example_input_function,      # 2 what the network sees
               "example_pvc_image": figure_example_pvc_image,                # 3 what PVC does
               "diagnostics_by_frame": figure_diagnostics_by_frame,          # 3
               "results_retrained": figure_retrained,                         # 4 does retraining help
               "mismatch_overview": figure_mismatch_overview,                 # 5 matched and mismatched input
               "mismatch_anatomy": figure_mismatch_anatomy,                   # 5 and what kind of error it is
               "scaling": draw_sweeps,                                        # 6 against correction strength
               "results_motion": figure_motion,                               # 7 motion correction
               "convergence_val_loss": figure_convergence,                    # 8 training
               "effects_ki": figure_effects_ki,                               # 9 how much reaches Ki
               "effects_rmse": figure_effects_rmse}                           # every comparison (appendix)
    for name, draw in figures.items():
        if args.only is None or name in args.only:
            draw(config, figures_folder)
    if args.overview:
        draw_overview(config, figures_folder)


if __name__ == "__main__":
    main()
