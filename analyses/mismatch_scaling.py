"""How much of the train-test mismatch is a change of level?

A mismatched condition and the matched condition with the same weights differ
only in the input: the same network predicts the same scan from differently
corrected images. For every network and scan, the least-squares scale factor

    c = <s, m> / <m, m>      (s: mismatched prediction, m: matched prediction)

is the single factor that brings the matched curve closest to the mismatched
one, and the share of their squared difference that this one factor removes is

    1 - mean((s - c m)^2) / mean((s - m)^2).

A share near one means the mismatch mainly scales the curve up or down. Also
reported: the ratio of the peaks, and the ratio of the summed late samples
(from 2.5 min). Medians and quartiles over network-scan pairs; runs that did
not converge are left out.

Reads   <work>/predictions/<network>.csv
Writes  <work>/analyses/<network>/mismatch_scaling.csv

Usage
    python analyses/mismatch_scaling.py --config configs/local.yaml --network fcdlif
"""

import argparse

import numpy as np
import pandas as pd

from pvc_dlif_pipeline.config import load_study_config

# (mismatched condition, matched condition with the same weights)
PAIRS = [("rl_forward_shift", "baseline"), ("rvc_forward_shift", "baseline"),
         ("rl_reverse_shift", "rl"), ("rvc_reverse_shift", "rvc"),
         ("rl_k3_forward_shift", "baseline"), ("rl_k3_reverse_shift", "rl_k3")]
LATE_FROM_MIN = 2.5


def main():
    parser = argparse.ArgumentParser(description="Scale factor between mismatched and matched predictions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--network", required=True)
    args = parser.parse_args()
    config = load_study_config(args.config)
    work_folder = config["paths"]["work_folder"]

    predictions = pd.read_csv(work_folder / "predictions" / f"{args.network}.csv")
    predictions = predictions[predictions["converged"]]
    keys = ["scan_id", "fold", "run", "frame"]

    rows = []
    for mismatched, matched in PAIRS:
        shifted = predictions[predictions["condition"] == mismatched].set_index(keys)["predicted"]
        reference = predictions[predictions["condition"] == matched].set_index(keys)
        paired = pd.concat([shifted.rename("shifted"), reference["predicted"].rename("matched"),
                            reference["time_min"]], axis=1, join="inner").reset_index()
        values = []
        for _, curve in paired.groupby(["scan_id", "fold", "run"]):
            curve = curve.sort_values("frame")
            s, m = curve["shifted"].to_numpy(), curve["matched"].to_numpy()
            late = curve["time_min"].to_numpy() >= LATE_FROM_MIN
            scale = (s @ m) / (m @ m)
            share = 1.0 - np.mean((s - scale * m) ** 2) / np.mean((s - m) ** 2)
            values.append((scale, share, s.max() / m.max(), s[late].sum() / m[late].sum()))
        values = np.asarray(values)
        rows.append({"condition": mismatched, "matched": matched, "pairs": len(values),
                     "scale_median": np.median(values[:, 0]),
                     "scale_q25": np.percentile(values[:, 0], 25), "scale_q75": np.percentile(values[:, 0], 75),
                     "share_removed_median": np.median(values[:, 1]),
                     "peak_ratio_median": np.median(values[:, 2]), "late_ratio_median": np.median(values[:, 3])})
    table = pd.DataFrame(rows)

    output_folder = work_folder / "analyses" / args.network
    output_folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_folder / "mismatch_scaling.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
