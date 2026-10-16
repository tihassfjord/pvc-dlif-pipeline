"""One table per endpoint: single run, ensemble and best of 10, side by side.

For every comparison of the families in the config, the median paired
difference with its 95 % interval and the Holm-adjusted p-value(s) of the
three estimands: single run (stage 07/08; Wilcoxon / permutation), ensemble
(stage 07/08; Wilcoxon) and best of 10 (analyses/best_of_10.py; Wilcoxon).
This collects numbers already computed; nothing new is tested.

Reads   <work>/results/<network>/{single_run,ensemble}/comparisons.csv, kinetics_comparisons.csv
        <work>/analyses/<network>/best_of_10_comparisons.csv, best_of_10_kinetics_comparisons.csv
Writes  <work>/analyses/summary_rmse.csv and summary_ki.csv (both networks)

Usage
    python analyses/summary_table.py --config configs/local.yaml
"""

import argparse

import pandas as pd

from pvc_dlif_pipeline.config import load_study_config

ENDPOINTS = {"rmse": ("comparisons.csv", "best_of_10_comparisons.csv", ["rmse"]),
             "ki": ("kinetics_comparisons.csv", "best_of_10_kinetics_comparisons.csv",
                    ["ki_abs_error_pct_Brain", "ki_abs_error_pct_Myocardium"])}


def formatted(row, with_permutation):
    """'+0.012 [-0.003, +0.025] p 0.21' (and '/0.35' for the permutation test)."""
    text = (f"{row['median_difference']:+.3f} [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] "
            f"p {row['p_wilcoxon_holm']:.2g}")
    if with_permutation:
        text += f"/{row['p_permutation_holm']:.2g}"
    return text


def main():
    parser = argparse.ArgumentParser(description="Summary tables of the three estimands.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_study_config(args.config)
    work_folder = config["paths"]["work_folder"]
    keys = ["family", "metric", "condition", "reference"]

    for endpoint, (result_file, best_file, metrics) in ENDPOINTS.items():
        rows = []
        for network in config["networks"]:
            sources = {"single_run": work_folder / "results" / network / "single_run" / result_file,
                       "ensemble": work_folder / "results" / network / "ensemble" / result_file,
                       "best_of_10": work_folder / "analyses" / network / best_file}
            if not all(path.exists() for path in sources.values()):
                print(f"{endpoint}, {network}: not all inputs exist, skipped")
                continue
            tables = {name: pd.read_csv(path).set_index(keys) for name, path in sources.items()}
            single = tables["single_run"]
            for key in single.index:
                if key[1] not in metrics:
                    continue
                row = dict(zip(keys, key), network=network,
                           median_reference=round(single.loc[key, "median_reference"], 3))
                for name, table in tables.items():
                    row[name] = formatted(table.loc[key], name == "single_run") if key in table.index else ""
                rows.append(row)
        table = pd.DataFrame(rows)
        if not table.empty:
            table.to_csv(work_folder / "analyses" / f"summary_{endpoint}.csv", index=False)
            print(f"Written: summary_{endpoint}.csv ({len(table)} rows)")


if __name__ == "__main__":
    main()
