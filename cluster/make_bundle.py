"""Copy what stage 05 needs for one or more representations into a folder for a GPU cluster.

The bundle holds only what training reads: this repository's code, the
manifest, the network inputs of the chosen representations, the measured
arterial curves, and the network definitions from the group's repository
(src/models/*.py; no weights). No DICOM, NIfTI or PVC output is copied.

The bundle has its own config, configs/study_cluster.yaml, with paths relative
to the bundle folder, so jobs are started from the bundle folder.

    <bundle>/code/          this repository
    <bundle>/work/          manifest.csv, network_inputs/<representation>/
    <bundle>/group_data/    AIF_SUV/
    <bundle>/group_code/    src/models/*.py

Usage
    python cluster/make_bundle.py --config configs/study.yaml --out /path/to/<bundle_name> --representations uncorrected
"""

import argparse
import shutil
from pathlib import Path

import yaml

from pvc_dlif_pipeline.config import load_study_config


def main():
    parser = argparse.ArgumentParser(description="Make a stage 05 bundle for a cluster.")
    parser.add_argument("--config", required=True, help="path to configs/study.yaml")
    parser.add_argument("--out", required=True, help="bundle folder to create (must not exist)")
    parser.add_argument("--representations", nargs="+", required=True, help="stage 04 folder names")
    args = parser.parse_args()

    config = load_study_config(args.config)
    paths = config["paths"]
    bundle = Path(args.out)
    if bundle.exists():
        raise SystemExit(f"{bundle} exists already; choose a new name so no other bundle is overwritten")
    repository = Path(__file__).resolve().parents[1]

    shutil.copytree(repository, bundle / "code",
                    ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", ".git", "jobs"))
    (bundle / "work").mkdir(parents=True)
    shutil.copy2(paths["work_folder"] / "manifest.csv", bundle / "work" / "manifest.csv")
    for representation in args.representations:
        shutil.copytree(paths["work_folder"] / "network_inputs" / representation,
                        bundle / "work" / "network_inputs" / representation)
    shutil.copytree(paths["group_data_folder"] / "AIF_SUV", bundle / "group_data" / "AIF_SUV")
    (bundle / "group_code" / "src" / "models").mkdir(parents=True)
    for source in (paths["group_code_folder"] / "src" / "models").glob("*.py"):
        shutil.copy2(source, bundle / "group_code" / "src" / "models" / source.name)

    with open(args.config, "r", encoding="utf-8") as config_file:
        cluster_config = yaml.safe_load(config_file)
    cluster_config["paths"] = {"dicom_folder": "unused", "group_data_folder": "group_data",
                               "group_code_folder": "group_code", "work_folder": "work"}
    with open(bundle / "code" / "configs" / "study_cluster.yaml", "w", encoding="utf-8") as config_file:
        yaml.safe_dump(cluster_config, config_file, sort_keys=False)
    print(f"Bundle written to {bundle}. Copy it to the cluster, e.g. with scp -r.")


if __name__ == "__main__":
    main()
