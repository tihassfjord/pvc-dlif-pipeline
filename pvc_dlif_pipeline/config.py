"""Read the study configuration file."""

from pathlib import Path

import yaml


def load_study_config(config_path):
    """Return the study configuration (configs/study.yaml) as a dictionary.

    Every value under `paths` is turned into a pathlib.Path, so the scripts
    can build file paths with the `/` operator.
    """
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    config["paths"] = {name: Path(value) for name, value in config["paths"].items()}
    return config


def normalise_scan_id(scan_id):
    """Return a scan ID in one spelling, for comparing IDs from different sources.

    The DICOM files name the Sherbrooke scans "Sherbrooke_A" while the group's
    exclusion list writes "Sherbrooke-A"; both become "SHERBROOKE_A".
    """
    return str(scan_id).strip().replace("-", "_").upper()
