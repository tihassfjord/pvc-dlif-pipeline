"""The study configuration loads, and a local copy keeps the protocol unchanged."""

from pathlib import Path

import pytest
import yaml

from pvc_dlif_pipeline.config import load_study_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_study_config_loads_and_families_refer_to_known_conditions():
    config = load_study_config(CONFIGS / "study.yaml")
    names = {condition["name"] for condition in config["conditions"]}
    for family in config["evaluation"]["comparison_families"]:
        assert family["reference"] in names
        assert set(family["conditions"]) <= names


def test_local_config_differs_only_in_paths():
    local = CONFIGS / "local.yaml"
    if not local.exists():
        pytest.skip("no configs/local.yaml")
    study = yaml.safe_load((CONFIGS / "study.yaml").read_text(encoding="utf-8"))
    mine = yaml.safe_load(local.read_text(encoding="utf-8"))
    study.pop("paths")
    mine.pop("paths")
    assert mine == study
