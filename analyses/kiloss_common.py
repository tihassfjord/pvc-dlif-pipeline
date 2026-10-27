"""Paths for the kiloss analyses (ridge, evaluation), from the study configuration (--config, default
configs/local.yaml). Outputs go to <work>/analyses/kiloss/. Runs in this repository's environment."""
import argparse
from pathlib import Path

from pvc_dlif_pipeline.config import load_study_config

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "local.yaml"))
CONFIG = load_study_config(_parser.parse_known_args()[0].config)
WORK = CONFIG["paths"]["work_folder"]
DATA = CONFIG["paths"]["group_data_folder"]
PYPET = CONFIG["paths"]["pypet_folder"]
A, B, C = (CONFIG["kinetics"][k] for k in ("plasma_ratio_a", "plasma_ratio_b", "plasma_ratio_c"))
OUT = WORK / "analyses" / "kiloss"
OUT.mkdir(parents=True, exist_ok=True)
