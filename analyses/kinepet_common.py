"""Shared settings for the KinePET analyses: paths and the PSF from the study configuration.

The configuration file is taken from --config on the command line (default configs/local.yaml), as for
the stages. KinePET runs in its own environment (it is not part of this repository); that environment
needs numpy, scipy, nibabel, matplotlib and pyyaml besides kinepet.

Outputs go to <work>/analyses/kinepet/.
"""
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def _config_path():
    argv = sys.argv
    if "--config" in argv:
        return Path(argv[argv.index("--config") + 1])
    return REPO / "configs" / "local.yaml"


with open(_config_path(), encoding="utf-8") as _f:
    CONFIG = yaml.safe_load(_f)
WORK = Path(CONFIG["paths"]["work_folder"])
DATA = Path(CONFIG["paths"]["group_data_folder"])
FWHM_MM = CONFIG["pvc"]["fwhm_mm"]          # x, y, z in mm
OUT = WORK / "analyses" / "kinepet"
OUT.mkdir(parents=True, exist_ok=True)
