"""File layout of the group's preprocessed data, and the group's code.

The group's data folder holds one pickle per scan and modality:
    AIF_SUV/AIF_<ID>.pkl                arterial curve: AIF_t_int (min), AIF_A_int (SUV)
    IMG_SUV_96x48x48/IMG_<ID>.pkl       network input: IMG (frames, 96, 48, 48), IDIF_t (min)
    VOI_SUV/VOI_<ID>.pkl                tissue curves: VOI_t (min), VOI_A {region: curve}
"""

import pickle
import sys
from pathlib import Path

import numpy as np


def arterial_curve_path(group_data_folder, scan_id):
    return Path(group_data_folder) / "AIF_SUV" / f"AIF_{scan_id}.pkl"


def network_input_path(data_folder, scan_id, shape_zyx=(96, 48, 48)):
    """Path of a network input file; the folder name carries the input shape."""
    shape_text = "x".join(str(size) for size in shape_zyx)
    return Path(data_folder) / f"IMG_SUV_{shape_text}" / f"IMG_{scan_id}.pkl"


def tissue_curves_path(group_data_folder, scan_id):
    return Path(group_data_folder) / "VOI_SUV" / f"VOI_{scan_id}.pkl"


def load_arterial_curve(group_data_folder, scan_id):
    """Return (activity in SUV, time in minutes) of the measured arterial curve."""
    with open(arterial_curve_path(group_data_folder, scan_id), "rb") as pickle_file:
        content = pickle.load(pickle_file)
    activity = np.asarray(content["AIF_A_int"], dtype=float)
    time_min = np.asarray(content["AIF_t_int"], dtype=float)
    return activity, time_min


def load_tissue_curves(group_data_folder, scan_id):
    """Return ({VOI name: tissue curve in SUV}, time in minutes) of one scan."""
    with open(tissue_curves_path(group_data_folder, scan_id), "rb") as pickle_file:
        content = pickle.load(pickle_file)
    curves = {str(name): np.asarray(curve, dtype=float) for name, curve in dict(content["VOI_A"]).items()}
    time_min = np.asarray(content["VOI_t"], dtype=float)
    return curves, time_min


def load_network_input(data_folder, scan_id, shape_zyx=(96, 48, 48)):
    """Return (image (frames, z, y, x) as float32, frame times in minutes)."""
    with open(network_input_path(data_folder, scan_id, shape_zyx), "rb") as pickle_file:
        content = pickle.load(pickle_file)
    image = np.asarray(content["IMG"], dtype=np.float32)
    time_min = np.asarray(content["IDIF_t"], dtype=float)
    return image, time_min


def save_network_input(data_folder, scan_id, image, time_min):
    """Write a network input file in the group's format (IMG and IDIF_t, both float64).

    The group's dataloader reads these files unchanged; only the data folder
    differs between the input representations.
    """
    path = network_input_path(data_folder, scan_id, image.shape[1:])
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {"IMG": np.asarray(image, dtype=np.float64), "IDIF_t": np.asarray(time_min, dtype=np.float64)}
    with open(path, "wb") as pickle_file:
        pickle.dump(content, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


def add_group_code_to_python_path(group_code_folder):
    """Make the group's modules (src/models, src/datahandlers) importable."""
    source_folder = str(Path(group_code_folder) / "src")
    if source_folder not in sys.path:
        sys.path.insert(0, source_folder)


def load_group_labels(group_code_folder):
    """Return {scan_id: experimental group} from the group's shared_dicts.py.

    The groups are used to stratify the validation split in training.
    """
    add_group_code_to_python_path(group_code_folder)
    from datahandlers.shared_dicts import groups

    group_of_scan = {}
    for group_name, scan_ids in groups.items():
        for scan_id in scan_ids:
            group_of_scan[scan_id] = group_name
    return group_of_scan
