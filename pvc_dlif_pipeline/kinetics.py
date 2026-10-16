"""Kinetic models fitted with pyPET, the group's reference implementation.

pyPET (S. Kuttner, https://github.com/Kuttner/pyPET) is GPL-3 and this
project is MIT, so pyPET is not copied in: it is loaded from a local clone
whose folder is given in the configuration (paths.pypet_folder).

The functions here only adapt inputs and outputs:
- time goes to pyPET in seconds (its resampling step TIME_FRAMES is in the
  units of the time vector, so 2.5 means 2.5 s);
- pyPET gets copies of the curves, because it changes small values in place;
- a fit that raises an error gives NaN instead of stopping the whole run.
"""

import importlib.util
from functools import lru_cache
from pathlib import Path

import numpy as np

# pyPET's own defaults: uniform resampling every 2.5 s, parameter bounds (0, 10).
FRAME_SECONDS = 2.5
UPPER_BOUND = 10.0


@lru_cache(maxsize=None)
def load_pypet(pypet_folder):
    """Import Tissue_compartment_modeling.py from the pyPET clone (once per process)."""
    path = Path(pypet_folder) / "Tissue_compartment_modeling.py"
    if not path.exists():
        raise FileNotFoundError(f"pyPET not found: {path}. Clone https://github.com/Kuttner/pyPET "
                                "and set paths.pypet_folder in the configuration.")
    specification = importlib.util.spec_from_file_location("pypet_tissue_models", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def whole_blood_to_plasma(curve, time_min, a, b, c):
    """Multiply a whole-blood curve by the plasma-to-whole-blood ratio R(t) = a exp(-b t) + c."""
    curve = np.asarray(curve, dtype=np.float64)
    time_min = np.asarray(time_min, dtype=np.float64)[:curve.size]
    return curve[:time_min.size] * (a * np.exp(-b * time_min) + c)


def curves_for_pypet(tissue, plasma, time_min):
    """Copies of tissue and plasma curves of equal length, and the time in seconds."""
    n = min(len(tissue), len(plasma), len(time_min))
    tissue = np.array(tissue[:n], dtype=np.float64)
    plasma = np.array(plasma[:n], dtype=np.float64)
    time_s = np.array(time_min[:n], dtype=np.float64) * 60.0
    return tissue, plasma, time_s


def patlak_ki(pypet, tissue, plasma, time_min):
    """Patlak Ki (1/min) with pyPET; the break point is fitted per curve. NaN if the fit fails."""
    tissue, plasma, time_s = curves_for_pypet(tissue, plasma, time_min)
    try:
        with np.errstate(all="ignore"):
            ki, _intercept = pypet.patlak(plasma, time_s, tissue, time_s.copy(),
                                          INTERPOLATE=True, TIME_FRAMES=FRAME_SECONDS)
    except Exception:
        return float("nan")
    return float(ki)


def two_tissue_parameter_names(irreversible):
    """Names of the reported two-tissue parameters."""
    if irreversible:
        return ["k1", "k2", "k3", "vb", "ki"]
    return ["k1", "k2", "k3", "k4", "vb", "ki"]


def fit_two_tissue(pypet, tissue, plasma, time_min, irreversible):
    """Two-tissue compartment model with fitted blood volume, with pyPET.

    Returns a dict with the parameters, ki = K1 k3 / (k2 + k3) as pyPET
    reports it, converged (False if the fit raised an error), bounds_hit (a
    parameter ended at pyPET's upper bound, so the bound and not the data set
    it) and r_squared of the fitted tissue curve.
    """
    tissue, plasma, time_s = curves_for_pypet(tissue, plasma, time_min)
    failed = {name: float("nan") for name in two_tissue_parameter_names(irreversible)}
    failed.update(converged=False, bounds_hit=False, r_squared=float("nan"))
    if tissue.size < 6:
        return failed
    model = pypet.twoTCMirrev if irreversible else pypet.twoTCMrev
    try:
        with np.errstate(all="ignore"):
            parameters, ki, fitted_tissue, _ = model(plasma, time_s, tissue, time_s.copy(),
                                                     INTERPOLATE=True, TIME_FRAMES=FRAME_SECONDS)
    except Exception:
        return failed

    # pyPET's parameter order: k1, k2, vb, k3 (irreversible) and k1, k2, vb, k3, k4 (reversible).
    parameters = np.asarray(parameters, dtype=np.float64)
    result = {"k1": float(parameters[0]), "k2": float(parameters[1]), "k3": float(parameters[3])}
    if not irreversible:
        result["k4"] = float(parameters[4])
    result["vb"] = float(parameters[2])
    result["ki"] = float(ki)
    result["converged"] = True
    result["bounds_hit"] = bool(np.any(parameters >= UPPER_BOUND * (1 - 1e-3)))
    residual = np.asarray(fitted_tissue, dtype=np.float64) - tissue
    total = float(np.sum((tissue - tissue.mean()) ** 2))
    result["r_squared"] = float(1 - np.sum(residual ** 2) / total) if total > 1e-12 else float("nan")
    return result


def relative_error(value, reference):
    """(value - reference) / reference; NaN if either is not finite or the reference is about zero."""
    if not np.isfinite(value) or not np.isfinite(reference) or abs(reference) < 1e-12:
        return float("nan")
    return float((value - reference) / reference)


def fit_scan(task):
    """All kinetic fits of one scan: once with the measured input, then once per predicted curve.

    task: dict with scan_id, pypet_folder, irreversible, plasma_ratio (a, b, c),
    time_min (tissue time axis), tissue ({voi: curve}), measured (input curve)
    and curves (list of dicts with the curve keys and 'predicted').
    Returns one row per curve, VOI, model and parameter.
    """
    pypet = load_pypet(task["pypet_folder"])
    time_min = task["time_min"]
    a, b, c = task["plasma_ratio"]
    irreversible = task["irreversible"]
    measured_plasma = whole_blood_to_plasma(task["measured"], time_min, a, b, c)

    reference_fits = {}
    for voi, tissue in task["tissue"].items():
        reference_fits[voi] = (patlak_ki(pypet, tissue, measured_plasma, time_min),
                               fit_two_tissue(pypet, tissue, measured_plasma, time_min, irreversible))

    rows = []
    for curve in task["curves"]:
        keys = {name: value for name, value in curve.items() if name != "predicted"}
        predicted_plasma = whole_blood_to_plasma(curve["predicted"], time_min, a, b, c)
        for voi, tissue in task["tissue"].items():
            reference_ki, reference_two_tissue = reference_fits[voi]
            ki = patlak_ki(pypet, tissue, predicted_plasma, time_min)
            rows.append({**keys, "voi": voi, "model": "patlak", "parameter": "ki",
                         "value_predicted": ki, "value_measured": reference_ki,
                         "relative_error": relative_error(ki, reference_ki)})
            two_tissue = fit_two_tissue(pypet, tissue, predicted_plasma, time_min, irreversible)
            for name in two_tissue_parameter_names(irreversible):
                rows.append({**keys, "voi": voi, "model": "two_tissue", "parameter": name,
                             "value_predicted": two_tissue[name],
                             "value_measured": reference_two_tissue[name],
                             "relative_error": relative_error(two_tissue[name], reference_two_tissue[name]),
                             "r_squared_predicted": two_tissue["r_squared"],
                             "r_squared_measured": reference_two_tissue["r_squared"],
                             "converged_predicted": two_tissue["converged"],
                             "converged_measured": reference_two_tissue["converged"],
                             "bounds_hit_predicted": two_tissue["bounds_hit"],
                             "bounds_hit_measured": reference_two_tissue["bounds_hit"]})
    return rows
