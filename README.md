# pvc-dlif-pipeline

Code for the master thesis *Partial volume correction for deep learning-derived
input functions in preclinical FDG PET* (Tor-Ivar Hassfjord, FYS-3941, UiT The
Arctic University of Norway, 2026).

The thesis asks whether partial volume correction (PVC) of dynamic mouse FDG
PET images improves the input function predicted by a deep learning network
(DLIF), and what it costs when the images a network is shown were corrected
differently from the images it was trained on. Two networks are studied:
FC-DLIF (DLIFNet_MAX) and DLIF2024 (CNN3_1dConv3_linear). Motion correction is
studied the same way, alone and combined with PVC in both orders.

The pipeline goes from the scanner's DICOM exports to every number and figure
in the thesis, in ten stages. Each stage is one script that reads the output of
the earlier stages and writes its own; nothing is kept in memory between
stages, so every stage can be checked on its own.

## What you need

The data and the code dependencies below are not part of this repository.

| What | Where it comes from | Used in |
|---|---|---|
| DICOM exports of the 70 FDG scans | UiT ML4PET group (not public) | 00, 01 |
| The group's preprocessed data (`AIF_SUV/`, `IMG_SUV_96x48x48/`, `VOI_SUV/`) | UiT ML4PET group (not public) | 00, 01, 06, 08, 09 |
| The group's DLIF repository (network definitions, group labels) | UiT ML4PET group (private) | 05, 06 |
| pyPET (kinetic models) | https://github.com/Kuttner/pyPET, GPL-3 | 08 |
| PETPVC | conda-forge (`petpvc`) | 02, 03 |
| FALCON (motion correction) | PyPI (`falconz`) | 03 |
| KinePET (compartment model fitter, C. Salomonsen) | Not public; available on request from C. Salomonsen | KinePET analyses only |
| A CUDA GPU | | 05 (training); 06 runs on a CPU too |

pyPET is GPL-3 and this repository is MIT, so pyPET is loaded at run time
from its own folder (`paths.pypet_folder`) and never copied in.

## Installation

```
conda env create -f environment.yml
conda activate pvc-dlif-pipeline
uv pip install -r requirements-lock.txt --index-strategy unsafe-best-match
uv pip install -e .
```

`requirements-lock.txt` pins every Python package of the environment the
pipeline was checked in, including torch with CUDA 12.4 and falconz. The index
option lets uv take torch from the PyTorch index and everything else from PyPI. Without it, `uv pip install -e ".[torch]" falconz` installs
current versions instead.

NumPy is installed from PyPI (OpenBLAS), not from conda (MKL), which avoids a
clash between two OpenMP runtimes when PyTorch and NumPy are loaded together.

## Configuration

Everything the stages do is set in `configs/study.yaml`: the excluded scans,
the PVC settings, the motion correction, the training protocol, the conditions
and comparison families, and the statistics. Copy it to `configs/local.yaml`
(ignored by git) and set the five paths at the top. Leave the rest as it is
to reproduce the thesis.

## Stages

Run from the repository folder, in this order:

| Stage | Script | Writes (in the work folder) |
|---|---|---|
| 00 | `00_inventory_scans.py` | `manifest.csv`, `frame_schedule.json`: which scans are used |
| 01 | `01_convert_and_find_crop.py` | `native/`: NIfTI series; `crop_windows.csv`: where the network input is cut out |
| 02 | `02_apply_pvc.py` | `pvc/<correction>/`: corrected series and per-frame diagnostics |
| 03 | `03_motion_correction.py` | `motion/`: motion-corrected series, transforms, QC |
| 04 | `04_make_network_inputs.py` | `network_inputs/<representation>/`: network inputs in the group's format |
| 05 | `05_train.py` | `models/<network>/<representation>/fold_XX/run_YY/` |
| 06 | `06_predict.py` | `predictions/<network>.csv`, `<network>_ensemble.csv` |
| 07 | `07_compare_conditions.py` | `results/<network>/<estimand>/`: curve metrics and paired comparisons |
| 08 | `08_kinetics.py` | `results/<network>/<estimand>/`: Patlak and two-tissue fits, Ki comparisons |
| 09 | `09_figures.py` | `figures/`: the thesis figures; with `--overview` also `figures/overview/`, every plot type for every comparison family, metric and correction |

Every script takes `--config configs/local.yaml`; the docstring at the top of
each script says what it reads, what it writes and how to call it. Stages 01,
02, 04 and 05 skip outputs that already exist, so they can be stopped and
started again; stage 03 does not, and is run per scan with `--scan-ids`.

Training (stage 05) is 10 folds x 10 runs for each of 7 input representations
and 2 networks, 1000 epochs per run. `cluster/` has the scripts used to run
it as Kubernetes jobs on a GPU cluster.

## Analyses

`analyses/` holds the further analyses the thesis reports. Each answers one
question, reads only the outputs of the stages, and writes to
`<work>/analyses/<network>/`; run them after stage 08.

| Script | Question |
|---|---|
| `bias_variance.py` | Is the error bias (the mean prediction is off) or variance (the runs disagree)? |
| `training_convergence.py` | When does each run leave the loss plateau, and does the input change that? |
| `best_of_10.py` | Do the conclusions hold for the run one would deploy (lowest validation loss per fold)? |
| `run_subsampling.py` | How many runs per fold does a paired comparison need? |
| `mismatch_scaling.py` | How much of the train-test mismatch is a single scale factor? |
| `dead_units.py` | Are the runs that never converged dead inside (ReLU channels never active)? |
| `reproduce_paper_2026.py` | Does the retrained FC-DLIF reach the level published for it? |
| `summary_table.py` | Single run, ensemble and best of 10 side by side, for RMSE and Patlak Ki |

`dead_units.py` needs the trained weights and the group's code; the others
read only result files. `summary_table.py` runs last.

### KinePET analyses (exploratory)

These reproduce the voxelwise kinetics and the simulation with known parameters
in the thesis (Sections *Cross-Check with a Second Fitting Tool* and *Voxelwise
Kinetics and a Simulation with Known Parameters*). They were defined after the
main results had been seen and are exploratory. KinePET is not part of this
repository and is run in its own environment, which needs `kinepet`, `numpy`,
`scipy`, `nibabel`, `matplotlib` and `pyyaml`; nothing from KinePET is copied
in. Paths and the point spread function are read from `--config` (default
`configs/local.yaml`), and everything is written to `<work>/analyses/kinepet/`.
Run them after stage 08; the PVC step of the simulation runs in this
repository's environment, with PETPVC exactly as in stage 02.

| Script | Environment | Question | In the thesis |
|---|---|---|---|
| `kinepet_vs_pypet_measured.py` | KinePET | Do KinePET and pyPET give the same compartment parameters with the measured input? | Results, KinePET cross-check (text) |
| `kinepet_vs_pypet_predicted.py` | KinePET | Do they give the same $K_i$ error for the predicted inputs? | Table of the KinePET cross-check |
| `kinepet_voxelwise_test.py` | KinePET | How does voxelwise PVC change voxelwise fits on the measured scans? | Table of voxelwise fits on the measured scans |
| `kinepet_simulation_pvc.py --step make` | KinePET | Build the truth (known parameters) and a blurred, noisy measurement of it | Simulation |
| `simulation_apply_petpvc.py` | this repository | Correct the simulated measurement with PETPVC (RL and RVC, k = 3 and 15) | Simulation |
| `kinepet_simulation_pvc.py --step evaluate` | KinePET | Fit the corrected series and compare $K_i$ with the truth | Simulation |
| `simulation_summary.py` | KinePET | Collect all simulation runs into one table and figure | Simulation table and figure |

The simulation, for one scan (the thesis uses AA1, AB3, AV1 and M3; the
controls `--noise 0`, `--truth-image native` and `--seed 2`/`3` on AA1 go on
both the `make` and the `evaluate` call):

```
# KinePET environment, from the repository folder
python analyses/kinepet_simulation_pvc.py --step make --scan AA1 --config configs/local.yaml
# this repository's environment
python analyses/simulation_apply_petpvc.py --folder <work>/analyses/kinepet/simulation/AA1_rl_i15_noise1
# KinePET environment
python analyses/kinepet_simulation_pvc.py --step evaluate --scan AA1 --config configs/local.yaml
python analyses/simulation_summary.py --config configs/local.yaml
```

The noise of the simulation is drawn from a fixed seed, so a rerun reproduces
the thesis numbers exactly on the same platform.

### Training against Ki (exploratory)

These reproduce Appendix B of the thesis: FC-DLIF trained with the tissue
curves (VOI TACs) of the same scan, and a ridge regression from the tissue
curves alone. They were defined after the main results and are exploratory.
Settings are in the `kiloss` section of the configuration; everything else is
the thesis training protocol (folds, run seeds, validation split, epochs).
The models are written to `<work>/models_kiloss/<arm>/` and the tables to
`<work>/analyses/kiloss/`. Run them after stage 08.

| Script | Environment | Question | In the thesis |
|---|---|---|---|
| `kiloss_train.py --arm control/hybrid/aux/kiloss` | this repository, GPU | Train FC-DLIF unchanged (control), with the curves as input (hybrid), with a curve head (aux) or with a Patlak $K_i$ term in the loss (kiloss) | Appendix B |
| `kiloss_ridge_curves.py` | this repository | How well does a ridge regression predict the input function from the four tissue curves alone? | Appendix B (text) |
| `kiloss_ridge_ki.py` | this repository, pyPET | What $K_i$ error does that ridge input give? | Appendix B (text) |
| `kiloss_evaluate.py` | this repository, pyPET | RMSE, peak and $K_i$ error of each arm against control | Appendix B table |
| `kiloss_check.py` | KinePET | Does the $K_i$ gain of kiloss hold with the 2TCM fitted by KinePET? | Appendix B table (2TCM row) |

```
# smoke test (writes to <work>/models_kiloss_smoketest/)
python analyses/kiloss_train.py --config configs/local.yaml --arm kiloss --folds 1 --runs 1 --epochs 2
# the thesis: control and kiloss 10 folds x 10 runs, hybrid and aux --runs 1 2
python analyses/kiloss_train.py --config configs/local.yaml --arm control
python analyses/kiloss_train.py --config configs/local.yaml --arm kiloss
python analyses/kiloss_ridge_curves.py --config configs/local.yaml
python analyses/kiloss_ridge_ki.py --config configs/local.yaml
python analyses/kiloss_evaluate.py --config configs/local.yaml
# KinePET environment
python analyses/kiloss_check.py --config configs/local.yaml
```

The arms in the thesis were trained on an RTX 3090, one arm per job.

## Reproducibility

The pipeline was checked stage by stage against the original research code
that produced the thesis results.

- Stages 00, 01, 02 and 04 give the same output as the original, bitwise for
  the NIfTI series, the corrected series and the network inputs that were
  compared.
- Stages 06, 07 and 08, run on the trained networks of the thesis, reproduce
  the original predictions (to 4e-15), comparisons (to 3e-11) and Ki
  comparisons (to 1e-13); permutation p-values are identical.

Two stages cannot be repeated exactly, by anyone:

- **Motion correction (03).** FALCON (falconz 2.2.3; its log prints
  "V.2.0.0") does not give the same result twice on the same input. Two runs
  differ by a median of about 0.17 mm in translation, about half of the
  applied motion. The networks trained on motion-corrected
  input were trained on one particular FALCON run.
- **Training (05).** Training is seeded (`training.seed` and the run number
  fix the data split, the batch order, the augmentation and the initial
  weights) but not deterministic: GPU kernels and mixed precision differ in
  the last bits from run to run, and over 1000 epochs this gives a different
  network. A rerun is a new draw from the same distribution of networks, not
  a copy of the thesis networks. The networks of the thesis were trained with
  the original code, which built each network before seeding; this
  repository seeds first, so that the seed also fixes the initial weights.

One result is sensitive at the level of rounding errors: the two-tissue fit of
the myocardium is poorly determined (k2 runs to zero or k3 to the bound in a
few per cent of fits), and a change in the last digits of the input can move
such a fit to a different solution. Patlak Ki, the tested kinetic endpoint, is
not affected.

## Tests

```
uv pip install pytest
python -m pytest tests
```

The tests use small synthetic inputs with known answers (metrics, statistics,
crop window, plasma conversion, folds, configuration); they need no data.

## Repository layout

```
configs/study.yaml        the study protocol and paths
pvc_dlif_pipeline/        functions used by the stages, one module per topic
scripts/                  the stages, 00 to 09
analyses/                 further analyses reported in the thesis
tests/                    tests on synthetic inputs
cluster/                  running stage 05 on a Kubernetes GPU cluster
environment.yml           conda environment (Python, PETPVC, uv)
requirements-lock.txt     exact Python package versions
```

## License

MIT, see `LICENSE`. pyPET, PETPVC, FALCON and the group's code have their own
licenses and are not included.
