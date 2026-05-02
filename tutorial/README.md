# reconstruct — Tutorial Notebooks

All notebooks run **offline** using `SimulatedBackend` (no QICK hardware needed),
except sections explicitly marked **HARDWARE ONLY** in notebook 06.

## Sequence

| Notebook | Topic |
| --- | --- |
| [00_quickstart.ipynb](00_quickstart.ipynb) | First experiment in 5 minutes |
| [01_config_and_store.ipynb](01_config_and_store.ipynb) | `ExperimentConfig` + `CalibrationStore` |
| [02_running_experiments.ipynb](02_running_experiments.ipynb) | Standard sequence: ResonatorSpec → QubitSpec → PowerRabi → T1 → Ramsey |
| [03_batch_pipeline.ipynb](03_batch_pipeline.ipynb) | `BatchExperiment` and `ParallelExperiment` |
| [04_auto_calibrate.ipynb](04_auto_calibrate.ipynb) | `AutoCalibrate` — fully automated 7-step pipeline |
| [05_custom_experiment.ipynb](05_custom_experiment.ipynb) | Writing your own `BaseProgram` + `BaseExperiment` |
| [06_real_hardware.ipynb](06_real_hardware.ipynb) | Real hardware connection, HDF5 saving, REST service |
| [07_data_management.ipynb](07_data_management.ipynb) | List / load / compare HDF5 files, inspect CalibrationStore |

## Data folder

```text
tutorial/data/
├── generate.py              ← run once to create sample .h5 files
├── system_cfg_example.py    ← hardware config template (copy to reconstruct/config/system_cfg.py)
├── cal_store_Q1.json        ← example CalibrationStore with two qubits pre-populated
├── cal_store_generated.json ← written by generate.py
└── *.h5                     ← one file per experiment type (written by generate.py)
```

Generate sample data before opening notebook 07:

```bash
cd SQC_soc
python tutorial/data/generate.py
```

## Setup

```bash
cd SQC_soc
pip install numpy scipy matplotlib h5py scikit-learn fastapi uvicorn
jupyter notebook tutorial/
```

The notebooks add `../` to `sys.path` automatically, so no install step is required
for `reconstruct` itself.
