# QickworkspaceV2 Tutorial Notebooks

These notebooks assume a live QICK session. Before running hardware cells,
edit the `BaseExperiment.connect_pyro4(...)` host and data path for your lab setup.

## Sequence

| Notebook | Topic |
| --- | --- |
| [00_quickstart.ipynb](00_quickstart.ipynb) | First experiment setup |
| [01_config_and_store.ipynb](01_config_and_store.ipynb) | `ExperimentConfig` and `CalibrationStore` |
| [02_running_experiments.ipynb](02_running_experiments.ipynb) | Resonator, qubit, Rabi, T1, Ramsey sequence |
| [03_batch_pipeline.ipynb](03_batch_pipeline.ipynb) | `BatchExperiment` and `ParallelExperiment` |
| [04_auto_calibrate.ipynb](04_auto_calibrate.ipynb) | `AutoCalibrate` pipeline |
| [05_custom_experiment.ipynb](05_custom_experiment.ipynb) | Custom `BaseProgram` and `BaseExperiment` |
| [06_real_hardware.ipynb](06_real_hardware.ipynb) | QICK hardware connection and saving |
| [07_data_management.ipynb](07_data_management.ipynb) | HDF5 and calibration-store inspection |

## Backend

Use the real hardware session:

```python
from QickworkspaceV2 import BaseExperiment

BaseExperiment.connect_pyro4("192.168.10.82")
```
