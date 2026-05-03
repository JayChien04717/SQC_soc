"""
generate.py — Pre-generate sample HDF5 data files for the tutorial.

Run once to populate tutorial/data/ with realistic synthetic results:

    cd SQC_soc
    python tutorial/data/generate.py

Creates one HDF5 file per experiment type plus a CalibrationStore JSON.
All data is synthetic (SimulatedBackend) — no hardware required.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from QickworkspaceV2 import SimulatedBackend, ExperimentConfig, CalibrationStore
from QickworkspaceV2.experiments.resonator import ResonatorSpec
from QickworkspaceV2.experiments.qubit_ge  import QubitSpec, PowerRabi
from QickworkspaceV2.experiments.coherence import T1, Ramsey, SpinEcho

OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
STORE_PATH = os.path.join(OUT_DIR, 'cal_store_generated.json')

config_list = [{
    "name": "Q1",
    "ch":  {"ro_ch": 0, "res_ch": 0, "qb_ch": 1},
    "res": {"res_freq_ge": 6700.0, "res_gain": 0.5, "res_length": 2.0},
    "qb":  {"qb_freq_ge": 5000.0, "pi_gain_ge": 0.5, "sigma": 0.025},
    "reps": 500, "relax_delay": 300.0, "steps": 101,
    "kappa": 4.0, "qb_kappa": 3.0, "pi_gain_true": 0.48,
    "wait_time_start": 0.0, "wait_time_stop": 200.0,
    "T1_true": 50.0, "T2_true": 22.0, "ramsey_freq": 2.0,
}]

EXPERIMENTS = [
    ("res_spec",    ResonatorSpec, "res_freq_ge"),
    ("qubit_spec",  QubitSpec,     "qb_freq_ge"),
    ("power_rabi",  PowerRabi,     "pi_gain_ge"),
    ("t1",          T1,            "T1_us"),
    ("ramsey",      Ramsey,        "T2r_us"),
    ("spin_echo",   SpinEcho,      "T2e_us"),
]


def main():
    backend = SimulatedBackend(noise_level=0.015)
    backend.activate()

    cfg_all = ExperimentConfig(config_list)
    store   = CalibrationStore(STORE_PATH)

    print(f"Generating sample data in: {OUT_DIR}\n")

    for name, ExptClass, store_key in EXPERIMENTS:
        cfg = cfg_all.get_qubit("Q1")
        print(f"  Running {name} ...", end="", flush=True)

        result = ExptClass(cfg, backend=backend).run(py_avg=10)

        # Save HDF5
        h5_path = os.path.join(OUT_DIR, f"{name}.h5")
        result.save(h5_path)

        # Persist scalar in CalibrationStore
        if result.scalar_result is not None:
            store.set("Q1", store_key, result.scalar_result)
            # Feed back calibrated values so later experiments benefit
            if store_key in cfg_all.get_qubit("Q1"):
                cfg_all.update(store_key, result.scalar_result, q_index="Q1")

        status = result.quality.value
        val    = f"{result.scalar_result:.4f}" if result.scalar_result else "N/A"
        print(f" {status:<10} scalar={val}  → {os.path.basename(h5_path)}")

    print(f"\nCalibrationStore → {STORE_PATH}")
    print(store.summary("Q1"))
    print("\nDone. Open tutorial/07_data_management.ipynb to explore the files.")


if __name__ == "__main__":
    main()
