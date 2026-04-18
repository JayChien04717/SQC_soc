"""
Generate example HDF5 data files for testing the data browser.
Run with scqenv:
    python example/make_example_data.py
"""

import os
import json
import numpy as np
import h5py

OUT = os.path.dirname(__file__)


def write_h5(filename, experiment, qubit, tag, timestamp,
             x_vals, x_name, x_unit,
             iq, config=None,
             y_vals=None, y_name=None, y_unit=None):
    path = os.path.join(OUT, filename)
    avgi  = np.real(iq).astype(np.float32)
    avgq  = np.imag(iq).astype(np.float32)
    mag   = np.abs(iq).astype(np.float32)
    phase = np.degrees(np.arctan2(avgq, avgi)).astype(np.float32)

    with h5py.File(path, "w") as f:
        f.attrs["experiment"] = experiment
        f.attrs["qubit"]      = qubit
        f.attrs["timestamp"]  = timestamp
        f.attrs["tag"]        = tag
        f.attrs["config"]     = json.dumps(config or {}, indent=2)

        xg = f.create_group("x")
        xg.create_dataset("values", data=x_vals)
        xg.attrs["name"] = x_name
        xg.attrs["unit"] = x_unit

        if y_vals is not None:
            yg = f.create_group("y")
            yg.create_dataset("values", data=y_vals)
            yg.attrs["name"] = y_name
            yg.attrs["unit"] = y_unit

        dg = f.create_group("data")
        dg.create_dataset("avgi",  data=avgi)
        dg.create_dataset("avgq",  data=avgq)
        dg.create_dataset("mag",   data=mag)
        dg.create_dataset("phase", data=phase)

    print(f"  wrote {filename}")


# ── 1. Resonator Spec (1D Lorentzian) ────────────────────────────────────────
freq = np.linspace(6480, 6520, 201)          # MHz
f0, kappa = 6500.0, 3.0
lor = 1 / (1 + ((freq - f0) / (kappa / 2))**2)
noise = 0.02 * np.random.randn(len(freq))
iq = (lor + noise) + 1j * (0.1 * lor + 0.01 * np.random.randn(len(freq)))

write_h5("s002_res_spec_q2_001.h5",
         experiment="s002_onetone_ge", qubit=2, tag="OneTone",
         timestamp="2024-01-15T10:32:00",
         x_vals=freq * 1e6, x_name="Frequency", x_unit="Hz",
         iq=iq,
         config={"res_freq_ge": 6500.0, "reps": 1000, "py_avg": 10})

# ── 2. Qubit Spec (1D Lorentzian, narrower) ───────────────────────────────────
freq_q = np.linspace(5180, 5280, 201)
f_ge, kq = 5234.0, 1.5
lor_q = 1 / (1 + ((freq_q - f_ge) / (kq / 2))**2)
noise_q = 0.03 * np.random.randn(len(freq_q))
iq_q = (lor_q + noise_q) + 1j * 0.05 * np.random.randn(len(freq_q))

write_h5("s003_qubit_spec_q2_001.h5",
         experiment="s003_qubit_spec_ge", qubit=2, tag="TwoTone",
         timestamp="2024-01-15T10:45:00",
         x_vals=freq_q * 1e6, x_name="Frequency", x_unit="Hz",
         iq=iq_q,
         config={"f_ge": 5234.0, "reps": 1000, "py_avg": 10, "span": 100})

# ── 3. Time Rabi (1D oscillation) ────────────────────────────────────────────
t_rabi = np.linspace(0, 0.5, 101)           # µs
pi_time = 0.1
rabi_sig = np.cos(np.pi * t_rabi / pi_time) * np.exp(-t_rabi / 2.0)
noise_r  = 0.04 * np.random.randn(len(t_rabi))
iq_r = (rabi_sig + noise_r) + 1j * 0.02 * np.random.randn(len(t_rabi))

write_h5("s004_time_rabi_q2_001.h5",
         experiment="s004_time_rabi_ge", qubit=2, tag="TimeRabi",
         timestamp="2024-01-15T11:02:00",
         x_vals=t_rabi, x_name="Pulse Length", x_unit="us",
         iq=iq_r,
         config={"pi_time": pi_time, "reps": 1000, "py_avg": 20})

# ── 4. T1 decay ───────────────────────────────────────────────────────────────
t1_time = np.linspace(0, 200, 101)          # µs
T1 = 45.3
t1_sig = np.exp(-t1_time / T1)
noise_t1 = 0.03 * np.random.randn(len(t1_time))
iq_t1 = (t1_sig + noise_t1) + 1j * 0.01 * np.random.randn(len(t1_time))

write_h5("s008_T1_q2_001.h5",
         experiment="s008_T1_ge", qubit=2, tag="T1",
         timestamp="2024-01-15T11:20:00",
         x_vals=t1_time, x_name="Wait Time", x_unit="us",
         iq=iq_t1,
         config={"T1": T1, "reps": 2000, "py_avg": 50})

# ── 5. Ramsey (1D oscillation + decay) ───────────────────────────────────────
t2_time = np.linspace(0, 100, 101)          # µs
T2, f_det = 88.0, 0.2
ramsey_sig = np.cos(2 * np.pi * f_det * t2_time) * np.exp(-t2_time / T2)
noise_ram = 0.04 * np.random.randn(len(t2_time))
iq_ram = (ramsey_sig + noise_ram) + 1j * 0.02 * np.random.randn(len(t2_time))

write_h5("s006_Ramsey_q2_001.h5",
         experiment="s006_Ramsey_ge", qubit=2, tag="Ramsey",
         timestamp="2024-01-15T11:45:00",
         x_vals=t2_time, x_name="Wait Time", x_unit="us",
         iq=iq_ram,
         config={"T2": T2, "ramsey_freq": f_det, "reps": 1000, "py_avg": 30})

# ── 6. Resonator Punchout (2D: power × freq) ─────────────────────────────────
freq_2d = np.linspace(6490, 6510, 51)
power   = np.linspace(-40, 0, 21)           # dBm
F, P    = np.meshgrid(freq_2d, power)
f0_shift = 6500 + 0.5 * (P + 40) / 40      # frequency shifts with power
kappa_2d = 1.5 + 2.0 * (P + 40) / 40
lor_2d   = 1 / (1 + ((F - f0_shift) / (kappa_2d / 2))**2)
noise_2d = 0.03 * np.random.randn(*lor_2d.shape)
iq_2d    = (lor_2d + noise_2d).astype(complex)

write_h5("s002b_punchout_q2_001.h5",
         experiment="s002b_res_punchout_ge", qubit=2, tag="Punchout",
         timestamp="2024-01-15T14:10:00",
         x_vals=freq_2d * 1e6, x_name="Frequency", x_unit="Hz",
         y_vals=power,         y_name="Power",     y_unit="dBm",
         iq=iq_2d,
         config={"reps": 500, "py_avg": 5})

# ── 7. Qubit Flux Spec (2D: flux × freq) ──────────────────────────────────────
freq_fs = np.linspace(4800, 5600, 81)       # MHz
flux    = np.linspace(-0.5, 0.5, 41)        # mA
FQ, FL  = np.meshgrid(freq_fs, flux)
f_flux  = 5234 * np.sqrt(np.abs(np.cos(np.pi * FL / 0.5)))
lor_fs  = 1 / (1 + ((FQ - f_flux) / 1.0)**2)
noise_fs = 0.04 * np.random.randn(*lor_fs.shape)
iq_fs   = (lor_fs + noise_fs).astype(complex)

write_h5("s003a_flux_spec_q2_001.h5",
         experiment="s003a_qubit_flux_spec_ge", qubit=2, tag="FluxSpec",
         timestamp="2024-01-15T14:40:00",
         x_vals=freq_fs * 1e6, x_name="Frequency", x_unit="Hz",
         y_vals=flux,           y_name="Flux",      y_unit="mA",
         iq=iq_fs,
         config={"reps": 500, "py_avg": 5})

print("\nDone! Example files written to:", OUT)
