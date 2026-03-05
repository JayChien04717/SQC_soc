DATA_PATH = r"C:\Users\QEL\Desktop\2DQ12round3"

QubitIndex = 0

hw_cfg = {
    # DAC
    "res_ch": [0] * 6,  # Single Tone Readout Port, Full-speed DAC
    "qubit_ch": [5, 2, 2, 2, 2, 2],  # Qubit Channel Port, Full-speed DAC
    # "qubit_ch": [14, 7, 5, 11, 2, 2],  # Qubit Channel Port, Full-speed DAC
    "qubit_ch_ef": [2] * 6,  # Qubit ef Channel, Full-speed DAC
    "mux_ch": 12,
    "nqz_qubit": 2,
    "nqz_qubit_ef": 2,
    "nqz_res": 2,
    # ADC
    "ro_ch": [0] * 6,  # tproc configured readout channel
    "mux_ro_chs": [2, 3, 4, 5, 6, 7],
}


# Readout Configuration
readout_cfg = {
    "trig_time": 0.50,  # [Clock ticks] - get this value from TOF experiment
    "ro_length": 5.0,  # [us]
    "mixer_freq": 5600,  # [MHz] - used for mux_ch and interpolated_ch
    # Changes related to the resonator output channel
    "res_freq_ge": [5351, 7326, 7454.8, 5797.46, 5917.51, 6051.89],  # [MHzx]
    "res_gain_ge": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001],  # [DAC units]
    "res_freq_ef": [5000, 5100, 5200, 5300, 5400, 5500],  # [MHz]
    "res_gain_ef": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],  # [DAC units]
    "res_length": 5.0,  # [us] (1.0 for res spec)
    "res_phase": [0, 0, 0, 0, 0, 0],  # Rotation Angle From QICK Function
    "res_sigma": 0.004,
    # Threshold for Distinguish g/e, from QICK Function
    "threshold": [0, 0, 0, 0, 0, 0],
}

# Qubit Configuration
qubit_cfg = {
    # Freqs of Qubit g/e Transition
    "qubit_freq_ge": [4000, 4000, 4000, 4000, 4000, 4000],
    "qubit_gain_ge": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    "qubit_length_ge": 5,  # [us] for Constant Pulse
    "qubit_pi_gain_ge": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    "qubit_pi2_gain_ge": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    "qubit_ge_pulse_style": "arb",
    "qubit_flat_top_length_ge": 0.5,
    # [MHz] Freqs of Qubit e/f Transition
    "qubit_freq_ef": [4000, 4000, 4000, 4000, 4000, 4000],
    "qubit_gain_ef": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    "qubit_length_ef": 25.0,  # [us] for Constant Pulse
    "qubit_pi_gain_ef": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    "qubit_pi2_gain_ef": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    # digital mixer #[MHz]
    "qmixer_freq": [4000, 4000, 4000, 4000, 4000, 4000],
    "qmixer_freq_ef": [4000, 4000, 4000, 4000, 4000, 4000],
    "qubit_phase": 0,  # [deg]
    # [us] for Gaussian Pulse
    "sigma": [0.1 / 5, 0.1 / 5, 0.1 / 5, 0.1 / 5, 0.1 / 5, 0.1 / 5],
    "sigma_ef": [0.1 / 5, 0.1 / 5, 0.1 / 5, 0.1 / 5, 0.1 / 5, 0.1 / 5],
    "ramsey_freq": 2,  # [MHz]
}

cool_cfg = {
    "cooling": False,  # Enable Cooling Pulse
    "cool_ch1": 0,  # Cool Channel 1
    "cool_ch2": 1,  # Cool Channel 2
    "nqz_cool_ch1": 2,  # Number of Quantization Zones for Cool Channel 1
    "nqz_cool_ch2": 2,  # Number of Quantization Zones for Cool Channel 2
    "cool_mixer1": 4000,  # [MHz] Mixer Frequency for Cool Channel 1
    "cool_mixer2": 4000,  # [MHz] Mixer Frequency for Cool Channel 2
    "cool_length": 5.0,  # [us] Length of the Cool Pulse
    "cool_freq_1": 4000,  # [MHz] Frequency for Cool Channel 1
    "cool_freq_2": 4000,  # [MHz] Frequency for Cool Channel 2
    "cool_gain_1": 0.001,  # Gain for Cool Channel 1
    "cool_gain_2": 0.001,  # Gain for Cool Channel 2
}

expt_cfg = {
    "reps": 100,
    "soft_avgs": 2000,
    "relax_delay": 10,  # [us]
    "py_avg": 100,
}
