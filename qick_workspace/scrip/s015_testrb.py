"""
s015_testrb — Debug RB with legacy gate generation
====================================================
Uses the original clifford_1q matrix / gate_sequence / expand_full_sequence
logic for RB gate generation, but wraps the hardware side with BaseProgram
and the session pattern from BaseExperiment.

Usage
-----
    from scrip.s015_testrb import TestRB

    rb = TestRB(cfg)
    rb.run(py_avg=10, max_circuit_depth=500, delta_clifford=50, number_sample=30)
    rb.plot()
    rb.saveLabber(qb_idx=0)
"""

import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from .base_program import BaseProgram
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import hdf5_generator, get_next_filename_labber, config_to_yaml
from ..tools.fitting import fitrb, rb_func, rb_error, error_fit_err


# ======================================================= #
#  Legacy Clifford gate generation (original code)        #
# ======================================================= #

clifford_1q = dict()

clifford_1q["X"] = np.matrix(
    [
        [0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0],
    ]
)
clifford_1q["Y"] = np.matrix(
    [
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
    ]
)
clifford_1q["X/2"] = np.matrix(
    [
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
    ]
)
clifford_1q["Y/2"] = np.matrix(
    [
        [0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1],
    ]
)
clifford_1q["-X/2"] = np.matrix(
    [
        [0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0],
    ]
)
clifford_1q["-Y/2"] = np.matrix(
    [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
    ]
)
identity = np.diag([1] * 6)
clifford_1q["I"] = identity

step_pulses = [
    ("Y/2", "X"),
    ("Y/2", "X/2"),
    ("X/2", "-Y/2", "-X/2"),
    ("-X/2", "-Y/2"),
    ("Y/2", "-X/2"),
    ("-X/2", "Y/2", "-X/2"),
    ("X/2", "Y/2"),
    ("-Y/2", "X"),
    ("-X/2", "Y"),
    ("-Y/2", "-X/2"),
    ("X/2", "Y/2", "X/2"),
    ("-X/2", "Y/2"),
    ("X", "Y"),
    ("X/2", "Y"),
    ("-Y/2", "X/2"),
    ("X/2", "Y/2", "-X/2"),
    ("X/2", "-Y/2"),
]

for pulse in step_pulses:
    new_mat = clifford_1q[pulse[0]]
    for p in pulse[1:]:
        new_mat = new_mat @ clifford_1q[p]
    repeat = False
    for existing_pulse_name, existing_pulse in clifford_1q.items():
        if np.array_equal(new_mat, existing_pulse):
            repeat = True
    if not repeat:
        clifford_1q[pulse[0] + "," + ",".join(pulse[1:])] = new_mat

clifford_1q_names = list(clifford_1q.keys())
assert len(clifford_1q_names) == 24, (
    f"Clifford group has {len(clifford_1q_names)} elements instead of 24!"
)

for name, matrix in clifford_1q.items():
    z_new = np.argmax(matrix[:, 0])
    x_new = np.argmax(matrix[:, 1])
    clifford_1q[name] = (matrix, (z_new, x_new))


def gate_sequence(rb_depth, pulse_n_seq=None, debug=False):
    """
    Generate a random Clifford gate sequence with its inverting gate.

    Parameters
    ----------
    rb_depth : int
        Number of random Clifford gates in the sequence.
    pulse_n_seq : array-like of int or None, optional
        Explicit sequence of Clifford indices.  If ``None``, a random
        sequence is generated.
    debug : bool, optional
        Print the gate sequence and its inverse when ``True``.

    Returns
    -------
    pulse_name_seq : list of str
        List of Clifford gate names (may be composite, e.g. ``"X/2,Y"``).
    total_clifford : str
        Name of the inverting Clifford gate.
    """
    if pulse_n_seq is None:
        pulse_n_seq = (len(clifford_1q_names) * np.random.rand(rb_depth)).astype(int)
    pulse_name_seq = [clifford_1q_names[n] for n in pulse_n_seq]
    psi_nz = np.matrix([[1, 0, 0, 0, 0, 0]]).transpose()
    psi_nx = np.matrix([[0, 1, 0, 0, 0, 0]]).transpose()
    for n in pulse_n_seq:
        gates = clifford_1q_names[n].split(",")
        for gate in reversed(gates):
            psi_nz = clifford_1q[gate][0] @ psi_nz
            psi_nx = clifford_1q[gate][0] @ psi_nx
    psi_nz = psi_nz.flatten()
    psi_nx = psi_nx.flatten()

    total_clifford = None
    if np.argmax(psi_nz) == 0:
        total_clifford = "I"
    else:
        for clifford in clifford_1q_names:
            if clifford_1q[clifford][1] == (np.argmax(psi_nz), np.argmax(psi_nx)):
                total_clifford = clifford
                break
    assert total_clifford is not None, (
        f"Failed to invert gate sequence! {pulse_name_seq}"
    )
    if debug:
        print("pulse seq:", pulse_name_seq, "| inverse:", total_clifford)
    return pulse_name_seq, total_clifford


def interleaved_gate_sequence(rb_depth, gate_char: str, debug=False):
    """
    Generate an interleaved Clifford gate sequence for IRB.

    Interleaves ``gate_char`` after every random Clifford and returns the
    combined sequence with its inverting gate.

    Parameters
    ----------
    rb_depth : int
        Number of random Clifford gates (interleaved gate doubles the depth).
    gate_char : str
        Gate name to interleave (must be in ``clifford_1q_names``).
    debug : bool, optional
        Forwarded to ``gate_sequence`` for debug printing.

    Returns
    -------
    pulse_name_seq : list of str
        Combined (interleaved) Clifford gate name sequence.
    total_clifford : str
        Name of the inverting Clifford gate.
    """
    pulse_n_seq_rand = (len(clifford_1q_names) * np.random.rand(rb_depth)).astype(int)
    pulse_n_seq = []
    assert gate_char in clifford_1q_names
    n_gate_char = clifford_1q_names.index(gate_char)
    for n_rand in pulse_n_seq_rand:
        pulse_n_seq.append(n_rand)
        pulse_n_seq.append(n_gate_char)
    return gate_sequence(len(pulse_n_seq), pulse_n_seq=pulse_n_seq, debug=debug)


def expand_full_sequence(pulse_name_seq, total_clifford):
    """
    Expand composite Clifford names to a flat list of primitive gate strings.

    Parameters
    ----------
    pulse_name_seq : list of str
        Clifford gate names, possibly composite (e.g. ``"X/2,Y"``).
    total_clifford : str
        Name of the inverting Clifford gate.

    Returns
    -------
    final_sequence : list of str
        Flat list of primitive gate names (e.g. ``["X/2", "Y", "-X/2", ...]``).
    """
    full_sequence = []
    for name in pulse_name_seq:
        gates = name.split(",")
        for g in reversed(gates):
            full_sequence.append(g)

    for gate in total_clifford.split(","):
        neg = "-" in gate
        neg = not neg
        if neg:
            if "-" not in gate:
                gate = "-" + gate
        else:
            if "-" in gate:
                gate = gate.replace("-", "")
        full_sequence.append(gate)

    final_sequence = []
    for item in full_sequence:
        if "," in item:
            final_sequence.extend(item.split(","))
        else:
            final_sequence.append(item)
    return final_sequence


# Map legacy gate names → BaseProgram pulse names
# "-X" and "-Y" are self-inverse (X180 is its own inverse), map to same pulse
_GATE_TO_PULSE = {
    "X":    "x180_ge",
    "-X":   "x180_ge",
    "Y":    "y180_ge",
    "-Y":   "y180_ge",
    "X/2":  "x90_ge",
    "-X/2": "x90m_ge",
    "Y/2":  "y90_ge",
    "-Y/2": "y90m_ge",
}


# ======================================================= #
#  QICK Program (BaseProgram)                             #
# ======================================================= #

class TestRBProgram(BaseProgram):
    """QICK program for TestRB: applies a legacy gate sequence then measures."""

    def _initialize(self, cfg):
        """Set up resonator, qubit generator, and standard ge gates."""
        self.setup_resonator(cfg, prefix="ge")
        self.setup_qubit_gen(cfg, prefix="ge")
        self.setup_standard_gates(cfg, prefix="ge")
        if cfg.get("cooling", False):
            self.apply_cool(cfg)

    def _body(self, cfg):
        """Apply optional cooling, unrolled legacy gate sequence, then measure."""
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.cooling_body(cfg)

        for gate in cfg["gate_seq"]:
            if gate in ("I", "-I"):
                self.delay_auto(cfg[f"sigma_ge"] * 5)
            else:
                pulse_name = _GATE_TO_PULSE.get(gate)
                if pulse_name is None:
                    raise ValueError(f"Unknown gate '{gate}' in gate_seq")
                self.pulse(ch=cfg["qb_ch"], name=pulse_name, t=0)
                self.delay_auto(0.01)

        self.delay_auto(0.05)
        self.measure(cfg)


# ======================================================= #
#  Experiment class (session pattern)                     #
# ======================================================= #

class TestRB:
    """
    Debug RB experiment using legacy Clifford gate generation.

    Instantiate with config only after ``BaseExperiment.setup()`` has been called.
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Experiment configuration dictionary.

        Raises
        ------
        RuntimeError
            If ``BaseExperiment.setup()`` has not been called first.
        """
        from .base_experiment import BaseExperiment
        if BaseExperiment._soc is None:
            raise RuntimeError("Call BaseExperiment.setup(soc, soccfg, data_path) first.")
        self.soc = BaseExperiment._soc
        self.soccfg = BaseExperiment._soccfg
        self.cfg = config
        self.x = None
        self.rb_result = None
        self._number_sample = None
        self._interleaved = None
        self._iq_process = "abs"

    def run(
        self,
        py_avg,
        max_circuit_depth,
        delta_clifford,
        number_sample,
        interleaved_gate=None,
        randomize_depth_order=False,
        iq_process="abs",
    ):
        """
        Acquire Standard RB or IRB data using the legacy gate generator.

        Parameters
        ----------
        py_avg : int
            Software averages.
        max_circuit_depth : int
            Maximum Clifford depth (exclusive).
        delta_clifford : int
            Step size between depths.
        number_sample : int
            Random sequences per depth point.
        interleaved_gate : str or None, optional
            Gate name for IRB (must be in ``clifford_1q_names``), or ``None``
            for standard RB.
        randomize_depth_order : bool, optional
            Measure depths in random order to average out time drift.
        iq_process : str, optional
            IQ processing mode: ``"abs"`` or ``"real"``.  Use ``"real"``
            after readout optimisation.

        Raises
        ------
        ValueError
            If ``interleaved_gate`` is not in ``clifford_1q_names``.
        """
        self._iq_process = iq_process
        self._number_sample = number_sample
        self._interleaved = interleaved_gate

        self.x = np.arange(1, max_circuit_depth, delta_clifford)

        if interleaved_gate is not None and interleaved_gate not in clifford_1q_names:
            raise ValueError(
                f"interleaved_gate '{interleaved_gate}' not in clifford_1q_names.\n"
                f"Valid gates: {clifford_1q_names}"
            )

        desc = f"IRB ({interleaved_gate})" if interleaved_gate else "Standard RB"

        # Depth measurement order
        depth_indices = np.arange(len(self.x))
        if randomize_depth_order:
            np.random.shuffle(depth_indices)
            print(f"  Depth order: {self.x[depth_indices].tolist()}")

        rb_result = [None] * len(self.x)
        for idx in tqdm(depth_indices, desc=desc):
            depth = self.x[idx]
            rblist = []
            for _ in tqdm(range(number_sample), desc="Samples", leave=False):
                if interleaved_gate is None:
                    pulse_name_seq, total_clifford = gate_sequence(depth, debug=False)
                else:
                    pulse_name_seq, total_clifford = interleaved_gate_sequence(
                        depth, gate_char=interleaved_gate, debug=False
                    )

                full_sequence = expand_full_sequence(pulse_name_seq, total_clifford)
                self.cfg["gate_seq"] = full_sequence

                prog = TestRBProgram(
                    self.soccfg,
                    reps=self.cfg["reps"],
                    final_delay=self.cfg["relax_delay"],
                    cfg=self.cfg,
                )
                iq_list = prog.acquire(self.soc, rounds=py_avg, progress=False)
                rblist.append(iq_list[0][0].dot([1, 1j]))

            rb_result[idx] = rblist

        self.rb_result = rb_result

    def plot(self, label="TestRB", color="steelblue", ax=None, show_individual=False):
        """
        Fit and plot the RB decay curve.

        Parameters
        ----------
        label : str, optional
            Legend label for this curve.
        color : str, optional
            Line and marker colour.
        ax : matplotlib.axes.Axes or None, optional
            Axes to plot into.  A new figure is created when ``None``.
        show_individual : bool, optional
            Whether to show individual circuit samples as scatter points.

        Returns
        -------
        epc : float
            Error per Clifford.
        epc_err : float
            One-sigma uncertainty on EPC.
        p_fit : float
            Fitted decay parameter.
        p_err : float
            One-sigma uncertainty on the decay parameter.
        pCov : ndarray
            Covariance matrix from the fit.

        Raises
        ------
        RuntimeError
            If ``run()`` has not been called first.
        """
        if self.x is None or self.rb_result is None:
            raise RuntimeError("Call run() first.")

        _proc = np.real if self._iq_process == "real" else np.abs
        raw = _proc(np.array(self.rb_result))
        avg = raw.mean(axis=1)

        pOpt, pCov = fitrb(self.x, avg)
        p_fit = float(pOpt[0])
        p_err = float(np.sqrt(np.diag(pCov))[0]) if pCov is not None else 0.0
        epc = rb_error(p_fit, d=2)
        epc_err = float(np.sqrt(error_fit_err(pCov[0, 0], d=2))) if pCov is not None else 0.0

        print(f"\n--- {label} ---")
        print(f"  p   = {p_fit * 100:.6f} ± {p_err * 100:.6f} %")
        print(f"  EPC = {epc * 100:.6f} ± {epc_err * 100:.6f} %")

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 5))

        if show_individual:
            for s in range(raw.shape[1]):
                ax.scatter(self.x, raw[:, s], s=6, color="gray", alpha=0.25,
                           linewidths=0, zorder=1)

        xfit = np.linspace(self.x.min(), self.x.max(), 400)
        ax.plot(xfit, rb_func(xfit, *pOpt), color=color, linewidth=2.0, zorder=3)
        ax.errorbar(self.x, avg,
                    yerr=raw.std(axis=1) / np.sqrt(raw.shape[1]),
                    fmt="none", ecolor=color, capsize=3, zorder=4)
        ax.scatter(self.x, avg, s=60, color=color, edgecolors="black",
                   label=label, zorder=5)
        ax.set_xlabel("Cliffords")
        ax.set_ylabel("Signal amplitude (a.u.)")
        ax.legend()
        plt.show()

        return epc, epc_err, p_fit, p_err, pCov

    def saveLabber(self, qb_idx, config_all=None, yoko_value=None):
        """
        Save RB data to an HDF5/Labber file.

        Parameters
        ----------
        qb_idx : int
            Qubit index appended to the experiment name.
        config_all : object or None, optional
            Full config object with a ``to_yaml(q_id)`` method.
        yoko_value : float or None, optional
            Yokogawa flux bias value embedded in the filename.

        Raises
        ------
        RuntimeError
            If ``run()`` has not been called first.
        """
        if self.x is None or self.rb_result is None:
            raise RuntimeError("Call run() first.")

        tag = f"IRB_{self._interleaved}" if self._interleaved else "ref"
        expt_name = f"s015_testrb_{tag}_{qb_idx}"

        from .base_experiment import BaseExperiment
        save_dir = BaseExperiment._data_path or DATA_PATH
        file_path = get_next_filename_labber(save_dir, expt_name, yoko_value)
        dict_val = (
            config_all.to_yaml(q_id=qb_idx)
            if config_all is not None
            else config_to_yaml(self.cfg)
        )
        raw = np.array(self.rb_result)

        hdf5_generator(
            filepath=file_path,
            x_info={"name": "Circuit Depth", "unit": "", "values": self.x.astype(float)},
            y_info={"name": "Sample Number", "unit": "",
                    "values": np.arange(self._number_sample, dtype=float)},
            z_info={"name": "Signal", "unit": "ADC unit", "values": raw.T},
            comment=str(dict_val),
            tag="RB",
        )
        print(f"Saved to {file_path}")
