"""
s015 — Single Qubit Randomized Benchmarking (RB)
==================================================
Clifford group generation + random sequence + inverse gate.
Uses setup_qb_pulse for all RB pulses (X, -X, X/2, -X/2, Y, -Y, Y/2, -Y/2, I, -I).
"""
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from .base_program import BaseProgram
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import hdf5_generator, get_next_filename_labber, config_to_yaml
from ..tools.fitting import fitrb, rb_func, rb_error, error_fit_err


# ── Clifford Group (preserved from original) ──

clifford_1q = dict()
clifford_1q["X"] = np.matrix([
    [0,0,0,1,0,0],[0,1,0,0,0,0],[0,0,0,0,0,1],
    [1,0,0,0,0,0],[0,0,0,0,1,0],[0,0,1,0,0,0]])
clifford_1q["Y"] = np.matrix([
    [0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,1,0,0,0],
    [1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,0,0,0,1]])
clifford_1q["X/2"] = np.matrix([
    [0,0,1,0,0,0],[0,1,0,0,0,0],[0,0,0,1,0,0],
    [0,0,0,0,0,1],[0,0,0,0,1,0],[1,0,0,0,0,0]])
clifford_1q["Y/2"] = np.matrix([
    [0,0,0,0,1,0],[1,0,0,0,0,0],[0,0,1,0,0,0],
    [0,1,0,0,0,0],[0,0,0,1,0,0],[0,0,0,0,0,1]])
clifford_1q["-X/2"] = np.matrix([
    [0,0,0,0,0,1],[0,1,0,0,0,0],[1,0,0,0,0,0],
    [0,0,1,0,0,0],[0,0,0,0,1,0],[0,0,0,1,0,0]])
clifford_1q["-Y/2"] = np.matrix([
    [0,1,0,0,0,0],[0,0,0,1,0,0],[0,0,1,0,0,0],
    [0,0,0,0,1,0],[1,0,0,0,0,0],[0,0,0,0,0,1]])
identity = np.diag([1] * 6)
clifford_1q["I"] = identity

step_pulses = [
    ("Y/2","X"), ("Y/2","X/2"), ("X/2","-Y/2","-X/2"), ("-X/2","-Y/2"),
    ("Y/2","-X/2"), ("-X/2","Y/2","-X/2"), ("X/2","Y/2"), ("-Y/2","X"),
    ("-X/2","Y"), ("-Y/2","-X/2"), ("X/2","Y/2","X/2"), ("-X/2","Y/2"),
    ("X","Y"), ("X/2","Y"), ("-Y/2","X/2"), ("X/2","Y/2","-X/2"),
    ("X/2","-Y/2"),
]

for pulse in step_pulses:
    new_mat = clifford_1q[pulse[0]]
    for p in pulse[1:]:
        new_mat = new_mat @ clifford_1q[p]
    repeat = False
    for existing_pulse_name, existing_pulse in clifford_1q.items():
        if isinstance(existing_pulse, tuple):
            if np.array_equal(new_mat, existing_pulse[0]):
                repeat = True
                break
        elif np.array_equal(new_mat, existing_pulse):
            repeat = True
            break
    if not repeat:
        clifford_1q[",".join(pulse)] = new_mat

clifford_1q_names = list(clifford_1q.keys())
assert len(clifford_1q_names) == 24, f"{len(clifford_1q_names)} elements instead of 24"

for name, matrix in clifford_1q.items():
    if isinstance(matrix, tuple):
        continue
    z_new = np.argmax(matrix[:, 0])
    x_new = np.argmax(matrix[:, 1])
    clifford_1q[name] = (matrix, (z_new, x_new))


def gate_sequence(rb_depth, pulse_n_seq=None, debug=False):
    if pulse_n_seq is None:
        pulse_n_seq = (len(clifford_1q_names) * np.random.rand(rb_depth)).astype(int)
    pulse_name_seq = [clifford_1q_names[n] for n in pulse_n_seq]
    psi_nz = np.matrix([[1,0,0,0,0,0]]).transpose()
    psi_nx = np.matrix([[0,1,0,0,0,0]]).transpose()
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
    assert total_clifford is not None, f"Failed to invert gate sequence!"
    return pulse_name_seq, total_clifford


def interleaved_gate_sequence(rb_depth, gate_char, debug=False):
    pulse_n_seq_rand = (len(clifford_1q_names) * np.random.rand(rb_depth)).astype(int)
    pulse_n_seq = []
    n_gate_char = clifford_1q_names.index(gate_char)
    for n_rand in pulse_n_seq_rand:
        pulse_n_seq.append(n_rand)
        pulse_n_seq.append(n_gate_char)
    return gate_sequence(len(pulse_n_seq), pulse_n_seq=pulse_n_seq, debug=debug)


def expand_full_sequence(pulse_name_seq, total_clifford):
    full_sequence = []
    for name in pulse_name_seq:
        for g in reversed(name.split(",")):
            full_sequence.append(g)
    for gate in total_clifford.split(","):
        neg = "-" in gate
        neg = not neg
        if neg:
            gate = "-" + gate if "-" not in gate else gate
        else:
            gate = gate.replace("-", "") if "-" in gate else gate
        full_sequence.append(gate)
    final = []
    for item in full_sequence:
        final.extend(item.split(",")) if "," in item else final.append(item)
    return final


# ── Program ──

class RBProgram(BaseProgram):
    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, 'ge')

        # RB-specific pulses: X, -X, X/2, -X/2, Y, -Y, Y/2, -Y/2
        # I and -I are handled as delays in _body, no pulse needed
        rb_pulses = [
            ("X",    0,  cfg["pi_gain_ge"]),
            ("-X",   0, -cfg["pi_gain_ge"]),
            ("X/2",  0,  cfg["pi2_gain_ge"]),
            ("-X/2", 0, -cfg["pi2_gain_ge"]),
            ("Y",    90,  cfg["pi_gain_ge"]),
            ("-Y",   90, -cfg["pi_gain_ge"]),
            ("Y/2",  90,  cfg["pi2_gain_ge"]),
            ("-Y/2", 90, -cfg["pi2_gain_ge"]),
        ]
        for pulse_name, phase, gain in rb_pulses:
            self.setup_qb_pulse(
                cfg, 'ge', name=pulse_name, phase=phase, gain_override=gain,
            )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)

        for gate in cfg["gate_seq"]:
            if gate in ("I", "-I"):
                self.delay_auto(cfg["sigma_ge"] * 5)
            else:
                self.pulse(ch=cfg["qb_ch"], name=gate, t=0)
                self.delay_auto(0.01)

        self.delay_auto(0.05)
        self.measure(cfg)


# ── Experiment ──

class RandomizedBenchmarking:
    """Standard or Interleaved Randomized Benchmarking."""

    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config
        self.x = None

    def run(self, py_avg, max_circuit_depth, delta_clifford, number_sample,
            interleaved_gate=None, simulate=False):
        self.x = np.arange(1, max_circuit_depth, delta_clifford)

        if simulate:
            from .mock_signals import mock_rb
            # Generate mock RB data: p~0.99 per Clifford with per-sample noise
            rb_result = []
            for depth in self.x:
                samples = []
                for _ in range(number_sample):
                    val = mock_rb(np.array([depth]), p=0.99, amp=0.5, offset=0.5, noise=0.03)
                    samples.append(val[0])
                rb_result.append(samples)
            self.rb_result = rb_result
            print(f"[SIMULATED] RB data generated: {len(self.x)} depths × {number_sample} samples")
            return

        rb_result = []
        desc = "Standard RB depth"
        if interleaved_gate is not None:
            assert interleaved_gate in clifford_1q_names
            desc = f"Interleaved RB ({interleaved_gate}) depth"

        for depth in tqdm(self.x, desc=desc):
            rblist = []
            for _ in tqdm(range(number_sample), desc="Samples", leave=False):
                if interleaved_gate is None:
                    pulse_name_seq, total_clifford = gate_sequence(depth)
                else:
                    pulse_name_seq, total_clifford = interleaved_gate_sequence(
                        depth, gate_char=interleaved_gate)
                self.cfg["gate_seq"] = expand_full_sequence(pulse_name_seq, total_clifford)
                prog = RBProgram(
                    self.soccfg, reps=self.cfg["reps"],
                    final_delay=self.cfg["relax_delay"], cfg=self.cfg,
                )
                iq_list = prog.acquire(self.soc, rounds=py_avg, progress=False)
                rblist.append(iq_list[0][0].dot([1, 1j]))
            rb_result.append(rblist)
        self.rb_result = rb_result

    def plot(self, label, color=None):
        if self.x is None:
            raise RuntimeError("Must run() before plotting.")
        std_r_avg = np.abs(np.mean(self.rb_result, axis=1))
        std_r_std = np.abs(np.std(self.rb_result, axis=1))
        pOpt, pCov = fitrb(self.x, std_r_avg)
        p_fit = pOpt[0]
        p_fit_err = np.sqrt(np.diag(pCov))[0] if pCov is not None else 0
        epc = rb_error(p_fit, d=2)
        epc_err = np.sqrt(error_fit_err(pCov[0, 0], d=2)) if pCov is not None else 0

        print(f"\n--- {label} ---")
        print(f"  p = {p_fit*100:.6f} ± {p_fit_err*100:.6f} %")
        print(f"  EPC = {epc*100:.6f} ± {epc_err*100:.6f} %")

        xfit = np.linspace(np.min(self.x), np.max(self.x), 200)
        yfit = rb_func(xfit, *pOpt)
        plt.errorbar(self.x, std_r_avg, yerr=std_r_std, fmt="o",
                      label=f"{label} (Data)", capsize=5, color=color)
        plt.plot(xfit, yfit, "-",
                 label=f"{label}: p={p_fit*100:.3f}±{p_fit_err*100:.3f}%, EPC={epc*100:.3f}±{epc_err*100:.3f}%",
                 color=color)
        return epc, epc_err, p_fit, p_fit_err
