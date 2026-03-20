# ===================================================================
# 1. Standard & Third-Party Scientific Libraries
# ===================================================================
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from IPython.display import display, clear_output

# ===================================================================
# 2. QICK Libraries
# ===================================================================
from qick import *
from qick.pyro import make_proxy
from qick.asm_v2 import AveragerProgramV2, QickSpan, QickSweep1D

# ===================================================================
# 3. User/Local Libraries
# ===================================================================
from ..tools.system_cfg import *
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import get_next_filename_labber, hdf5_generator
from ..tools.fitting import *
from ..tools.yamltool import yml_comment


# ######################################################
# ### Randomized Benchmarking QICK Program Class     ###
# ######################################################


"""
Define matrices representing (all) Clifford gates for single
qubit in the basis of Z, X, Y, -Z, -X, -Y, indicating
where on the 6 cardinal points of the Bloch sphere the
+Z, +X, +Y axes go after each gate. Each Clifford gate
can be uniquely identified just by checking where +X and +Y
go.
"""
clifford_1q = dict()
# clifford_1q["Z"] = np.matrix(
#     [
#         [1, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0],
#         [0, 0, 0, 0, 0, 1],
#         [0, 0, 0, 1, 0, 0],
#         [0, 1, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0, 0],
#     ]
# )
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
# clifford_1q["Z/2"] = np.matrix(
#     [
#         [1, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 1],
#         [0, 1, 0, 0, 0, 0],
#         [0, 0, 0, 1, 0, 0],
#         [0, 0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0],
#     ]
# )
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
# clifford_1q["-Z/2"] = np.matrix(
#     [
#         [1, 0, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0],
#         [0, 0, 0, 1, 0, 0],
#         [0, 0, 0, 0, 0, 1],
#         [0, 1, 0, 0, 0, 0],
#     ]
# )
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


# Read pulse as a matrix product acting on state (meaning apply pulses in reverse order of the tuple)
# two_step_pulses = [
#     ("X", "Z/2"),
#     ("X/2", "Z/2"),
#     ("-X/2", "Z/2"),
#     ("Y", "Z/2"),
#     ("Y/2", "Z/2"),
#     ("-Y/2", "Z/2"),
#     ("X", "Z"),
#     ("X/2", "Z"),
#     ("-X/2", "Z"),
#     ("Y", "Z"),
#     ("Y/2", "Z"),
#     ("-Y/2", "Z"),
#     ("X", "-Z/2"),
#     ("X/2", "-Z/2"),
#     ("-X/2", "-Z/2"),
#     ("Y", "-Z/2"),
#     ("Y/2", "-Z/2"),
#     ("-Y/2", "-Z/2"),
# ]

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
    # Make sure there are no repeats
    for existing_pulse_name, existing_pulse in clifford_1q.items():
        if np.array_equal(new_mat, existing_pulse):
            print("found repeat", pulse, existing_pulse_name)
            repeat = True
    if not repeat:
        clifford_1q[pulse[0] + "," + ",".join(pulse[1:])] = new_mat
clifford_1q_names = list(clifford_1q.keys())
assert len(clifford_1q_names) == 24, (
    f"you have {len(clifford_1q_names)} elements in your Clifford group instead of 24!"
)
# print(len(clifford_1q_names), "elements in clifford_1q")
# print(clifford_1q_names)

# Get the average number of X/2 gates per Clifford gate
count = 0
for n in range(len(clifford_1q_names)):  # n is index in clifford_1q_names
    gates = clifford_1q_names[n].split(",")
    for gate in gates:
        # print(gate)
        if gate == "I" or "Z" in gate:
            continue
        if "/2" in gate:
            count += 1
            # print("added 1 to count")
        else:
            count += 2
            # print("added 2 to count")
# print("Average number of X/2 gates per Clifford gate:", count / len(clifford_1q_names))

for name, matrix in clifford_1q.items():
    z_new = np.argmax(matrix[:, 0])  # +Z goes to row where col 0 is 1
    x_new = np.argmax(matrix[:, 1])  # +X goes to row where col 1 is 1
    # print(name, z_new, x_new)
    clifford_1q[name] = (matrix, (z_new, x_new))


def gate_sequence(rb_depth, pulse_n_seq=None, debug=False):
    """
    Generate RB forward gate sequence of length rb_depth as a list of pulse names;
    also return the Clifford gate that is equivalent to the total pulse sequence.
    The effective inverse is pi phase + the total Clifford.
    Optionally, provide pulse_n_seq which is a list of the indices of the Clifford
    gates to apply in the sequence.
    """
    if pulse_n_seq is None:
        pulse_n_seq = (len(clifford_1q_names) * np.random.rand(rb_depth)).astype(int)
    pulse_name_seq = [clifford_1q_names[n] for n in pulse_n_seq]
    if debug:
        print("pulse seq", pulse_name_seq)
    psi_nz = np.matrix([[1, 0, 0, 0, 0, 0]]).transpose()
    psi_nx = np.matrix([[0, 1, 0, 0, 0, 0]]).transpose()
    for n in pulse_n_seq:  # n is index in clifford_1q_names
        gates = clifford_1q_names[n].split(",")
        for gate in reversed(gates):  # Apply matrices from right to left of gates
            psi_nz = clifford_1q[gate][0] @ psi_nz
            psi_nx = clifford_1q[gate][0] @ psi_nx
    psi_nz = psi_nz.flatten()
    psi_nx = psi_nx.flatten()
    if debug:
        print("+Z axis after seq:", psi_nz, "+X axis after seq:", psi_nx)

    total_clifford = None
    if np.argmax(psi_nz) == 0:
        total_clifford = "I"
    else:
        for (
            clifford
        ) in clifford_1q_names:  # Get the clifford equivalent to the total seq
            if clifford_1q[clifford][1] == (np.argmax(psi_nz), np.argmax(psi_nx)):
                # z_new, x_new = clifford_1q[clifford][1]
                # if z_new == np.argmax(psi_nz):
                total_clifford = clifford
                break
    assert total_clifford is not None, (
        f"Failed to invert gate sequence! {pulse_name_seq} which brings +Z to {psi_nz}"
    )

    if debug:
        total_clifford_mat = clifford_1q[total_clifford][0]
        print("Total gate matrix:\n", total_clifford_mat)

    return pulse_name_seq, total_clifford


def interleaved_gate_sequence(rb_depth, gate_char: str, debug=False):
    """
    Generate RB gate sequence with rb_depth random gates interleaved with gate_char
    Returns the total gate list (including the interleaved gates) and the total
    Clifford gate equivalent to the total pulse sequence.
    """
    pulse_n_seq_rand = (len(clifford_1q_names) * np.random.rand(rb_depth)).astype(int)
    pulse_n_seq = []
    assert gate_char in clifford_1q_names
    n_gate_char = clifford_1q_names.index(gate_char)
    if debug:
        print("n gate char:", n_gate_char, clifford_1q_names[n_gate_char])
    for n_rand in pulse_n_seq_rand:
        pulse_n_seq.append(n_rand)
        pulse_n_seq.append(n_gate_char)
    return gate_sequence(len(pulse_n_seq), pulse_n_seq=pulse_n_seq, debug=debug)


def expand_full_sequence(pulse_name_seq, total_clifford):
    """
    Expand a full pulse_name_seq into the actual flat play sequence,
    handling normal and inverse gates correctly.

    Args:
        pulse_name_seq : list of str
            e.g., ['X/2', 'X/2', 'X/2,-Y/2,-X/2', 'X/2']
        total_clifford : str
            e.g., 'Y/2,-X/2'

    Returns:
        list of str
            Final flat sequence to be played on hardware, with each gate separately listed
    """
    full_sequence = []

    # Expand normal pulse names (right-to-left)
    for name in pulse_name_seq:
        gates = name.split(",")
        for g in reversed(gates):  # right-to-left
            full_sequence.append(g)

    # Expand total_clifford separately (left-to-right, inverse)
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


class RBProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        qb_ch = cfg["qb_ch"]

        # --- Standardized Generator Declaration ---
        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])

        if self.soccfg["gens"][qb_ch]["type"] == "axis_sg_int4_v2":
            self.declare_gen(ch=qb_ch, nqz=cfg["nqz_qb"], mixer_freq=cfg["qb_mixer"])
        else:
            self.declare_gen(ch=qb_ch, nqz=cfg["nqz_qb"])

        # --- Standardized Readout Declaration ---
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])
        self.add_readoutconfig(
            ch=ro_ch, name="myro", freq=cfg["res_freq_ge"], gen_ch=res_ch
        )

        # --- Standardized Readout Pulse ---
        self.add_gauss(
            ch=res_ch,
            name="readout",
            sigma=cfg["res_sigma"],
            length=5 * cfg["res_sigma"],
            even_length=True,
        )
        self.add_pulse(
            ch=res_ch,
            name="res_pulse",
            ro_ch=ro_ch,
            style="flat_top",
            envelope="readout",
            length=cfg["res_length"],
            freq=cfg["res_freq_ge"],
            phase=cfg["res_phase"],
            gain=cfg["res_gain_ge"],
        )

        # --- Standardized qb Pulses (RB-specific) ---
        self.add_gauss(
            ch=qb_ch,
            name="ramp",
            sigma=cfg["sigma"],
            length=cfg["sigma"] * 5,
            even_length=True,
        )

        # Define RB pulse properties
        pulse_list = [
            ("X", 0, cfg["pi_gain_ge"]),
            ("-X", 0, -cfg["pi_gain_ge"]),
            ("X/2", 0, cfg["pi2_gain_ge"]),
            ("-X/2", 0, -cfg["pi2_gain_ge"]),
            ("Y", 90, cfg["pi_gain_ge"]),
            ("-Y", 90, -cfg["pi_gain_ge"]),
            ("Y/2", 90, cfg["pi2_gain_ge"]),
            ("-Y/2", 90, -cfg["pi2_gain_ge"]),
            ("I", 0, 0),  # Add Identity
            ("-I", 0, 0),  # Add Identity
        ]

        if cfg["pulse_type"] == "arb":
            for name, phase, gain in pulse_list:
                self.add_pulse(
                    ch=qb_ch,
                    name=name,
                    style="arb",
                    envelope="ramp",
                    freq=cfg["qb_freq_ge"],
                    phase=phase,
                    gain=gain,
                )

        elif cfg["pulse_type"] == "flat_top":
            for name, phase, gain in pulse_list:
                self.add_pulse(
                    ch=qb_ch,
                    name=name,
                    style="flat_top",
                    envelope="ramp",
                    freq=cfg["qb_freq_ge"],
                    phase=phase,
                    gain=gain,
                    length=cfg["flat_top_len"],
                )

    def apply_cool(self, cfg):
        # --- Standardized Cooling Method ---
        cool_ch1 = cfg["cool_ch1"]
        cool_ch2 = cfg["cool_ch2"]
        if self.soccfg["gens"][cool_ch1]["type"] == "axis_sg_int4_v2":
            self.declare_gen(
                ch=cool_ch1, nqz=cfg["nqz_cool_ch1"], mixer_freq=cfg["cool_mixer1"]
            )
        else:
            self.declare_gen(ch=cool_ch1, nqz=cfg["nqz_cool_ch1"])

        if self.soccfg["gens"][cool_ch2]["type"] == "axis_sg_int4_v2":
            self.declare_gen(
                ch=cool_ch2, nqz=cfg["nqz_cool_ch2"], mixer_freq=cfg["cool_mixer2"]
            )
        else:
            self.declare_gen(ch=cool_ch2, nqz=cfg["nqz_cool_ch2"])

        self.add_gauss(
            ch=cool_ch1,
            name="cooling1",
            sigma=cfg["res_sigma"],  # Use res_sigma for cooling ramp
            length=5 * cfg["res_sigma"],
            even_length=True,
        )
        self.add_pulse(
            ch=cool_ch1,
            name="cool_pulse1",
            style="flat_top",
            envelope="cooling1",
            length=cfg["cool_length"],
            freq=cfg["cool_freq_1"],
            phase=0,
            gain=cfg["cool_gain_1"],
        )
        self.add_gauss(
            ch=cool_ch2,
            name="cooling2",
            sigma=cfg["res_sigma"],
            length=5 * cfg["res_sigma"],
            even_length=True,
        )
        self.add_pulse(
            ch=cool_ch2,
            name="cool_pulse2",
            style="flat_top",
            envelope="cooling2",
            length=cfg["cool_length"],
            freq=cfg["cool_freq_2"],
            phase=0,
            gain=cfg["cool_gain_2"],
        )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)

        # --- Cooling (if enabled) ---
        if cfg.get("cooling", False) is True:
            self.apply_cool(cfg)
            self.pulse(ch=self.cfg["cool_ch1"], name="cool_pulse1", t=0)
            self.pulse(ch=self.cfg["cool_ch2"], name="cool_pulse2", t=0)
            self.delay_auto(0.5, tag="Ring down")  # Wait for cooling pulse to ring down
        else:
            pass

        # --- RB Gate Sequence ---
        for i in self.cfg["gate_seq"]:
            if i == "I" or i == "-I":
                self.delay_auto(cfg["sigma"] * 5)
            else:
                self.pulse(ch=self.cfg["qb_ch"], name=f"{i}", t=0)
                self.delay_auto(0.01)  # Small delay between pulses

        # --- Readout ---
        self.delay_auto(0.05)  # wait_time after last pulse
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)  # play probe pulse
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


# ######################################################
# ### Randomized Benchmarking Experiment Class       ###
# ######################################################


class RandomizedBenchmarking:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config
        self.x = None  # x-axis (depths) will be set by run()

    def run(
        self,
        py_avg,
        max_circuit_depth,
        delta_clifford,
        number_sample,
        interleaved_gate=None,
    ):
        """
        Runs Standard or Interleaved Randomized Benchmarking.

        Parameters:
        - py_avg (int): Software averages.
        - max_circuit_depth (int): Maximum number of Clifford gates.
        - delta_clifford (int): Step size for circuit depth.
        - number_sample (int): Number of random sequences per depth.
        - interleaved_gate (str, optional):
            If None (default), runs Standard RB.
            If a gate name (e.g., "X/2"), runs Interleaved RB with that gate.
        """
        self.x = np.arange(1, max_circuit_depth, delta_clifford)
        rb_result = []

        run_desc = "Standard RB depth"
        if interleaved_gate is not None:
            if interleaved_gate not in clifford_1q_names:
                raise ValueError(
                    f"Interleaved gate '{interleaved_gate}' is not in the defined clifford_1q_names"
                )
            run_desc = f"Interleaved RB ({interleaved_gate}) depth"

        for i in tqdm(self.x, desc=run_desc):
            rblist = []
            for _ in tqdm(range(number_sample), desc="Number of samples", leave=False):
                if interleaved_gate is None:
                    # --- Standard RB ---
                    pulse_name_seq, total_clifford = gate_sequence(i, debug=False)
                else:
                    # --- Interleaved RB ---
                    pulse_name_seq, total_clifford = interleaved_gate_sequence(
                        i, gate_char=interleaved_gate, debug=False
                    )

                full_sequence = expand_full_sequence(pulse_name_seq, total_clifford)
                self.cfg["gate_seq"] = full_sequence

                rb = RBProgram(
                    self.soccfg,
                    reps=self.cfg["reps"],
                    final_delay=self.cfg["relax_delay"],
                    cfg=self.cfg,
                )

                iq_list = rb.acquire(self.soc, rounds=py_avg, progress=False)
                # Convert I/Q list to complex number
                rblist.append(iq_list[0][0].dot([1, 1j]))

            rb_result.append(rblist)
        self.rb_result = rb_result

    def plot(self, label, color=None):
        """
        Plots and fits the RB data.

        Parameters:
        - rb_result (list): The data returned from the run() method.
        - label (str): Label for the data series (e.g., "Standard RB").
        - color (str, optional): Color for the plot.

        Returns:
        - tuple: (epc, epc_err, p_fit, p_fit_err)
        """
        if self.x is None:
            raise RuntimeError("Must run() the experiment before plotting.")

        # Process data: Calculate mean and std dev over samples
        std_r_avg = np.abs(np.mean(self.rb_result, axis=1))
        std_r_std = np.abs(np.std(self.rb_result, axis=1))

        # Fit the data to the RB decay function
        pOpt, pCov = fitrb(self.x, std_r_avg)

        p_fit = pOpt[0]
        p_fit_err = np.sqrt(np.diag(pCov))[0] if pCov is not None else 0

        # Calculate Error Per Clifford (EPC)
        epc = rb_error(p_fit, d=2)
        epc_err = np.sqrt(error_fit_err(pCov[0, 0], d=2)) if pCov is not None else 0

        print(f"\n--- Fitting Results for: {label} ---")
        print(f"  Fitted p = {p_fit * 100:.6f} ± {p_fit_err * 100:.6f} %")
        print(f"  Error per Clifford (EPC) = {epc * 100:.6f} ± {epc_err * 100:.6f} %")

        # Generate fitted curve
        xfit = np.linspace(np.min(self.x), np.max(self.x), 200)
        yfit = rb_func(xfit, *pOpt)

        # Plot data points with error bars
        plt.errorbar(
            self.x,
            std_r_avg,
            yerr=std_r_std,
            fmt="o",
            label=f"{label} (Data)",
            capsize=5,
            color=color,
        )

        # Plot the fit
        fit_label = (
            f"{label} (Fit): $p = {p_fit * 100:.3f} \pm {p_fit_err * 100:.3f}$ %\n"
            f"EPC = {epc * 100:.3f} \pm {epc_err * 100:.3f}$ %"
        )
        plt.plot(xfit, yfit, "-", label=fit_label, color=color)

        # Return the key metrics for further analysis
        return (epc, epc_err, p_fit, p_fit_err)
