import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from .base_program import BaseProgram

# 修改 Import：改用類別方式
from .RB_generator import SingleQubitRB
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import hdf5_generator, get_next_filename_labber, config_to_yaml
from ..tools.fitting import fitrb, rb_func, rb_error, error_fit_err

# ====================================================== #
# Gate name translation
# ====================================================== #
_GEN_TO_QICK = {
    "I": None,
    "X": "x180_{pfx}",
    "Y": "y180_{pfx}",
    "X/2": "x90_{pfx}",
    "-X/2": "x90m_{pfx}",
    "Y/2": "y90_{pfx}",
    "-Y/2": "y90m_{pfx}",
}


def _to_qick_gates(pulse_list, prefix="ge"):
    out = []
    for g in pulse_list:
        template = _GEN_TO_QICK.get(g)
        if template is None:
            continue
        out.append(template.format(pfx=prefix))
    return out


# 檔名後置詞映射
_INTERLEAVED_FILE_SUFFIX = {
    "X": "X",
    "Y": "Y",
    "X/2": "halfX",
    "-X/2": "halfXm",
    "Y/2": "halfY",
    "-Y/2": "halfYm",
}


# ====================================================== #
# QICK Program
# ====================================================== #
class RBProgram(BaseProgram):
    def _initialize(self, cfg):
        prefix = cfg.get("prefix", "ge")
        self.setup_resonator(cfg, prefix=prefix)
        self.setup_qubit_gen(cfg, prefix=prefix)
        self.setup_standard_gates(cfg, prefix=prefix)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.cooling_body(cfg)
        for gate in cfg["gate_seq"]:
            self.pulse(ch=cfg["qb_ch"], name=gate, t=0)
            self.delay_auto(0.01)
        self.delay_auto(0.05)
        self.measure(cfg)


# ====================================================== #
# Experiment class
# ====================================================== #
class RandomizedBenchmarking:
    def __init__(self, config):
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
        self.rb_gen = SingleQubitRB()

    def run(
        self,
        py_avg,
        max_circuit_depth,
        delta_clifford,
        number_sample,
        interleaved_gate=None,
        seed=None,
        prefix="ge",
        iq_process="abs",
        randomize_depth_order=False,
    ):
        """
        iq_process : "abs" | "real"
            Use "real" after readout optimization (best SNR on I axis).
        randomize_depth_order : bool
            If True, measure depths in a random order instead of ascending order.
            Results are reordered back to ascending depth before fitting/plotting,
            so time drift is spread evenly across all depth points.
        """
        self._iq_process = iq_process
        self.x = np.arange(1, max_circuit_depth, delta_clifford)
        self._number_sample = number_sample
        self._interleaved = interleaved_gate
        self._prefix = prefix

        is_irb = interleaved_gate is not None
        desc = f"IRB ({interleaved_gate})" if is_irb else "Standard RB"
        parent_rng = np.random.default_rng(seed)

        # Build measurement order (shuffled or sorted)
        depth_indices = np.arange(len(self.x))
        if randomize_depth_order:
            parent_rng.shuffle(depth_indices)
            print(f"  Depth order: {self.x[depth_indices].tolist()}")

        rb_result_unordered = [None] * len(self.x)
        for idx in tqdm(depth_indices, desc=desc):
            depth = self.x[idx]
            rblist = []
            for _ in tqdm(range(number_sample), desc="Samples", leave=False):
                child_seed = int(parent_rng.integers(0, 2**31))

                # 使用新版 SingleQubitRB 類別
                seq_objs = self.rb_gen.generate_single_qb_rb(
                    n_cliffords=depth,
                    n_samples=1,
                    interleave_gate=interleaved_gate,
                    seed=child_seed,
                )
                seq_obj = seq_objs[0]
                raw_pulses = seq_obj.pulse_sequence

                gate_seq = _to_qick_gates(raw_pulses, prefix=prefix)
                self.cfg["gate_seq"] = gate_seq

                prog = RBProgram(
                    self.soccfg,
                    reps=self.cfg["reps"],
                    final_delay=self.cfg["relax_delay"],
                    cfg=self.cfg,
                )
                iq_list = prog.acquire(self.soc, rounds=py_avg, progress=False)
                rblist.append(iq_list[0][0].dot([1, 1j]))

            rb_result_unordered[idx] = rblist

        self.rb_result = rb_result_unordered

    def saveLabber(self, qb_idx, config_all=None, yoko_value=None, title=None):
        if self.x is None or self.rb_result is None:
            raise RuntimeError("Must call run() before saveLabber().")

        if title is not None:
            expt_name = f"s015_RB_{title}_{qb_idx}"
        elif self._interleaved is not None:
            suffix = _INTERLEAVED_FILE_SUFFIX.get(self._interleaved, self._interleaved)
            expt_name = f"s015_RB_{suffix}_{qb_idx}"
        else:
            expt_name = f"s015_RB_{qb_idx}_ref"

        from .base_experiment import BaseExperiment
        save_dir = BaseExperiment._data_path or DATA_PATH
        file_path = get_next_filename_labber(save_dir, expt_name, yoko_value)
        dict_val = (
            config_all.to_yaml(q_id=qb_idx)
            if config_all is not None
            else config_to_yaml(self.cfg)
        )
        raw = np.array(self.rb_result)
        iq_data = raw.T

        hdf5_generator(
            filepath=file_path,
            x_info={
                "name": "Circuit Depth",
                "unit": "",
                "values": self.x.astype(float),
            },
            y_info={
                "name": "Sample Number",
                "unit": "",
                "values": np.arange(self._number_sample, dtype=float),
            },
            z_info={"name": "Signal", "unit": "ADC unit", "values": iq_data},
            comment=str(dict_val),
            tag="RB",
        )
        print(f"RB data saved to {file_path}")

    def plot(self, label, color=None, ax=None, marker="o", show_individual=False):
        if self.x is None or self.rb_result is None:
            raise RuntimeError("Must call run() before plot().")

        _proc = np.real if getattr(self, "_iq_process", "abs") == "real" else np.abs
        raw = np.array(self.rb_result)
        amp = _proc(raw)
        avg = amp.mean(axis=1)

        pOpt, pCov = fitrb(self.x, avg)
        p_fit = pOpt[0]
        p_fit_err = float(np.sqrt(np.diag(pCov))[0]) if pCov is not None else 0.0
        epc = rb_error(p_fit, d=2)
        epc_err = (
            float(np.sqrt(error_fit_err(pCov[0, 0], d=2))) if pCov is not None else 0.0
        )

        print(f"\n--- {label} ---")
        print(f"  p   = {p_fit * 100:.6f} ± {p_fit_err * 100:.6f} %")
        print(f"  EPC = {epc * 100:.6f} ± {epc_err * 100:.6f} %")

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 5))
        c = color or "steelblue"

        if show_individual:
            for s in range(amp.shape[1]):
                ax.scatter(
                    self.x,
                    amp[:, s],
                    s=6,
                    color="gray",
                    alpha=0.25,
                    linewidths=0,
                    zorder=1,
                )

        xfit = np.linspace(self.x.min(), self.x.max(), 400)
        ax.plot(xfit, rb_func(xfit, *pOpt), color=c, linewidth=2.0, zorder=3)
        ax.errorbar(
            self.x,
            avg,
            yerr=amp.std(axis=1) / np.sqrt(amp.shape[1]),
            fmt="none",
            ecolor=c,
            capsize=3,
            zorder=4,
        )
        ax.scatter(
            self.x,
            avg,
            s=60,
            color=c,
            marker=marker,
            edgecolors="black",
            label=label,
            zorder=5,
        )

        return epc, epc_err, p_fit, p_fit_err, pCov
