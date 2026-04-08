"""
AutoRB — Automated Standard + Interleaved RB in one call
=========================================================
Wraps RandomizedBenchmarking to run Standard RB + multiple IRB gates,
compute gate fidelities, plot everything, and save all data.

Usage
-----
    auto = AutoRB(soc, soccfg, cfg)
    auto.run(
        py_avg=10,
        max_circuit_depth=1000,
        delta_clifford=100,
        number_sample=30,
        interleaved_gates=["X", "X/2", "Y", "Y/2"],
    )
    auto.plot()
    auto.saveLabber(qubit, config_all=config_all)
    print(auto.summary())

    # Access results as dict
    auto.results["X/2"]   # {"fidelity": ..., "epc": ..., ...}
    auto.results["ref"]   # reference RB result
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from tqdm.auto import tqdm

from .s015_Single_qubit_RB import RandomizedBenchmarking
from ..tools.fitting import fitrb, rb_func, rb_error, error_fit_err


# ──────────────────────────────────────────────
# Gate fidelity formula (Magesan et al. 2012)
# ──────────────────────────────────────────────


def _gate_fidelity(p_ref, p_irb, d=2):
    """F_gate = 1 - (d-1)/d * (1 - p_irb/p_ref)"""
    epc = (d - 1) / d * (1 - p_irb / p_ref)
    return 1 - epc, epc


def _gate_fidelity_err(p_ref, p_irb, var_p_ref, var_p_irb, d=2):
    """
    Error propagation for EPC = (d-1)/d * (1 - p_irb/p_ref).
    dEPC/dp_ref  = (d-1)/d * p_irb / p_ref^2
    dEPC/dp_irb  = -(d-1)/d / p_ref
    """
    c = (d - 1) / d
    depc_dpref = c * p_irb / p_ref**2
    depc_dpirb = -c / p_ref
    var_epc = depc_dpref**2 * var_p_ref + depc_dpirb**2 * var_p_irb
    return float(np.sqrt(var_epc))


# ──────────────────────────────────────────────
# AutoRB
# ──────────────────────────────────────────────


class AutoRB:
    """
    Automated Standard + Interleaved RB.

    Parameters
    ----------
    soc, soccfg, cfg : standard QICK objects / config dict

    After run():
    ------------
    self.results : dict
        {
            "ref": {
                "p": float,          # decay parameter
                "p_err": float,
                "epc": float,        # error per Clifford
                "epc_err": float,
                "pOpt": array,       # [p, A, B]
                "pCov": array,
                "rb_obj": RandomizedBenchmarking,
            },
            "X/2": {
                "p": float,
                "p_err": float,
                "epc": float,        # EPC of this specific gate
                "epc_err": float,
                "fidelity": float,   # gate fidelity = 1 - epc
                "fidelity_err": float,
                "pOpt": array,
                "pCov": array,
                "rb_obj": RandomizedBenchmarking,
            },
            ...
        }
    """

    # Plot colours
    _REF_COLOR = "tab:red"
    _IRB_COLORS = [
        "tab:blue",
        "tab:green",
        "tab:orange",
        "tab:purple",
        "tab:cyan",
        "tab:brown",
    ]

    def __init__(self, soc, soccfg, cfg):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = cfg
        self.results = {}
        self._gates = []
        self._run_params = {}

    def run(
        self,
        py_avg,
        max_circuit_depth,
        delta_clifford,
        number_sample,
        interleaved_gates=None,
        seed=None,
        prefix="ge",
    ):
        if interleaved_gates is None:
            interleaved_gates = []

        self._gates = list(interleaved_gates)
        self._run_params = dict(
            py_avg=py_avg,
            max_circuit_depth=max_circuit_depth,
            delta_clifford=delta_clifford,
            number_sample=number_sample,
            prefix=prefix,
        )

        # ── 1. Standard RB (Reference) ────────────────────────────────────────
        print("\n=== Standard RB (Reference) ===")
        rb_ref = RandomizedBenchmarking(self.soc, self.soccfg, self.cfg)
        rb_ref.run(
            py_avg=py_avg,
            max_circuit_depth=max_circuit_depth,
            delta_clifford=delta_clifford,
            number_sample=number_sample,
            seed=seed,
            prefix=prefix,
        )
        self.results["ref"] = self._fit_rb(rb_ref)

        p_ref = self.results["ref"]["p"]
        var_ref = self.results["ref"]["pCov"][0, 0]

        print(f"  p   = {p_ref * 100:.4f} ± {self.results['ref']['p_err'] * 100:.4f} %")
        print(f"  EPC = {self.results['ref']['epc'] * 100:.4f} %")

        # ── 2. IRB for each gate ──────────────────────────────────────────────
        for gate in interleaved_gates:
            # 檢查字串 gate 名稱是否有效 (避免跑一半噴錯)
            if isinstance(gate, str):
                from .RB_generator import SingleQubitRB
                if gate not in SingleQubitRB.ALLOWED_INTERLEAVE:
                    print(f"⚠️ 警告: Gate '{gate}' 未知或是未定義於 ALLOWED_INTERLEAVE，跳過此項。")
                    continue

            print(f"\n=== IRB  gate = {gate} ===")
            rb_irb = RandomizedBenchmarking(self.soc, self.soccfg, self.cfg)

            # 因為我們改過 s015，這裡直接傳入 gate 字串即可
            rb_irb.run(
                py_avg=py_avg,
                max_circuit_depth=max_circuit_depth,
                delta_clifford=delta_clifford,
                number_sample=number_sample,
                interleaved_gate=gate,
                seed=seed,
                prefix=prefix,
            )

            res = self._fit_rb(rb_irb)
            p_irb = res["p"]
            var_irb = res["pCov"][0, 0]

            # 計算 Gate Fidelity (d=2 對於單位元 RB)
            fidelity, epc = _gate_fidelity(p_ref, p_irb, d=2)
            epc_err = _gate_fidelity_err(p_ref, p_irb, var_ref, var_irb, d=2)

            res["fidelity"] = fidelity
            res["fidelity_err"] = epc_err
            res["epc"] = epc
            res["epc_err"] = epc_err
            self.results[gate] = res

            print(f"  p        = {p_irb * 100:.4f} ± {res['p_err'] * 100:.4f} %")
            print(f"  F_gate   = {fidelity * 100:.4f} ± {epc_err * 100:.4f} %")
            print(f"  EPC      = {epc * 100:.4f} ± {epc_err * 100:.4f} %")

    # ── Plot ──────────────────────────────────────────────────────────────────

    def plot(self, title=None, show_individual=True, figsize=(7, 5)):
        """
        Plot Reference RB + all IRB curves on one axes with annotation box.

        Returns
        -------
        fig, ax
        """
        if not self.results:
            raise RuntimeError("Must call run() before plot().")

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(0.8)
        ax.tick_params(
            colors="black", labelsize=11, direction="in", top=True, right=True, length=4
        )
        ax.grid(False)

        # Reference
        self._plot_one(ax, "ref", "Reference", self._REF_COLOR, "^", show_individual)

        # IRB gates
        for i, gate in enumerate(self._gates):
            if gate not in self.results:
                continue
            c = self._IRB_COLORS[i % len(self._IRB_COLORS)]
            self._plot_one(ax, gate, f"IRB  {gate}", c, "o", show_individual)

        # Annotation box
        ref = self.results["ref"]
        lines = [
            f"EPC (avg Clifford): {ref['epc'] * 100:.3f} ± {ref['epc_err'] * 100:.3f} %"
        ]
        for gate in self._gates:
            if gate not in self.results:
                continue
            r = self.results[gate]
            lines.append(
                f"F({gate:5s}): {r['fidelity'] * 100:.4f} ± {r['fidelity_err'] * 100:.4f} %"
            )
        ax.text(
            0.97,
            0.05,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            color="black",
            linespacing=1.7,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor="black",
                linewidth=0.8,
            ),
        )

        ax.set_xlabel("Cliffords", fontsize=12)
        ax.set_ylabel("Signal amplitude (a.u.)", fontsize=12)
        ax.legend(
            fontsize=10,
            frameon=True,
            framealpha=1.0,
            edgecolor="black",
            loc="upper right",
        )
        if title:
            ax.set_title(title, fontsize=12)
        fig.tight_layout()
        plt.show()
        return fig, ax

    # ── Save ──────────────────────────────────────────────────────────────────

    def saveLabber(self, qb_idx, config_all=None, yoko_value=None):
        """Save all RB / IRB datasets to Labber HDF5 files."""
        if not self.results:
            raise RuntimeError("Must call run() before saveLabber().")

        # Reference RB
        self.results["ref"]["rb_obj"].saveLabber(
            qb_idx=qb_idx,
            config_all=config_all,
            yoko_value=yoko_value,
        )

        # IRB gates
        for gate in self._gates:
            if gate not in self.results:
                continue
            self.results[gate]["rb_obj"].saveLabber(
                qb_idx=qb_idx,
                config_all=config_all,
                yoko_value=yoko_value,
            )

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self):
        """Return a formatted string summary of all results."""
        if not self.results:
            return "No results yet. Call run() first."

        lines = ["=" * 50, "  AutoRB Summary", "=" * 50]
        ref = self.results["ref"]
        lines.append(
            f"  Reference EPC : {ref['epc'] * 100:.4f} ± {ref['epc_err'] * 100:.4f} %"
        )
        lines.append(
            f"  p (decay)     : {ref['p'] * 100:.4f} ± {ref['p_err'] * 100:.4f} %"
        )
        if self._gates:
            lines.append("")
            lines.append("  Gate fidelities:")
            for gate in self._gates:
                if gate not in self.results:
                    continue
                r = self.results[gate]
                lines.append(
                    f"    {gate:6s}  F = {r['fidelity'] * 100:.4f} ± "
                    f"{r['fidelity_err'] * 100:.4f} %  "
                    f"(EPC = {r['epc'] * 100:.4f} %)"
                )
        lines.append("=" * 50)
        return "\n".join(lines)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fit_rb(self, rb_obj):
        """Fit a RandomizedBenchmarking object and return result dict."""
        raw = np.array(rb_obj.rb_result)
        amp = np.abs(raw)
        avg = amp.mean(axis=1)

        pOpt, pCov = fitrb(rb_obj.x, avg)
        p_fit = float(pOpt[0])
        p_err = float(np.sqrt(np.diag(pCov))[0]) if pCov is not None else 0.0
        epc = rb_error(p_fit, d=2)
        epc_err = (
            float(np.sqrt(error_fit_err(pCov[0, 0], d=2))) if pCov is not None else 0.0
        )

        return {
            "p": p_fit,
            "p_err": p_err,
            "epc": epc,
            "epc_err": epc_err,
            "pOpt": pOpt,
            "pCov": pCov,
            "x": rb_obj.x,
            "avg": avg,
            "rb_obj": rb_obj,
        }

    def _plot_one(self, ax, key, label, color, marker, show_individual):
        """Plot one RB/IRB curve onto ax."""
        res = self.results[key]
        x = res["x"]
        avg = res["avg"]
        raw = np.abs(np.array(res["rb_obj"].rb_result))
        sem = raw.std(axis=1) / np.sqrt(raw.shape[1])

        if show_individual:
            for s in range(raw.shape[1]):
                ax.scatter(
                    x, raw[:, s], s=4, color="gray", alpha=0.15, linewidths=0, zorder=1
                )

        xfit = np.linspace(x.min(), x.max(), 400)
        yfit = rb_func(xfit, *res["pOpt"])
        ax.plot(
            xfit, yfit, color=color, linewidth=2.0, zorder=3, solid_capstyle="round"
        )
        ax.errorbar(
            x,
            avg,
            yerr=sem,
            fmt="none",
            ecolor=color,
            elinewidth=1.0,
            capsize=3,
            capthick=1.0,
            zorder=4,
            alpha=0.8,
        )
        ax.scatter(
            x,
            avg,
            s=60,
            color=color,
            marker=marker,
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
            label=label,
        )
