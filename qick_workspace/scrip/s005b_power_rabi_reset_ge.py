"""
s005b — Power Rabi ge (Active Reset)
======================================
Amplitude Rabi sweeping qubit drive gain, with active reset after each shot.

Sequence per shot:
  1. Probe pulse (swept gain)
  2. Readout → ADC integration (trigger 1 = actual Rabi data)
  3. wait_auto: tProc hardware WAIT until ADC window ends
  4. read_and_jump: if I < threshold (|g⟩) → skip π pulse
     else (|e⟩) → apply π pulse to reset qubit to |g⟩
  5. resync: re-establish ref_time for next shot
  6. AveragerProgramV2 final_delay: relax_delay for thermal re-thermalisation

Active reset guarantees qubit is in |g⟩ at the START of each shot.
The actual Rabi data comes from trigger 1 (iq_list[0][0]).

Required cfg keys (beyond standard Power Rabi):
  threshold   : float  — single-sample discrimination threshold in ADC units
                         Scaled internally by us2cycles(res_length).
                         I < threshold → |g⟩; I ≥ threshold → |e⟩.
  pi_gain_ge  : float  — fixed π-pulse DAC gain used for reset.
"""
import matplotlib.pyplot as plt
import numpy as np

from .base_program import BaseProgram
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import hdf5_generator, get_next_filename_labber, config_to_yaml
from ..tools.fitting import decaysin, fitdecaysin, fix_phase
from ..plotter.plot_utils import plot_final


class PowerRabiResetProgram(BaseProgram):
    """QICK program for Power Rabi with active reset after each shot."""

    def _initialize(self, cfg):
        """Set up resonator, qubit generator, swept probe pulse, and fixed pi reset pulse."""
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")

        # Swept probe pulse — gain varies with gainloop
        self.add_loop("gainloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse")

        # Fixed π pulse for active reset (gain does NOT sweep)
        self.setup_qb_pulse(
            cfg, "ge", name="pi_pulse", gain_override=cfg["pi_gain_ge"]
        )

    def _body(self, cfg):
        """Apply probe pulse, measure, then conditionally apply reset π pulse."""
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)

        # ── Probe pulse (Power Rabi sweep) ───────────────────────────────
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
        self.delay_auto(t=0.05)

        # ── Readout (trigger 1 = actual Rabi data) ───────────────────────
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])

        # Hardware WAIT: tProc truly pauses until ADC integration window ends.
        # Equivalent to v1 wait_all(); ref_time advances to end of readout.
        self.wait_auto(t=0, ros=True)

        # ── Active reset ─────────────────────────────────────────────────
        # Threshold = per-sample value × readout length in cycles.
        # I < threshold  →  |g⟩  →  jump (no reset needed)
        # I ≥ threshold  →  |e⟩  →  fall through to π pulse
        threshold = int(
            np.round(
                cfg["threshold"]
                * self.soccfg.us2cycles(cfg["res_length"], ro_ch=cfg["ro_ch"])
            )
        )
        self.read_and_jump(
            ro_ch=cfg["ro_ch"],
            component="I",
            threshold=threshold,
            test="<",
            label="after_reset",
        )

        # |e⟩ path: π pulse brings qubit back to |g⟩
        self.pulse(ch=cfg["qb_ch"], name="pi_pulse", t=0)

        self.label("after_reset")
        # Re-synchronise all channels after the conditional branch.
        # Prevents timeline drift when the two branches have different lengths.
        self.delay_auto(t=0.05)

        # ── Second readout (trigger 2 = post-reset verification) ─────────
        # Mirrors v1's second measure(..., wait=True, syncdelay=relax_delay).
        # iq_list[0][1] can be used offline to check reset efficiency.
        # AveragerProgramV2 appends final_delay (= cfg["relax_delay"]) after
        # the body, which covers the readout window + thermal re-thermalisation.
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class PowerRabiReset:
    """
    Power Rabi ge with active reset after each shot.

    Two triggers per body:

    * ``iq_list[0][0]`` — trigger 1: actual Rabi measurement data
    * ``iq_list[0][1]`` — trigger 2: post-reset verification (reset efficiency)

    Parameters
    ----------
    config : dict
        Experiment configuration dictionary.  Must contain ``threshold`` and
        ``pi_gain_ge`` in addition to the standard Power Rabi keys.

    Raises
    ------
    RuntimeError
        If ``BaseExperiment.setup()`` has not been called before instantiation.

    Examples
    --------
    ::

        cfg["threshold"] = <single-sample I threshold>  # from single-shot calibration
        expt = PowerRabiReset(cfg)
        expt.run(py_avg=100)
        expt.plot()
        expt.saveLabber(qb_idx=0)
    """

    EXPT_NAME = "s005b_power_rabi_reset_ge"
    TAG = "Rabi"

    def __init__(self, config):
        from .base_experiment import BaseExperiment
        if BaseExperiment._soc is None:
            raise RuntimeError("Call BaseExperiment.setup(soc, soccfg, data_path) first.")
        self.soc = BaseExperiment._soc
        self.soccfg = BaseExperiment._soccfg
        self.cfg = config
        self.x_vals = None        # sweep axis (DAC gain)
        self.iqdata = None        # complex IQ — trigger 1 (Rabi data)
        self.iqdata_reset = None  # complex IQ — trigger 2 (reset verification)
        self.fit_params = None
        self._iq_process = "abs"

    def run(self, py_avg=1, iq_process="abs", progress=True):
        """
        Acquire Power Rabi data with active reset.

        Parameters
        ----------
        py_avg : int, optional
            Software averages (rounds).
        iq_process : str, optional
            IQ processing mode: ``"abs"`` or ``"real"``.  Use ``"real"``
            after readout optimisation.
        progress : bool, optional
            Whether to show a tqdm progress bar.

        Returns
        -------
        iqdata : ndarray of complex
            Complex IQ data from trigger 1 (actual Rabi measurement),
            shape ``(n_gains,)``.
        """
        self._iq_process = iq_process

        prog = PowerRabiResetProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )
        iq_list = prog.acquire(self.soc, rounds=py_avg, progress=progress)

        self.x_vals = prog.get_pulse_param("qb_pulse", "gain", as_array=True)
        # trigger 1: actual Rabi data
        self.iqdata = iq_list[0][0].dot([1, 1j])
        # trigger 2: post-reset state (can be used to monitor reset efficiency)
        self.iqdata_reset = iq_list[0][1].dot([1, 1j])
        return self.iqdata

    def plot(self):
        """
        Plot Power Rabi with decaying-sinusoid fit and mark pi / pi/2 gains.

        Returns
        -------
        pi_gain : float
            Pi pulse gain (rounded to 6 decimal places).
        pi2_gain : float
            Pi/2 pulse gain (rounded to 6 decimal places).

        Raises
        ------
        RuntimeError
            If ``run()`` has not been called first.
        """
        if self.iqdata is None:
            raise RuntimeError("Must call run() before plot().")

        self.fit_params, _, fig, ax = plot_final(
            self.x_vals, self.iqdata, "Dac Gain (a.u)", fitdecaysin, decaysin,
            return_ax=True,
        )
        fig.suptitle("Power Rabi ge (Active Reset)")
        fig.tight_layout()
        pi_gain, pi2_gain = fix_phase(self.fit_params)
        ax.axvline(pi_gain,  color="red", linestyle="--", label=r"$\pi$ Gain")
        ax.axvline(pi2_gain, color="red", linestyle="--", label=r"$\pi/2$ Gain")
        ax.legend()
        print(f"π gain   = {pi_gain:.1f}")
        print(f"π/2 gain = {pi2_gain:.1f}")
        return round(pi_gain, 6), round(pi2_gain, 6)

    def saveLabber(self, qb_idx, config_all=None, yoko_value=None):
        """
        Save Rabi data (trigger 1) to an HDF5/Labber file.

        Parameters
        ----------
        qb_idx : int
            Qubit index; appended to the experiment name.
        config_all : object or None, optional
            Full config object with a ``to_yaml(q_id)`` method.  If ``None``,
            the current ``cfg`` dict is serialised with ``config_to_yaml``.
        yoko_value : float or None, optional
            Yokogawa flux bias value; embedded in the filename when provided.

        Raises
        ------
        RuntimeError
            If ``run()`` has not been called first.
        """
        if self.iqdata is None:
            raise RuntimeError("Must call run() before saveLabber().")

        from .base_experiment import BaseExperiment
        save_dir = BaseExperiment._data_path or DATA_PATH
        expt_name = f"{self.EXPT_NAME}_{qb_idx}"
        file_path = get_next_filename_labber(save_dir, expt_name, yoko_value)

        dict_val = (
            config_all.to_yaml(q_id=qb_idx)
            if config_all is not None
            else config_to_yaml(self.cfg)
        )
        hdf5_generator(
            filepath=file_path,
            x_info={
                "name": "Gain",
                "unit": "DAC unit",
                "values": self.x_vals.astype(float),
            },
            z_info={"name": "Signal", "unit": "ADC unit", "values": self.iqdata},
            comment=str(dict_val),
            tag=self.TAG,
        )
        print(f"Saved to {file_path}")
