"""
Coherence/ramsey — s006: Ramsey (ge) + AC Stark.
"""

from __future__ import annotations

from ...core.base_program import BaseProgram
from ...core.base_experiment import BaseExperiment
from ...analysis.qubit import RamseyAnalysis


class RamseyProgram(BaseProgram):
    """QICK program for Ramsey: two π/2 pulses with swept inter-pulse delay."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.add_loop("waitloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse1", gain_key="pi2_gain_ge")
        ramsey_phase = (
            cfg.get("qb_phase", 0) + cfg["wait_time"] * 360 * cfg["ramsey_freq"]
        )
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse2", gain_key="pi2_gain_ge", phase=ramsey_phase)

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse1", t=0)
        self.delay_auto(cfg["wait_time"] + 0.01, tag="wait")
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse2", t=0)
        self.delay_auto(0.05)
        self.measure(cfg)


class Ramsey(BaseExperiment):
    """
    Ramsey (ge): two π/2 pulses with swept delay.

    Fits decaying sinusoid (ramsey_freq≠0) or exponential (ramsey_freq=0)
    to extract T2* and frequency detuning.
    """

    EXPT_NAME = "s006_Ramsey_ge"
    TAG = "Ramsey"
    X_LABEL = "Ramsey Times (us)"
    TITLE_PREFIX = "Qubit Ramsey ge"
    SWEEP_KEYS_TO_REMOVE = ["wait_time"]
    X_SAVE_NAME = "Times"
    X_SAVE_UNIT = "s"
    X_SAVE_SCALE = 1e-6

    Analysis = RamseyAnalysis

    def _create_program(self):
        return RamseyProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        self.delay_times = prog.get_time_param("wait", "t", as_array=True)
        return self.delay_times

    def correct_detune(self):
        """Correct qubit ge frequency based on fitted detuning."""
        if self.result is None:
            raise RuntimeError("Run the experiment first.")
        detune = self.result.fit_result.get("detune_MHz", (None,))[0]
        if detune is None:
            print("Detune not available (ramsey_freq=0 or fit failed).")
            return self.cfg["qb_freq_ge"]
        if abs(detune - self.cfg["ramsey_freq"]) > 0.005:
            self.cfg["qb_freq_ge"] = self.cfg["qb_freq_ge"] - round(
                (detune - self.cfg["ramsey_freq"]), 2
            )
            print(f"over detune {round((detune - self.cfg['ramsey_freq']), 5)}MHz")
            return round(self.cfg["qb_freq_ge"], 5)
        else:
            print("Detune < 5kHz")
            return self.cfg["qb_freq_ge"]

    def _save_comment(self, dict_val):
        if self.result is not None:
            T2 = self.result.fit_result.get("T2r_us", (None,))[0]
            if T2 is not None:
                return f"T2 Ramsey = {T2:.2f} us\n{dict_val}"
        return str(dict_val)


class ACStarkProgram(BaseProgram):
    """QICK program for AC Stark shift measurement."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.add_loop("waitloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse1", gain_key="pi2_gain_ge")
        ramsey_phase = cfg.get("qb_phase", 0) + cfg["wait_time"] * 360 * cfg["ramsey_freq"]
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse2", gain_key="pi2_gain_ge", phase=ramsey_phase)

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)  # Stark tone on during wait
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse1", t=0)
        self.delay_auto(cfg["wait_time"] + 0.01, tag="wait")
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse2", t=0)
        self.delay_auto(0.05)
        self.measure(cfg)


class ACStark(Ramsey):
    """AC Stark shift measurement (s006_ac_stark)."""

    EXPT_NAME = "s006_ac_stark"
    TAG = "ACStark"
    TITLE_PREFIX = "AC Stark Shift"

    def _create_program(self):
        return ACStarkProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )
