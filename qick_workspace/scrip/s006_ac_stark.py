"""
s006 — AC Stark Phase Calibration (ge)
=======================================
Phase Error Pulse Sequence to measure and correct AC Stark shift induced
by the qubit drive pulse at a given amplitude and shape.

Sequence:
    X_pi/2  ->  (X_2pi)^N  ->  (X_pi/2)_phi

where phi is swept.  The AC Stark shift rotates the Bloch vector by an
extra angle delta per 2pi pulse.  After N repetitions the net phase error
is N * delta, visible as a shift of the interference fringe vs phi.

Usage
-----
Typical config additions needed:
    cfg["ac_stark_amp_key"]  : cfg key whose gain is under test (default "pi_gain_ge")
    cfg["ac_stark_N"]        : number of 2pi pulses (default 1; increase to amplify)

Run:
    expt = AcStarkCalib(cfg)
    expt.run(py_avg=200)
    phase_error_per_2pi, corrected_freq = expt.correct_ac_stark()
"""

import numpy as np

from .base_program import BaseProgram
from .base_experiment import BaseExperiment
from ..tools.fitting import sinfunc, fitsin
from ..plotter.plot_utils import plot_final


class AcStarkProgram(BaseProgram):
    """
    Sweep the second pi/2 pulse phase phi over [0, 360) deg.

    The 2pi block uses gain = 2 * pi_gain_ge (the target pulse whose AC Stark
    shift we want to characterise).  Amplitude and envelope shape must
    match the operational gate exactly.
    """

    def _initialize(self, cfg):
        """Set up resonator, qubit generator, and the three pulse types for AC Stark."""
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.add_loop("phaseloop", cfg["steps"])

        # First pi/2 pulse (fixed phase)
        self.setup_qb_pulse(cfg, "ge", name="pi2_pulse", gain_key="pi2_gain_ge")

        # 2pi pulse: same shape/amplitude as the gate under test.
        # gain = 2 * pi_gain_ge
        amp_key = cfg.get("ac_stark_amp_key", "pi_gain_ge")
        pi_gain = cfg[amp_key]
        self.setup_qb_pulse(
            cfg,
            "ge",
            name="twopi_pulse",
            gain_override=2 * pi_gain,
            phase=0.0,
        )

        # Second pi/2 pulse — phase swept by the outer loop
        self.setup_qb_pulse(
            cfg,
            "ge",
            name="pi2_phi_pulse",
            gain_key="pi2_gain_ge",
            phase=cfg.get("qb_phase", 0.0),
        )

    def _body(self, cfg):
        """Apply X_pi/2, then N X_2pi pulses, then swept-phase X_pi/2, then measure."""
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)

        N = int(cfg.get("ac_stark_N", 1))

        # X_pi/2
        self.pulse(ch=cfg["qb_ch"], name="pi2_pulse", t=0)
        self.delay_auto(0.01)

        # (X_2pi)^N
        for _ in range(N):
            self.pulse(ch=cfg["qb_ch"], name="twopi_pulse", t=0)
            self.delay_auto(0.01)

        # (X_pi/2)_phi
        self.pulse(ch=cfg["qb_ch"], name="pi2_phi_pulse", t=0)
        self.delay_auto(0.05)
        self.measure(cfg)


class AcStarkCalib(BaseExperiment):
    """
    AC Stark shift calibration via Phase Error Pulse Sequence.

    Sweep axis : second pi/2 phase phi in [0, 360) deg.
    Fit        : sinusoid -> phase offset = AC Stark phase error per N 2pi pulses.
    """

    EXPT_NAME = "s006_ac_stark_ge"
    TAG = "AcStark"
    X_LABEL = "Second pi/2 Phase (deg)"
    TITLE_PREFIX = "AC Stark Calibration ge"
    SWEEP_KEYS_TO_REMOVE = ["qb_phase"]
    X_SAVE_NAME = "Phase"
    X_SAVE_UNIT = "deg"
    X_SAVE_SCALE = 1.0

    def _create_program(self):
        """Instantiate and return the AcStarkProgram."""
        return AcStarkProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        """Return the phase sweep axis in degrees."""
        self.phase_pts = prog.get_pulse_param("pi2_phi_pulse", "phase", as_array=True)
        return self.phase_pts

    def _post_fit(self, x_vals):
        """
        Fit a sinusoid to the phase-swept data and extract AC Stark phase error.

        Parameters
        ----------
        x_vals : ndarray
            Phase sweep axis in degrees.

        Returns
        -------
        phase_err_per_2pi : float
            AC Stark phase error in degrees per single 2pi pulse.
        error : array
            One-sigma parameter uncertainties from the sinusoid fit.
        """
        self.fit_params, error, fig = plot_final(
            x_vals, self.iqdata, "Phase (deg)", fitsin, sinfunc
        )
        N = int(self.cfg.get("ac_stark_N", 1))
        phase_err_total   = float(self.fit_params[2])   # deg; sin fit phase offset
        phase_err_per_2pi = phase_err_total / N          # deg per single 2pi pulse

        fig.suptitle(
            f"AC Stark phase error = {phase_err_per_2pi:.3f} deg / 2π pulse  "
            f"(N={N},  total={phase_err_total:.3f} deg)",
            fontsize=13,
        )
        fig.tight_layout()

        self._phase_err_per_2pi = phase_err_per_2pi
        self._phase_err_total   = phase_err_total
        return phase_err_per_2pi, error

    def correct_ac_stark(self):
        """
        Translate the measured phase error into a qubit frequency correction.

        A phase error delta (deg) accumulated over one 2pi pulse of duration
        T_2pi (us) corresponds to a frequency offset::

            df = delta / (360 * T_2pi)   [MHz]

        Returns
        -------
        phase_err_per_2pi : float
            Measured phase error in degrees per 2pi pulse.
        corrected_freq : float
            Suggested corrected ``qb_freq_ge`` in MHz.

        Raises
        ------
        RuntimeError
            If the experiment has not been run yet (``_post_fit`` not called).
        """
        if not hasattr(self, "_phase_err_per_2pi"):
            raise RuntimeError("Run the experiment first (call run() then _post_fit).")

        delta_deg = self._phase_err_per_2pi

        # Duration of one 2pi pulse
        sigma = self.cfg.get("sigma_ge")
        pulse_type = self.cfg.get("pulse_type", "arb")
        if sigma is not None and pulse_type in ("arb", "drag"):
            length_mult = self.cfg.get("length_mult", 5)
            T_2pi = 2 * sigma * length_mult       # us
        else:
            T_pi  = self.cfg.get("pi_length_ge", self.cfg.get("sigma_ge", 0.05))
            T_2pi = 2 * T_pi                      # us

        df_MHz = delta_deg / (360.0 * T_2pi)
        corrected_freq = self.cfg["qb_freq_ge"] - df_MHz

        print(f"\n--- AC Stark Calibration Result ---")
        print(f"  Phase error per 2pi pulse : {delta_deg:+.4f} deg")
        print(f"  2pi pulse duration        : {T_2pi:.4f} us")
        print(f"  Frequency offset (AC Stark): {df_MHz*1e3:+.3f} kHz")
        print(f"  Current  qb_freq_ge       : {self.cfg['qb_freq_ge']:.6f} MHz")
        print(f"  Corrected qb_freq_ge      : {corrected_freq:.6f} MHz")

        if abs(df_MHz) < 0.001:
            print("  -> Phase error < 1 kHz equivalent — no correction needed.")
        else:
            print("  -> Apply corrected_freq to cfg['qb_freq_ge'] and re-run gates.")

        return delta_deg, round(corrected_freq, 6)

    def _save_comment(self, dict_val):
        """Return a comment string including AC Stark phase error."""
        N = int(self.cfg.get("ac_stark_N", 1))
        return (
            f"AC Stark phase error = {self._phase_err_per_2pi:.4f} deg/2pi  "
            f"(N={N})\n{dict_val}"
        )
