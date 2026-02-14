from .base_program import BaseProgram
from .base_experiment import BaseExperiment
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import hdf5_generator, get_next_filename_labber, config_to_yaml
from ..plotter.liveplot import liveplotfun


# ── Program ──

class PunchoutProgram(BaseProgram):
    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.add_loop("gainloop", cfg["g_steps"])
        self.add_loop("freqloop", cfg["f_steps"])

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.measure(cfg)


# ── Experiment ──

class Punchout(BaseExperiment):
    """Resonator punchout: 2D sweep over gain and frequency."""
    
    EXPT_NAME = "s002b_res_ge_punchout"
    TAG = "OneTone"
    X_LABEL = "Frequency (MHz)"
    Y_LABEL = "DAC Gains"
    TITLE_PREFIX = "Resonator Punchout"
    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6
    Y_SAVE_NAME = "DAC Gains"
    Y_SAVE_UNIT = "a.u."
    Y_SAVE_SCALE = 1.0

    def _create_program(self):
        return PunchoutProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_pulse_param("res_pulse", "freq", as_array=True)

    def run(self, py_avg, liveplot=True, simulate=False):
        # ── Simulate mode ──
        if simulate:
            return self._run_simulate()

        # ── Hardware mode ──
        if liveplot:
            self.liveplot(py_avg)
        else:
            prog = self._create_program()
            self.iq_list = prog.acquire(self.soc, rounds=py_avg, progress=True)
            self.iqdata = self.iq_list[0][0].dot([1, 1j])
            self.freqs = prog.get_pulse_param("res_pulse", "freq", as_array=True)
            self.gains = prog.get_pulse_param("res_pulse", "gain", as_array=True)
            
            # Prepare data for saveLabber
            self._sweep_vals = self.freqs
            self._sweep_vals_y = self.gains

    def _run_simulate(self):
        """Generate mock 2D punchout data without hardware."""
        from .mock_signals import mock_lorentzian_2d
        
        # Determine frequency (x) axis
        res_freq = self.cfg.get("res_freq_ge")
        if hasattr(res_freq, "start"):
            self.freqs = np.linspace(res_freq.start, res_freq.stop, self.cfg.get("f_steps", 51))
        else:
            self.freqs = np.linspace(6000, 6100, self.cfg.get("f_steps", 51))

        # Determine gain (y) axis
        res_gain = self.cfg.get("res_gain_ge")
        if hasattr(res_gain, "start"):
            self.gains = np.linspace(res_gain.start, res_gain.stop, self.cfg.get("g_steps", 11))
        else:
            self.gains = np.linspace(0, 1, self.cfg.get("g_steps", 11))

        self.iqdata = mock_lorentzian_2d(self.freqs, self.gains,
                                          f0=(self.freqs[0] + self.freqs[-1]) / 2)
        
        # Prepare data for saveLabber
        self._sweep_vals = self.freqs
        self._sweep_vals_y = self.gains

        # Plot
        data = np.abs(self.iqdata)
        data_norm = np.array([
            (row - row.min()) / (row.max() - row.min())
            if row.max() != row.min() else row
            for row in data
        ])
        plt.figure(figsize=(8, 5))
        pcm = plt.pcolormesh(self.freqs, self.gains, data_norm)
        plt.title(f"{self.TITLE_PREFIX} [SIMULATED]")
        plt.xlabel(self.X_LABEL)
        plt.ylabel(f"{self.Y_LABEL} [a.u.]")
        plt.colorbar(pcm)
        plt.show()

    def liveplot(self, py_avg):
        prog = self._create_program()
        self.freqs = prog.get_pulse_param("res_pulse", "freq", as_array=True)
        self.gains = prog.get_pulse_param("res_pulse", "gain", as_array=True)

        self.iqdata, interrupted, avg_count = liveplotfun(
            prog=prog, soc=self.soc, py_avg=py_avg,
            x_axis_vals=self.freqs, y_axis_vals=self.gains,
            x_label=self.X_LABEL, y_label=self.Y_LABEL,
            title_prefix=self.TITLE_PREFIX, show_final_plot=False,
        )
        if self.iqdata is None:
            print("No data acquired.")
        elif interrupted:
            print(f"Interrupted at {avg_count} averages (data is partial).")
            
        # Prepare data for saveLabber (even if interrupted)
        self._sweep_vals = self.freqs
        self._sweep_vals_y = self.gains

    def plot(self):
        data = np.abs(self.iqdata)
        data_norm = np.array([
            (row - row.min()) / (row.max() - row.min())
            if row.max() != row.min() else row
            for row in data
        ])
        pcm = plt.pcolormesh(self.freqs, self.gains, data_norm)
        plt.title(self.TITLE_PREFIX)
        plt.xlabel(self.X_LABEL)
        plt.ylabel(f"{self.Y_LABEL} [a.u.]")
        plt.colorbar(pcm)
