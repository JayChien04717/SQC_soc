"""
s002g — TWPA Gain vs Pump Power (fixed flux, fixed pump frequency)
==================================================================
Sweeps pump power only; flux and pump frequency are held constant.
Inner loop: TWPAFlux acquires QICK frequency at a single Yoko value.

Produces a 2-D xarray (pump_power × frequency) suitable for quick
saturation-power characterization at a known operating point.

Typical usage
-------------
from qick_workspace.scrip.s002g_TWPA_power_scan import TWPAPowerScan

scan = TWPAPowerScan(
    run_cfg,
    pump,
    pump_freq   = 11.3e9,          # Hz — fixed
    pump_powers = np.arange(-20, 21, 2),   # dBm sweep
    flux_value  = 0.65e-3,         # A — fixed
)
scan.YOKO_CURRENT_RAMP_STEP = 10e-6
scan.run(1, yoko_inst=yoko_connect, reference=ref)
scan.analyze(reference=ref)
"""

import numpy as np
import xarray as xr
import os
import datetime
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

try:
    from IPython.display import display as ipy_display, update_display as ipy_update

    _HAS_IPY = True
except ImportError:
    _HAS_IPY = False

import pyvisa
from .s002d_TWPA_flux import TWPAFluxProgram
from .base_experiment import BaseExperiment
from ..tools.electrical_length import set_units_on_plot_axis
from ..tools.YOKOGS200 import YOKOGS200


class TWPAPowerScan:
    """
    TWPA gain vs pump power at a fixed flux and pump frequency.

    Parameters
    ----------
    run_cfg : dict-like
        Experiment config (same as TWPAFlux).
    pump_source : AnritsuMG3692
        Pump signal generator instance.
    pump_freq : float
        Fixed pump frequency in Hz.
    pump_powers : array-like
        Pump power values to sweep, in dBm.
    flux_value : float
        Fixed Yoko output value (A for current mode, V for voltage mode).
    """

    YOKO_VOLTAGE_RAMP_STEP: float = 1e-5
    YOKO_CURRENT_RAMP_STEP: float = 1e-8
    YOKO_RAMP_INTERVAL: float = 0.01

    def __init__(self, run_cfg, pump_source, pump_freq, pump_powers, flux_value):
        self.run_cfg = run_cfg
        self.pump = pump_source
        self.pump_freq = float(pump_freq)
        self.pump_powers = np.asarray(pump_powers, dtype=float)
        self.flux_value = float(flux_value)

        self._slices = []  # list of DataArray (frequency,) per power step
        self._collected_powers = []
        self._yoko_mode = None
        self._stop = False

    def stop(self):
        """Request a graceful stop. Call from another Jupyter cell."""
        self._stop = True
        print("Stop requested — will halt after the current power step.")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        py_avg,
        yoko_inst,
        yoko_mode="current",
        temp_folder=None,
        reference=None,
        **kwargs,
    ):
        """
        Sweep pump power and acquire a frequency sweep at each step.

        Parameters
        ----------
        py_avg : int
            Software averages per QICK acquisition.
        yoko_inst : str
            VISA address of the Yokogawa source.
        yoko_mode : str
            ``"current"`` or ``"voltage"``.
        save_raw : bool
            If True, save each power step as a Labber HDF5 file.
        qb_idx : str
            Label used in the Labber filename.
        temp_folder : str, optional
            If provided, checkpoint-save netCDF after each power step.
        reference : TWPAFlux, optional
            Unpumped reference passed to saveNetCDF when temp_folder is set.
        """
        self._yoko_mode = yoko_mode
        self._slices = []
        self._collected_powers = []
        self._stop = False

        ## Setup yoko ##
        _rm = pyvisa.ResourceManager()
        _yoko = YOKOGS200(yoko_inst, _rm)
        _yoko.voltage_ramp_step = self.YOKO_VOLTAGE_RAMP_STEP
        _yoko.current_ramp_step = self.YOKO_CURRENT_RAMP_STEP
        _yoko.ramp_interval = self.YOKO_RAMP_INTERVAL
        if yoko_mode == "current":
            _yoko.SetMode("current")
            _yoko.SetCurrent(self.flux_value)
        elif yoko_mode == "voltage":
            _yoko.SetMode("voltage")
            _yoko.SetVoltage(self.flux_value)
        else:
            raise ValueError(
                f"Invalid yoko_mode: {yoko_mode}. Must be 'current' or 'voltage'."
            )
        ## setup pump ##
        self.pump.frequency = self.pump_freq
        self.pump.on()
        print(
            f"Pump ON  | freq = {self.pump_freq / 1e9:.4f} GHz"
            f" | flux = {self.flux_value * 1e3:.4f} mA"
        )

        # Precompute reference row for live gain display
        _ref_row = None
        if reference is not None:
            try:
                _ref_s21 = reference._build_s21_xarray()
                _smallest = float(np.abs(_ref_s21["ifbl"]).min())
                _ref_row = _ref_s21.sel(ifbl=_smallest).drop_vars("ifbl")
            except Exception:
                _ref_row = None

        # Live plot setup
        _cmap = plt.cm.viridis
        _cnorm = plt.Normalize(self.pump_powers.min(), self.pump_powers.max())
        _fig_live, _ax_live = plt.subplots(figsize=(9, 5))
        _ax_live.set_xlabel("Frequency (GHz)")
        _ax_live.set_ylabel("Gain (dB)" if _ref_row is not None else "Amplitude")
        _ax_live.set_title(
            f"TWPA Power Scan (running…)\n"
            f"pump_freq = {self.pump_freq / 1e9:.4f} GHz"
            f"  |  flux = {self.flux_value * 1e3:.4f} mA"
        )
        _ax_live.grid(True, alpha=0.3)
        _sm = plt.cm.ScalarMappable(cmap=_cmap, norm=_cnorm)
        _sm.set_array([])
        _fig_live.colorbar(_sm, ax=_ax_live, label="Pump power (dBm)")
        _fig_live.tight_layout()
        _live_id = f"twpa-ps-live-{np.random.randint(int(1e9))}"
        if _HAS_IPY:
            ipy_display(_fig_live, display_id=_live_id)

        # Build QICK program once — config is identical for every power step
        _soc = BaseExperiment._soc
        _prog = TWPAFluxProgram(
            BaseExperiment._soccfg,
            reps=self.run_cfg["reps"],
            final_delay=self.run_cfg["relax_delay"],
            cfg=self.run_cfg,
        )
        _freq_hz = _prog.get_pulse_param("res_pulse", "freq", as_array=True) * 1e6

        try:
            for pp in tqdm(self.pump_powers, desc="pump_power sweep"):
                if self._stop:
                    tqdm.write("Stopped by user.")
                    break

                tqdm.write(f"  pump_power = {pp:+.1f} dBm")
                self.pump.power = pp

                try:
                    iq_list = _prog.acquire(_soc, rounds=py_avg, progress=False)
                except KeyboardInterrupt:
                    tqdm.write(
                        "\nKeyboardInterrupt — saving collected data and stopping."
                    )
                    self._stop = True
                    break

                iq_data = iq_list[0][0].dot([1, 1j])  # (freq_steps,) complex
                s21_1d = xr.DataArray(
                    iq_data,
                    dims=["frequency"],
                    coords={"frequency": _freq_hz, "pump_power": pp},
                )
                self._slices.append(s21_1d)
                self._collected_powers.append(pp)

                # Update live plot
                _ax_live.cla()
                _ax_live.set_xlabel("Frequency (GHz)")
                _ax_live.set_ylabel(
                    "Gain (dB)" if _ref_row is not None else "Amplitude"
                )
                _ax_live.set_title(
                    f"TWPA Power Scan  [{len(self._collected_powers)}/{len(self.pump_powers)}]"
                    f"  —  latest: {pp:+.1f} dBm\n"
                    f"pump_freq = {self.pump_freq / 1e9:.4f} GHz"
                    f"  |  flux = {self.flux_value * 1e3:.4f} mA"
                )
                _ax_live.grid(True, alpha=0.3)
                for _pp, _sl in zip(self._collected_powers, self._slices):
                    _freq = _sl["frequency"].values / 1e9
                    if _ref_row is not None:
                        _y = 20 * np.log10(np.abs(_sl.values / _ref_row.values))
                    else:
                        _y = np.abs(_sl.values)
                    _is_latest = _pp == pp
                    _ax_live.plot(
                        _freq,
                        _y,
                        color=_cmap(_cnorm(_pp)),
                        lw=1.2 if _is_latest else 0.8,
                        alpha=1.0 if _is_latest else 0.6,
                    )
                if _HAS_IPY:
                    ipy_update(_fig_live, display_id=_live_id)
                else:
                    plt.pause(0.01)

                if temp_folder is not None:
                    self.saveNetCDF(
                        reference=reference,
                        save_dir=temp_folder,
                        filename="temp_power_scan",
                    )
                    tqdm.write(
                        f"  [checkpoint] {len(self._collected_powers)}"
                        f"/{len(self.pump_powers)} powers saved"
                    )

        finally:
            if temp_folder is not None and self._slices:
                self.saveNetCDF(
                    reference=reference,
                    save_dir=temp_folder,
                    filename="temp_power_scan",
                )
                print(
                    f"[final save] {len(self._collected_powers)}"
                    f"/{len(self.pump_powers)} power steps → {temp_folder}"
                )

    # ------------------------------------------------------------------
    # Build 2-D xarray
    # ------------------------------------------------------------------

    def _build_xarray(self):
        """Return 2-D DataArray (pump_power × frequency)."""
        if not self._slices:
            raise RuntimeError("No data. Run the experiment first.")
        return xr.concat(self._slices, dim="pump_power")  # (pump_power, frequency)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def saveNetCDF(self, reference=None, save_dir=None, filename=None):
        """Save gain data as netCDF.

        Returns
        -------
        gain_path : str
        ref_path : str or None
        """
        da = self._build_xarray()

        ds = xr.Dataset(
            {
                "magnitude": xr.apply_ufunc(np.abs, da).assign_attrs(
                    long_name="|S21| linear"
                ),
                "phase": xr.apply_ufunc(np.angle, da).assign_attrs(
                    long_name="arg(S21)", units="rad"
                ),
            },
            attrs={
                "pump_freq": self.pump_freq,
                "flux_value": self.flux_value,
                "pump_state": 1,
                "yoko_mode": self._yoko_mode or "current",
                "frequency_unit": "Hz",
                "flux_unit": "V" if self._yoko_mode == "voltage" else "A",
            },
        )
        ds = ds.assign_coords(
            pump_freq=self.pump_freq,
            flux_value=self.flux_value,
        )

        root = save_dir or BaseExperiment._data_path or "."
        yy, mm, dd = datetime.datetime.today().strftime("%Y-%m-%d").split("-")
        out_dir = os.path.join(root, yy, mm, f"Data_{mm}{dd}")
        os.makedirs(out_dir, exist_ok=True)

        if filename is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"TWPA_power_scan_{ts}"

        gain_path = os.path.join(out_dir, filename + ".nc")
        ds.to_netcdf(gain_path)
        print(f"Power scan saved to {gain_path}")

        ref_path = None
        if reference is not None:
            ref_s21 = reference._build_s21_xarray()
            smallest_flux = float(np.abs(ref_s21["ifbl"]).min())
            ref_row = ref_s21.sel(ifbl=smallest_flux).drop_vars("ifbl")

            ds_ref = xr.Dataset(
                {
                    "magnitude": xr.apply_ufunc(np.abs, ref_row).assign_attrs(
                        long_name="|S21| linear"
                    ),
                    "phase": xr.apply_ufunc(np.angle, ref_row).assign_attrs(
                        long_name="arg(S21)", units="rad"
                    ),
                },
                attrs={"pump_state": 0, "ifbl": smallest_flux},
            )
            ds_ref = ds_ref.assign_coords(pump_state=0, ifbl=smallest_flux)
            ref_path = os.path.join(out_dir, filename + "_reference.nc")
            ds_ref.to_netcdf(ref_path)
            print(f"Reference saved to {ref_path}")

        return gain_path, ref_path

    # ------------------------------------------------------------------
    # Analyze
    # ------------------------------------------------------------------

    def analyze(
        self,
        reference,
        f_min=4e9,
        f_max=8e9,
        gain_min=10,
        freq_exclude=None,
    ):
        """
        Normalize by reference, plot gain vs frequency for each pump power.

        Parameters
        ----------
        reference : TWPAFlux
            Unpumped TWPAFlux measurement (pump OFF).
        f_min, f_max : float
            Frequency range in Hz to highlight.
        gain_min : float
            Minimum gain dB line drawn on plot for reference.
        freq_exclude : list of (float, float), optional
            Frequency bands to shade/mask on the plot.

        Returns
        -------
        gain_db : xr.DataArray
            Gain in dB, shape (pump_power × frequency).
        """
        da = self._build_xarray()  # (pump_power, frequency)

        # Normalize by unpumped zero-flux row
        ref_s21 = reference._build_s21_xarray()
        smallest_flux = float(np.abs(ref_s21["ifbl"]).min())
        ref_row = ref_s21.sel(ifbl=smallest_flux).drop_vars("ifbl")  # (frequency,)

        gain_norm = da / ref_row  # broadcast over pump_power
        gain_db = 20 * np.log10(np.abs(gain_norm))  # (pump_power, frequency)

        freq_ghz = gain_db["frequency"].values / 1e9
        powers = gain_db["pump_power"].values
        freq = gain_db["frequency"].values
        band_mask = (freq >= f_min) & (freq <= f_max)

        # Precompute median gain within band for each power
        medians = {}
        for pp in powers:
            g_band = gain_db.sel(pump_power=pp).values[band_mask]
            medians[pp] = float(np.nanmedian(g_band)) if g_band.size else float("nan")

        cmap = plt.cm.viridis
        cnorm = plt.Normalize(powers.min(), powers.max())

        fig, ax = plt.subplots(figsize=(9, 5))

        for pp in powers:
            curve = gain_db.sel(pump_power=pp).values
            lbl = f"{pp:.0f} dBm  median {medians[pp]:.1f} dB"
            ax.plot(freq_ghz, curve, color=cmap(cnorm(pp)), lw=0.9, label=lbl)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)
        fig.colorbar(sm, ax=ax, label="Pump power (dBm)")

        ax.axhline(gain_min, color="red", ls="--", lw=0.8, label=f"{gain_min} dB line")
        ax.axvline(f_min / 1e9, color="gray", ls="--", lw=0.8)
        ax.axvline(f_max / 1e9, color="gray", ls="--", lw=0.8)

        if freq_exclude:
            for f_lo, f_hi in freq_exclude:
                ax.axvspan(f_lo / 1e9, f_hi / 1e9, alpha=0.15, color="gray")

        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Gain (dB)")
        ax.set_title(
            f"TWPA Gain vs Pump Power\n"
            f"pump_freq = {self.pump_freq / 1e9:.4f} GHz"
            f"  |  flux = {self.flux_value * 1e3:.4f} mA"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()

        # Print gain summary (band_mask and medians already computed above)
        print(
            f"\n=== Gain summary  (pump_freq = {self.pump_freq / 1e9:.4f} GHz,"
            f"  flux = {self.flux_value * 1e3:.4f} mA) ==="
        )
        print(
            f"  {'Power (dBm)':>12}  {'Peak gain (dB)':>15}  {'Median gain (dB)':>17}"
        )
        print("-" * 50)
        for pp in powers:
            g_band = gain_db.sel(pump_power=pp).values[band_mask]
            peak = float(np.nanmax(g_band)) if g_band.size else float("nan")
            print(f"  {pp:>12.1f}  {peak:>15.2f}  {medians[pp]:>17.2f}")

        return gain_db
