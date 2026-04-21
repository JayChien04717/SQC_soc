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

from .s002d_TWPA_flux import TWPAFlux
from .base_experiment import BaseExperiment
from ..tools.electrical_length import set_units_on_plot_axis


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

        self._slices = []          # list of DataArray (frequency,) per power step
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
        save_raw=False,
        qb_idx="TWPA",
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

        yoko_arr = np.array([self.flux_value])

        self.pump.frequency = self.pump_freq
        self.pump.on()
        print(
            f"Pump ON  | freq = {self.pump_freq / 1e9:.4f} GHz"
            f" | flux = {self.flux_value * 1e3:.4f} mA"
        )

        try:
            for pp in tqdm(self.pump_powers, desc="pump_power sweep"):
                if self._stop:
                    tqdm.write("Stopped by user.")
                    break

                tqdm.write(f"  pump_power = {pp:+.1f} dBm")
                self.pump.power = pp

                try:
                    exp = TWPAFlux(self.run_cfg)
                    exp.YOKO_VOLTAGE_RAMP_STEP = self.YOKO_VOLTAGE_RAMP_STEP
                    exp.YOKO_CURRENT_RAMP_STEP = self.YOKO_CURRENT_RAMP_STEP
                    exp.YOKO_RAMP_INTERVAL = self.YOKO_RAMP_INTERVAL
                    exp.run(
                        py_avg,
                        yoko_inst=yoko_inst,
                        yoko_value=yoko_arr,
                        yoko_mode=yoko_mode,
                    )
                except KeyboardInterrupt:
                    tqdm.write("\nKeyboardInterrupt — saving collected data and stopping.")
                    self._stop = True
                    break

                # _build_s21_xarray returns (ifbl, frequency); squeeze single flux point
                s21 = exp._build_s21_xarray()  # (ifbl=1, frequency)
                s21_1d = s21.isel(ifbl=0).drop_vars("ifbl")  # (frequency,)
                s21_1d = s21_1d.assign_coords(pump_power=pp)
                self._slices.append(s21_1d)
                self._collected_powers.append(pp)

                if save_raw:
                    title = f"power_{pp:+.1f}dBm"
                    exp.saveLabber(qb_idx, title=title)

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

        cmap = plt.cm.viridis
        cnorm = plt.Normalize(powers.min(), powers.max())

        fig, ax = plt.subplots(figsize=(9, 5))

        for pp in powers:
            curve = gain_db.sel(pump_power=pp).values
            ax.plot(freq_ghz, curve, color=cmap(cnorm(pp)), lw=0.9)

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
        set_units_on_plot_axis(ax.xaxis, 1e9, "GHz", decimals=1)
        fig.tight_layout()
        plt.show()

        # Print gain at band edges and peak for each power
        freq = gain_db["frequency"].values
        band_mask = (freq >= f_min) & (freq <= f_max)
        print(f"\n=== Gain summary  (pump_freq = {self.pump_freq / 1e9:.4f} GHz,"
              f"  flux = {self.flux_value * 1e3:.4f} mA) ===")
        print(f"  {'Power (dBm)':>12}  {'Peak gain (dB)':>15}  {'Median gain (dB)':>17}")
        print("-" * 50)
        for pp in powers:
            g = gain_db.sel(pump_power=pp).values
            g_band = g[band_mask]
            peak = float(np.nanmax(g_band)) if g_band.size else float("nan")
            median = float(np.nanmedian(g_band)) if g_band.size else float("nan")
            print(f"  {pp:>12.1f}  {peak:>15.2f}  {median:>17.2f}")

        return gain_db
