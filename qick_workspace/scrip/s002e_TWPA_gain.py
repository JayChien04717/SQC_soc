"""
s002e — TWPA Gain Scan (pump_freq × flux)
==========================================
Outer loop: sweep pump frequency via Anritsu MG3692.
Inner loop: TWPAFlux sweeps QICK frequency × Yoko flux.

Produces a 3-D xarray (frequency × pump_freq × ifbl) compatible with
the AI TWPA scoring.ipynb notebook.

Typical usage
-------------
from qick_workspace.scrip.s002e_TWPA_gain import TWPAGain
from qick_workspace.tools.mg3692 import AnritsuMG3692

pump = AnritsuMG3692("192.168.10.182")
ref  = TWPAFlux(run_cfg_unpumped)          # pump OFF, already run
ref.run(10, yoko_inst=yoko, yoko_value=yoko_range)

gain = TWPAGain(run_cfg, pump, pump_freqs=np.linspace(10.5e9, 11.0e9, 11), pump_power=-10)
gain.run(10, yoko_inst=yoko, yoko_value=yoko_range)
gain.saveNetCDF(reference=ref)
"""

import numpy as np
import xarray as xr
import os
import datetime
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from .s002d_TWPA_flux import TWPAFlux
from .base_experiment import BaseExperiment
from ..tools.scoring import (
    score_ai_twpa_c_gain_data,
    find_best_operation_point,
    plot_gain_at_operation_point,
    plot_operation_point_parameters,
)
from ..tools.electrical_length import set_units_on_plot_axis


class TWPAGain:
    """
    TWPA gain scan: sweeps pump frequency (MG3692) × Yoko flux × QICK frequency.

    Parameters
    ----------
    run_cfg : dict-like
        Experiment config (same as TWPAFlux).
    pump_source : AnritsuMG3692
        Pump signal generator instance.
    pump_freqs : array-like
        Pump frequencies to sweep, in Hz.
    pump_power : float
        Pump output power in dBm. Default -10.
    """

    YOKO_VOLTAGE_RAMP_STEP: float = 1e-5
    YOKO_CURRENT_RAMP_STEP: float = 1e-8
    YOKO_RAMP_INTERVAL: float = 0.01

    def __init__(self, run_cfg, pump_source, pump_freqs, pump_power=-10):
        self.run_cfg = run_cfg
        self.pump = pump_source
        self.pump_freqs = np.asarray(pump_freqs)
        self.pump_power = pump_power

        self._slices = []  # list of xr.DataArray per pump_freq
        self._yoko_mode = None
        self._stop = False

    def stop(self):
        """Request a graceful stop. Call from another Jupyter cell."""
        self._stop = True
        print("Stop requested — will halt after the current pump_freq step.")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        py_avg,
        yoko_inst,
        yoko_value,
        yoko_mode="current",
        save_raw=False,
        qb_idx="TWPA",
        temp_folder=None,
        reference=None,
        **kwargs,
    ):
        """
        Sweep pump frequency and acquire TWPAFlux data at each step.

        Parameters
        ----------
        py_avg : int
            Software averages per QICK acquisition.
        yoko_inst : str
            VISA address of the Yokogawa source.
        yoko_value : array-like
            Flux sweep values (A or V).
        yoko_mode : str
            ``"current"`` or ``"voltage"``.
        save_raw : bool
            If True, save each pump-freq slice as a Labber HDF5 file.
        qb_idx : str
            Label used in the Labber filename. Default ``"TWPA"``.
        temp_folder : str, optional
            If provided, auto-save gain + reference netCDF to this folder
            immediately after the run completes (for quick offline analysis).
        reference : TWPAFlux, optional
            Unpumped reference passed to saveNetCDF when temp_folder is set.
        """
        self._yoko_mode = yoko_mode
        self._slices = []
        self._stop = False

        self.pump.power = self.pump_power
        self.pump.on()
        print(f"Pump ON  | power = {self.pump_power} dBm")

        try:
            for pf in tqdm(self.pump_freqs, desc="pump_freq sweep"):
                if self._stop:
                    tqdm.write("Stopped by user.")
                    break

                tqdm.write(f"  pump_freq = {pf / 1e9:.4f} GHz")
                self.pump.frequency = pf

                try:
                    exp = TWPAFlux(self.run_cfg)
                    exp.YOKO_VOLTAGE_RAMP_STEP = self.YOKO_VOLTAGE_RAMP_STEP
                    exp.YOKO_CURRENT_RAMP_STEP = self.YOKO_CURRENT_RAMP_STEP
                    exp.YOKO_RAMP_INTERVAL = self.YOKO_RAMP_INTERVAL
                    exp.run(
                        py_avg,
                        yoko_inst=yoko_inst,
                        yoko_value=yoko_value,
                        yoko_mode=yoko_mode,
                    )
                except KeyboardInterrupt:
                    tqdm.write(
                        "\nKeyboardInterrupt — saving collected data and stopping."
                    )
                    self._stop = True

                if not self._stop:
                    s21 = exp._build_s21_xarray()  # (ifbl, frequency)
                    s21 = s21.assign_coords(pump_freq=pf)  # add scalar coord
                    self._slices.append(s21)

                    if save_raw:
                        title = f"pumpfreq_{pf / 1e9:.2f}GHz_power_{self.pump_power:+.1f}dBm"
                        exp.saveLabber(qb_idx, title=title)

                if self._stop:
                    break

        finally:
            # self.pump.off()
            # print("Pump OFF")
            if temp_folder is not None and self._slices:
                self.saveNetCDF(
                    reference=reference, save_dir=temp_folder, filename="temp_gain"
                )
                print(
                    f"[saved] {len(self._slices)}/{len(self.pump_freqs)} pump_freq steps"
                    f" → {temp_folder}"
                )

    # ------------------------------------------------------------------
    # Build 3-D xarray
    # ------------------------------------------------------------------

    def _build_gain_xarray(self):
        """Return 3-D DataArray (pump_freq × ifbl × frequency)."""
        if not self._slices:
            raise RuntimeError("No data. Run the experiment first.")
        return xr.concat(self._slices, dim="pump_freq")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def saveNetCDF(self, reference=None, save_dir=None, filename=None):
        """
        Save gain data as netCDF for scoring.ipynb.

        Parameters
        ----------
        reference : TWPAFlux, optional
            Unpumped TWPAFlux result used as normalization reference.
            If None, no reference file is saved separately.
        save_dir : str, optional
            Output directory. Defaults to BaseExperiment._data_path.
        filename : str, optional
            Base filename (no extension). Defaults to
            ``TWPA_gain_<YYYYMMDD_HHMMSS>``.

        Returns
        -------
        gain_path : str
            Path of the saved gain netCDF.
        ref_path : str or None
            Path of the saved reference netCDF (None if reference not provided).
        """
        gain = self._build_gain_xarray()

        # scoring.ipynb expects dims (frequency, pump_freq, ifbl)
        gain = gain.transpose("frequency", "pump_freq", "ifbl")

        ds_gain = xr.Dataset(
            {
                "magnitude": xr.apply_ufunc(np.abs, gain).assign_attrs(
                    long_name="|S21| linear"
                ),
                "phase": xr.apply_ufunc(np.angle, gain).assign_attrs(
                    long_name="arg(S21)", units="rad"
                ),
            },
            attrs={
                "pump_power": self.pump_power,
                "pump_state": 1,
                "yoko_mode": self._yoko_mode or "current",
                "frequency_unit": "Hz",
                "flux_unit": "V" if self._yoko_mode == "voltage" else "A",
            },
        )
        # Add scalar coords expected by scoring.ipynb
        ds_gain = ds_gain.assign_coords(
            pump_power=self.pump_power,
            pump_state=1,
        )

        root = save_dir or BaseExperiment._data_path or "."
        yy, mm, dd = datetime.datetime.today().strftime("%Y-%m-%d").split("-")
        out_dir = os.path.join(root, yy, mm, f"Data_{mm}{dd}")
        os.makedirs(out_dir, exist_ok=True)

        if filename is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"TWPA_gain_{ts}"

        gain_path = os.path.join(out_dir, filename + ".nc")
        ds_gain.to_netcdf(gain_path)
        print(f"Gain data saved to {gain_path}")

        ref_path = None
        if reference is not None:
            ref_s21 = reference._build_s21_xarray()  # (ifbl, frequency)

            # Use zero-flux row as reference (pump OFF, ifbl ≈ 0)
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
            ds_ref = ds_ref.assign_coords(
                pump_state=0,
                ifbl=smallest_flux,
                pump_power=self.pump_power,
                pump_freq=float("nan"),
            )

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
        gain_min=12,
        gain_median=15,
        ripple_max=5,
        f_min=4e9,
        f_max=8e9,
        n_best=5,
        exclusion_radius=20,
        freq_exclude=None,
    ):
        """
        Normalize gain, score, and plot results inline.

        Parameters
        ----------
        reference : TWPAFlux
            Unpumped TWPAFlux measurement (pump OFF) used as reference.
        gain_min : float
            Minimum gain threshold in dB for scoring.
        gain_median : float
            Target median gain in dB for scoring.
        ripple_max : float
            Max allowed pk-to-pk gain ripple in dB for scoring.
        f_min, f_max : float
            Frequency range in Hz for scoring.
        n_best : int
            Number of top operation points to find and plot.
        exclusion_radius : float
            Exclusion radius around found points (see scoring.py).
        freq_exclude : list of (float, float), optional
            Frequency bands to mask before scoring, e.g.
            [(4.55e9, 4.65e9), (5.75e9, 5.85e9)].

        Returns
        -------
        best_points : list
            Top operation points as xr.DataArray.
        total_score : xr.DataArray
            Score heatmap (pump_freq × ifbl).
        """
        gain = self._build_gain_xarray()  # (pump_freq × ifbl × frequency)

        # Normalize by unpumped zero-flux reference
        ref_s21 = reference._build_s21_xarray()  # (ifbl × frequency)
        smallest_flux = float(np.abs(ref_s21["ifbl"]).min())
        ref_row = ref_s21.sel(ifbl=smallest_flux).drop_vars("ifbl")  # (frequency,)
        gain_normalized = gain / ref_row  # broadcast over pump_freq & ifbl

        # Transpose to (frequency × pump_freq × ifbl) for scoring functions
        gain_normalized = gain_normalized.transpose("frequency", "pump_freq", "ifbl")

        # Apply frequency exclusion masks before scoring
        if freq_exclude:
            f = gain_normalized["frequency"]
            mask = xr.ones_like(f, dtype=bool)
            for f_lo, f_hi in freq_exclude:
                mask = mask & ~((f >= f_lo) & (f <= f_hi))
            gain_scored = gain_normalized.where(mask)
            print(
                f"Excluded: {[f'{a / 1e9:.3f}–{b / 1e9:.3f} GHz' for a, b in freq_exclude]}"
            )
        else:
            gain_scored = gain_normalized

        # Score
        total_score = score_ai_twpa_c_gain_data(
            gain_data=gain_scored,
            gain_min=gain_min,
            gain_median=gain_median,
            ripple_max=ripple_max,
            f_min=f_min,
            f_max=f_max,
        )

        # Flux display unit
        flux_scale = 1e-6
        flux_unit = "µA"

        # --- Plot 1: score heatmap + best points ---
        best_points = []
        for _ in range(n_best):
            pt = find_best_operation_point(
                total_score,
                excluded_points=best_points,
                exclusion_radius=exclusion_radius,
            )
            if pt is None:
                break
            best_points.append(pt)

        fig1, ax1 = plt.subplots(figsize=(7, 5))
        plot_operation_point_parameters(total_score, best_points, ax=ax1)
        set_units_on_plot_axis(ax1.xaxis, flux_scale, flux_unit)
        set_units_on_plot_axis(ax1.yaxis, 1e9, "GHz", decimals=1)
        ax1.set_title("Score heatmap (pump_freq × ifbl)")
        fig1.tight_layout()

        # --- Plot 2: gain curves at best points ---
        if not best_points:
            print(
                "No operation points found. Try lowering gain_min or adjusting scoring parameters."
            )
        else:
            ncols = 2
            nrows = max(1, int(np.ceil(len(best_points) / ncols)))
            fig2, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
            axes = np.array(axes).reshape(-1)
            for i, op in enumerate(best_points):
                plot_gain_at_operation_point(gain_normalized, op=op, ax=axes[i])
                pf_ghz = float(op["pump_freq"]) / 1e9
                ib_ua = float(op["ifbl"]) / 1e-6
                axes[i].set_title(
                    f"#{i + 1}  pump={pf_ghz:.3f} GHz  ifbl={ib_ua:.1f} µA"
                )
                set_units_on_plot_axis(axes[i].xaxis, 1e9, "GHz", decimals=1)
            for ax in axes[len(best_points) :]:
                ax.set_visible(False)
            fig2.suptitle("Best operation points", fontsize=13)
            fig2.tight_layout()

        plt.show()

        # Print summary
        print("\n=== Best operation points ===")
        for i, op in enumerate(best_points):
            print(
                f"  #{i + 1}  pump_freq = {float(op['pump_freq']) / 1e9:.4f} GHz"
                f"  |  ifbl = {float(op['ifbl']) / 1e-6:.2f} µA"
                f"  |  score = {float(op):.3f}"
            )

        return best_points, total_score
