"""
s002d — TWPA Spectroscopy vs Flux
===================================
Sweeps VNA-like frequency and flux (Yokogawa current/voltage).
Saves data in xarray/netCDF format compatible with the AI TWPA
electrical_length analysis notebook.

Expected xarray format
----------------------
Dimensions  : ifbl (flux bias, A or V), frequency (Hz)
Data vars   : magnitude (linear |S21|), phase (arg(S21), radians)
"""

import numpy as np
import xarray as xr
import os
import datetime
import matplotlib.pyplot as plt

from .base_program import BaseProgram
from .base_experiment import BaseExperiment
from ..tools.electrical_length import (
    plot_electrical_length,
    set_units_on_plot_axis,
    estimate_electrical_length,
)


class TWPAFluxProgram(BaseProgram):
    """QICK program for TWPA spectroscopy vs flux: 2D sweep."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg, prefix="ge")

        if "flux_ch" in cfg:
            self.declare_gen(ch=cfg["flux_ch"], nqz=1)
            self.add_pulse(
                ch=cfg["flux_ch"],
                name="flux_pulse",
                style="const",
                length=cfg["flux_length"],
                freq=0,
                phase=0,
                gain=cfg["flux_gain"],
            )
            self.add_loop("fluxloop", cfg["steps_flux"])

        self.add_loop("freqloop", cfg["steps"])

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)

        if "flux_ch" in cfg:
            self.pulse(ch=cfg["flux_ch"], name="flux_pulse", t=0)
            self.delay(cfg.get("saturate_times", 0.1))

        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)

        self.measure(cfg)


class TWPAFlux(BaseExperiment):
    """
    TWPA spectroscopy vs flux experiment.

    Sweeps frequency on the inner axis and flux (Yokogawa current/voltage)
    on the outer axis.  Provides :meth:`saveNetCDF` to export data in the
    xarray format expected by the AI TWPA electrical_length notebook.
    """

    EXPT_NAME = "TWPA_flux"
    TAG = "TWPA"
    X_LABEL = "Frequency (MHz)"
    Y_LABEL = "Flux (A)"
    TITLE_PREFIX = "TWPA Flux Spectroscopy"
    SWEEP_KEYS_TO_REMOVE = ["res_freq_ge", "flux_gain"]

    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6

    Y_SAVE_NAME = "Flux"
    Y_SAVE_UNIT = "A"
    Y_SAVE_SCALE = 1.0

    def _create_program(self):
        return TWPAFluxProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_pulse_param("res_pulse", "freq", as_array=True)

    def _extract_sweep_axis_y(self, prog):
        yoko_val = self.cfg.get("yoko_value")
        if yoko_val is not None:
            return np.asarray(yoko_val)
        if "flux_ch" in self.cfg:
            return prog.get_pulse_param("flux_pulse", "gain", as_array=True)
        return None

    def saveLabber(self, qb_idx, config_all=None, title=None, **kwargs):
        if self._yoko_mode == "voltage":
            self.Y_SAVE_UNIT = "V"
        elif self._yoko_mode == "current":
            self.Y_SAVE_UNIT = "A"

        super().saveLabber(qb_idx, yoko_value=None, config_all=config_all, title=title)

    def _build_s21_xarray(self):
        """Build complex S21 xr.DataArray from iqdata (ifbl × frequency)."""
        freq_hz = self._sweep_vals_x * 1e6
        ifbl = self._sweep_vals_y if self._sweep_vals_y is not None else np.array([0.0])
        return xr.DataArray(
            self.iqdata,
            dims=["ifbl", "frequency"],
            coords={"ifbl": ifbl, "frequency": freq_hz},
        )

    def plot(self, f_min=None, f_max=None, normalize=False):
        """
        Plot S21 heatmap and electrical length vs flux.

        Parameters
        ----------
        f_min, f_max : float, optional
            Frequency range in Hz for the electrical length fit.
        normalize : bool, optional
            If True, divide S21 by the zero-flux row before computing
            electrical length (removes common cable delay, shows only the
            flux-dependent variation). Default False (raw S21).
        """
        if self.iqdata is None:
            raise RuntimeError("No data. Run the experiment first.")

        s21 = self._build_s21_xarray()

        # Flux display unit
        if self._yoko_mode == "voltage":
            flux_scale, flux_unit = 1e-3, "mV"
        else:
            flux_scale, flux_unit = 1e-6, "µA"

        if normalize:
            smallest_flux = float(np.abs(s21["ifbl"]).min())
            reference = s21.sel(ifbl=smallest_flux).drop_vars("ifbl")
            s21_for_el = s21 / reference
        else:
            s21_for_el = s21

        # --- Compute electrical length for each flux point ---
        f_arr = s21["frequency"].values
        el_fit_kw = {}
        if f_min is not None:
            el_fit_kw["f_min"] = f_min
        if f_max is not None:
            el_fit_kw["f_max"] = f_max
        f_lo = el_fit_kw.get("f_min", 0)
        f_hi = el_fit_kw.get("f_max", np.inf)

        flux_vals, el_vals = [], []
        for flux, row in s21_for_el.groupby("ifbl"):
            el = estimate_electrical_length(
                f=f_arr,
                s21=row.values.reshape(-1),
                f_min=f_lo,
                f_max=f_hi,
            )
            flux_vals.append(flux)
            el_vals.append(el / 1e-9)  # → ns

        flux_vals = np.array(flux_vals)
        el_vals = np.array(el_vals)
        sort_idx = np.argsort(flux_vals)
        flux_vals, el_vals = flux_vals[sort_idx], el_vals[sort_idx]

        print(
            f"Peak-to-peak electrical length = {el_vals.max() - el_vals.min():.3f} ns"
        )

        # --- Combined plot: heatmap (x=flux, y=freq) + electrical length line ---
        fig, ax_map = plt.subplots(figsize=(8, 5))

        p = (20 * np.log10(np.abs(s21))).plot(x="ifbl", y="frequency", ax=ax_map)
        p.colorbar.ax.set_title("S21 (dB)")
        set_units_on_plot_axis(ax_map.xaxis, flux_scale, flux_unit)
        set_units_on_plot_axis(ax_map.yaxis, 1e9, "GHz", decimals=1)
        ax_map.set_title("TWPA Flux Spectroscopy")

        # Right y-axis for electrical length
        ax_el = ax_map.twinx()
        ax_el.plot(
            flux_vals,
            el_vals,
            color="dodgerblue",
            linewidth=2,
            label="Electrical length",
        )
        ax_el.axhline(el_vals.min(), color="gray", ls=":", linewidth=0.8)
        ax_el.axhline(el_vals.max(), color="gray", ls=":", linewidth=0.8)
        ax_el.set_ylabel("Electrical length (ns)", color="dodgerblue")
        ax_el.tick_params(axis="y", labelcolor="dodgerblue")
        ax_el.legend(loc="upper right")

        fig.tight_layout()
        plt.show()
        return fig

    def saveNetCDF(self, save_dir=None, filename=None):
        """
        Save normalized S21 data as netCDF for the AI TWPA electrical_length notebook.

        Normalizes by the zero-flux (smallest |ifbl|) reference before saving,
        so the file can be used directly without post-processing.

        The exported xr.Dataset has:
        - coordinate ``frequency`` in Hz
        - coordinate ``ifbl``       in A (or V if yoko_mode == "voltage")
        - data variable ``magnitude`` : linear |S21 / S21_ref|
        - data variable ``phase``     : arg(S21 / S21_ref) in radians

        Parameters
        ----------
        save_dir : str, optional
            Directory to save the file. Defaults to BaseExperiment._data_path.
        filename : str, optional
            File name (without extension). Defaults to
            ``TWPA_flux_<YYYYMMDD_HHMMSS>.nc``.
        """
        from .base_experiment import BaseExperiment, DATA_PATH

        if self.iqdata is None:
            raise RuntimeError("No data to save. Run the experiment first.")

        s21 = self._build_s21_xarray()

        # Normalize by zero-flux reference
        smallest_flux = float(np.abs(s21["ifbl"]).min())
        reference = s21.sel(ifbl=smallest_flux).drop_vars("ifbl")
        s21_norm = s21 / reference
        print(f"Normalized by ifbl = {smallest_flux:.3e} (smallest |flux|)")

        flux_unit = "V" if self._yoko_mode == "voltage" else "A"

        ds = xr.Dataset(
            {
                "magnitude": xr.apply_ufunc(np.abs, s21_norm).assign_attrs(
                    long_name="|S21/S21_ref| linear"
                ),
                "phase": xr.apply_ufunc(np.angle, s21_norm).assign_attrs(
                    long_name="arg(S21/S21_ref)", units="rad"
                ),
            },
            attrs={
                "flux_unit": flux_unit,
                "frequency_unit": "Hz",
                "yoko_mode": self._yoko_mode or "none",
                "normalized": 1,
                "reference_ifbl": smallest_flux,
            },
        )

        root = save_dir or BaseExperiment._data_path or DATA_PATH
        yy, mm, dd = datetime.datetime.today().strftime("%Y-%m-%d").split("-")
        out_dir = os.path.join(root, yy, mm, f"Data_{mm}{dd}")
        os.makedirs(out_dir, exist_ok=True)

        if filename is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"TWPA_flux_{ts}"

        path = os.path.join(out_dir, filename + ".nc")
        ds.to_netcdf(path)
        print(f"NetCDF saved to {path}")
        return path
