"""
Resonator/twpa — s002d-g: TWPA characterization experiments.
"""

from __future__ import annotations

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

from ...core.base_program import BaseProgram
from ...core.base_experiment import BaseExperiment
from ...config.system_cfg import DATA_PATH
from ...tools.scoring import (
    score_ai_twpa_c_gain_data,
    find_best_operation_point,
    plot_gain_at_operation_point,
    plot_operation_point_parameters,
)
from ...tools.electrical_length import (
    plot_electrical_length,
    set_units_on_plot_axis,
    estimate_electrical_length,
)
from ...instruments.YOKOGS200 import YOKOGS200


# ── s002d: TWPAFlux ───────────────────────────────────────────────────────────

class TWPAFluxProgram(BaseProgram):
    """QICK program for TWPA spectroscopy vs flux: 2D sweep."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg, prefix="ge")
        if "flux_ch" in cfg:
            self.declare_gen(ch=cfg["flux_ch"], nqz=1)
            self.add_pulse(ch=cfg["flux_ch"], name="flux_pulse", style="const",
                           length=cfg["flux_length"], freq=0, phase=0, gain=cfg["flux_gain"])
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
    """TWPA spectroscopy vs flux experiment."""

    EXPT_NAME = "TWPA_flux"
    TAG = "TWPA"
    X_LABEL = "Frequency (MHz)"
    Y_LABEL = "Flux"
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
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
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
        freq_hz = self._sweep_vals_x * 1e6
        ifbl = self._sweep_vals_y if self._sweep_vals_y is not None else np.array([0.0])
        return xr.DataArray(
            self.iqdata,
            dims=["ifbl", "frequency"],
            coords={"ifbl": ifbl, "frequency": freq_hz},
        )

    def plot(self, f_min=None, f_max=None, normalize=False):
        if self.iqdata is None:
            raise RuntimeError("No data. Run the experiment first.")
        s21 = self._build_s21_xarray()
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
            el = estimate_electrical_length(f=f_arr, s21=row.values.reshape(-1),
                                            f_min=f_lo, f_max=f_hi)
            flux_vals.append(flux)
            el_vals.append(el / 1e-9)
        flux_vals = np.array(flux_vals)
        el_vals = np.array(el_vals)
        sort_idx = np.argsort(flux_vals)
        flux_vals, el_vals = flux_vals[sort_idx], el_vals[sort_idx]
        print(f"Peak-to-peak electrical length = {el_vals.max() - el_vals.min():.3f} ns")
        fig, ax_map = plt.subplots(figsize=(8, 5))
        p = (20 * np.log10(np.abs(s21))).plot(x="ifbl", y="frequency", ax=ax_map)
        p.colorbar.ax.set_title("S21 (dB)")
        set_units_on_plot_axis(ax_map.xaxis, flux_scale, flux_unit)
        set_units_on_plot_axis(ax_map.yaxis, 1e9, "GHz", decimals=1)
        ax_map.set_title("TWPA Flux Spectroscopy")
        ax_el = ax_map.twinx()
        ax_el.plot(flux_vals, el_vals, color="dodgerblue", linewidth=2, label="Electrical length")
        ax_el.axhline(el_vals.min(), color="gray", ls=":", linewidth=0.8)
        ax_el.axhline(el_vals.max(), color="gray", ls=":", linewidth=0.8)
        ax_el.set_ylabel("Electrical length (ns)", color="dodgerblue")
        ax_el.tick_params(axis="y", labelcolor="dodgerblue")
        ax_el.legend(loc="upper right")
        fig.tight_layout()
        plt.show()
        return fig

    def saveNetCDF(self, save_dir=None, filename=None):
        if self.iqdata is None:
            raise RuntimeError("No data to save. Run the experiment first.")
        s21 = self._build_s21_xarray()
        smallest_flux = float(np.abs(s21["ifbl"]).min())
        reference = s21.sel(ifbl=smallest_flux).drop_vars("ifbl")
        s21_norm = s21 / reference
        print(f"Normalized by ifbl = {smallest_flux:.3e} (smallest |flux|)")
        flux_unit = "V" if self._yoko_mode == "voltage" else "A"
        ds = xr.Dataset(
            {
                "magnitude": xr.apply_ufunc(np.abs, s21_norm).assign_attrs(long_name="|S21/S21_ref| linear"),
                "phase": xr.apply_ufunc(np.angle, s21_norm).assign_attrs(long_name="arg(S21/S21_ref)", units="rad"),
            },
            attrs={"flux_unit": flux_unit, "frequency_unit": "Hz",
                   "yoko_mode": self._yoko_mode or "none", "normalized": 1,
                   "reference_ifbl": smallest_flux},
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


# ── s002e: TWPAGain ───────────────────────────────────────────────────────────

class TWPAGain:
    """TWPA gain scan: sweeps pump frequency × Yoko flux × QICK frequency."""

    YOKO_VOLTAGE_RAMP_STEP: float = 1e-5
    YOKO_CURRENT_RAMP_STEP: float = 1e-8
    YOKO_RAMP_INTERVAL: float = 0.01

    def __init__(self, run_cfg, pump_source, pump_freqs, pump_power=-10):
        self.run_cfg = run_cfg
        self.pump = pump_source
        self.pump_freqs = np.asarray(pump_freqs)
        self.pump_power = pump_power
        self._slices = []
        self._yoko_mode = None
        self._stop = False

    def stop(self):
        self._stop = True
        print("Stop requested — will halt after the current pump_freq step.")

    def run(self, py_avg, yoko_inst, yoko_value, yoko_mode="current",
            save_raw=False, qb_idx="TWPA", temp_folder=None, reference=None, **kwargs):
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
                    exp.run(py_avg, yoko_inst=yoko_inst, yoko_value=yoko_value, yoko_mode=yoko_mode)
                except KeyboardInterrupt:
                    tqdm.write("\nKeyboardInterrupt — saving collected data and stopping.")
                    self._stop = True
                if not self._stop:
                    s21 = exp._build_s21_xarray()
                    s21 = s21.assign_coords(pump_freq=pf)
                    self._slices.append(s21)
                    if save_raw:
                        title = f"pumpfreq_{pf / 1e9:.2f}GHz_power_{self.pump_power:+.1f}dBm"
                        exp.saveLabber(qb_idx, title=title)
                if self._stop:
                    break
        finally:
            if temp_folder is not None and self._slices:
                self.saveNetCDF(reference=reference, save_dir=temp_folder, filename="temp_gain")
                print(f"[saved] {len(self._slices)}/{len(self.pump_freqs)} pump_freq steps → {temp_folder}")

    def _build_gain_xarray(self):
        if not self._slices:
            raise RuntimeError("No data. Run the experiment first.")
        return xr.concat(self._slices, dim="pump_freq")

    def saveNetCDF(self, reference=None, save_dir=None, filename=None):
        gain = self._build_gain_xarray()
        gain = gain.transpose("frequency", "pump_freq", "ifbl")
        ds_gain = xr.Dataset(
            {
                "magnitude": xr.apply_ufunc(np.abs, gain).assign_attrs(long_name="|S21| linear"),
                "phase": xr.apply_ufunc(np.angle, gain).assign_attrs(long_name="arg(S21)", units="rad"),
            },
            attrs={"pump_power": self.pump_power, "pump_state": 1,
                   "yoko_mode": self._yoko_mode or "current",
                   "frequency_unit": "Hz",
                   "flux_unit": "V" if self._yoko_mode == "voltage" else "A"},
        )
        ds_gain = ds_gain.assign_coords(pump_power=self.pump_power, pump_state=1)
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
            ref_s21 = reference._build_s21_xarray()
            smallest_flux = float(np.abs(ref_s21["ifbl"]).min())
            ref_row = ref_s21.sel(ifbl=smallest_flux).drop_vars("ifbl")
            ds_ref = xr.Dataset(
                {"magnitude": xr.apply_ufunc(np.abs, ref_row).assign_attrs(long_name="|S21| linear"),
                 "phase": xr.apply_ufunc(np.angle, ref_row).assign_attrs(long_name="arg(S21)", units="rad")},
                attrs={"pump_state": 0, "ifbl": smallest_flux},
            )
            ds_ref = ds_ref.assign_coords(pump_state=0, ifbl=smallest_flux,
                                           pump_power=self.pump_power, pump_freq=float("nan"))
            ref_path = os.path.join(out_dir, filename + "_reference.nc")
            ds_ref.to_netcdf(ref_path)
            print(f"Reference saved to {ref_path}")
        return gain_path, ref_path

    def analyze(self, reference, gain_min=12, gain_median=15, ripple_max=5,
                f_min=4e9, f_max=8e9, n_best=5, exclusion_radius=20, freq_exclude=None):
        gain = self._build_gain_xarray()
        ref_s21 = reference._build_s21_xarray()
        smallest_flux = float(np.abs(ref_s21["ifbl"]).min())
        ref_row = ref_s21.sel(ifbl=smallest_flux).drop_vars("ifbl")
        gain_normalized = gain / ref_row
        gain_normalized = gain_normalized.transpose("frequency", "pump_freq", "ifbl")
        if freq_exclude:
            f = gain_normalized["frequency"]
            mask = xr.ones_like(f, dtype=bool)
            for f_lo, f_hi in freq_exclude:
                mask = mask & ~((f >= f_lo) & (f <= f_hi))
            gain_scored = gain_normalized.where(mask)
            print(f"Excluded: {[f'{a / 1e9:.3f}–{b / 1e9:.3f} GHz' for a, b in freq_exclude]}")
        else:
            gain_scored = gain_normalized
        total_score = score_ai_twpa_c_gain_data(
            gain_data=gain_scored, gain_min=gain_min, gain_median=gain_median,
            ripple_max=ripple_max, f_min=f_min, f_max=f_max,
        )
        flux_scale = 1e-6
        flux_unit = "µA"
        best_points = []
        for _ in range(n_best):
            pt = find_best_operation_point(total_score, excluded_points=best_points,
                                           exclusion_radius=exclusion_radius)
            if pt is None:
                break
            best_points.append(pt)
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        plot_operation_point_parameters(total_score, best_points, ax=ax1)
        set_units_on_plot_axis(ax1.xaxis, flux_scale, flux_unit)
        set_units_on_plot_axis(ax1.yaxis, 1e9, "GHz", decimals=1)
        ax1.set_title("Score heatmap (pump_freq × ifbl)")
        fig1.tight_layout()
        if best_points:
            ncols = 2
            nrows = max(1, int(np.ceil(len(best_points) / ncols)))
            fig2, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
            axes = np.array(axes).reshape(-1)
            for i, op in enumerate(best_points):
                plot_gain_at_operation_point(gain_normalized, op=op, ax=axes[i])
                pf_ghz = float(op["pump_freq"]) / 1e9
                ib_ua = float(op["ifbl"]) / 1e-6
                axes[i].set_title(f"#{i + 1}  pump={pf_ghz:.3f} GHz  ifbl={ib_ua:.1f} µA")
                set_units_on_plot_axis(axes[i].xaxis, 1e9, "GHz", decimals=1)
            for ax in axes[len(best_points):]:
                ax.set_visible(False)
            fig2.suptitle("Best operation points", fontsize=13)
            fig2.tight_layout()
        plt.show()
        print("\n=== Best operation points ===")
        for i, op in enumerate(best_points):
            print(f"  #{i + 1}  pump_freq = {float(op['pump_freq']) / 1e9:.4f} GHz"
                  f"  |  ifbl = {float(op['ifbl']) / 1e-6:.2f} µA"
                  f"  |  score = {float(op):.3f}")
        return best_points, total_score


# ── s002f: TWPAGainPower ──────────────────────────────────────────────────────

class TWPAGainPower:
    """TWPA gain scan: sweeps pump power × pump frequency × Yoko flux × QICK frequency."""

    YOKO_VOLTAGE_RAMP_STEP: float = 1e-5
    YOKO_CURRENT_RAMP_STEP: float = 1e-8
    YOKO_RAMP_INTERVAL: float = 0.01

    def __init__(self, run_cfg, pump_source, pump_freqs, pump_powers):
        self.run_cfg = run_cfg
        self.pump = pump_source
        self.pump_freqs = np.asarray(pump_freqs, dtype=float)
        self.pump_powers = np.asarray(pump_powers, dtype=float)
        self._power_slices = []
        self._yoko_mode = None
        self._stop = False

    def stop(self):
        self._stop = True
        print("Stop requested.")

    def run(self, py_avg, yoko_inst, yoko_value, yoko_mode="current",
            save_raw=False, qb_idx="TWPA", temp_folder=None, reference=None, **kwargs):
        self._yoko_mode = yoko_mode
        self._power_slices = []
        self._collected_powers = []
        self._stop = False
        self.pump.on()
        print("Pump ON")
        try:
            for pp in tqdm(self.pump_powers, desc="pump_power sweep"):
                if self._stop:
                    tqdm.write("Stopped by user.")
                    break
                tqdm.write(f"\n── pump_power = {pp:+.1f} dBm ──")
                self.pump.power = pp
                freq_slices = []
                try:
                    for pf in tqdm(self.pump_freqs, desc="  pump_freq", leave=False):
                        if self._stop:
                            tqdm.write("  Stopped by user.")
                            break
                        tqdm.write(f"    pump_freq = {pf / 1e9:.4f} GHz")
                        self.pump.frequency = pf
                        exp = TWPAFlux(self.run_cfg)
                        exp.YOKO_VOLTAGE_RAMP_STEP = self.YOKO_VOLTAGE_RAMP_STEP
                        exp.YOKO_CURRENT_RAMP_STEP = self.YOKO_CURRENT_RAMP_STEP
                        exp.YOKO_RAMP_INTERVAL = self.YOKO_RAMP_INTERVAL
                        exp.run(py_avg, yoko_inst=yoko_inst, yoko_value=yoko_value, yoko_mode=yoko_mode)
                        s21 = exp._build_s21_xarray()
                        s21 = s21.assign_coords(pump_freq=pf)
                        freq_slices.append(s21)
                        if save_raw:
                            title = f"power_{pp:+.1f}dBm_pumpfreq_{pf / 1e9:.3f}GHz"
                            exp.saveLabber(qb_idx, title=title)
                except KeyboardInterrupt:
                    tqdm.write("\nKeyboardInterrupt.")
                    self._stop = True
                if freq_slices:
                    self._power_slices.append(freq_slices)
                    self._collected_powers.append(pp)
                    if temp_folder is not None:
                        self.saveNetCDF(reference=reference, save_dir=temp_folder, filename="temp_gain_power")
                if self._stop:
                    break
        finally:
            self.pump.off()
            print("Pump OFF")

    def _build_xarray(self):
        if not self._power_slices:
            raise RuntimeError("No data.")
        collected = self._collected_powers if self._collected_powers else self.pump_powers
        power_das = []
        for pp, freq_slices in zip(collected, self._power_slices):
            da_pf = xr.concat(freq_slices, dim="pump_freq")
            da_pf = da_pf.assign_coords(pump_power=pp)
            power_das.append(da_pf)
        return xr.concat(power_das, dim="pump_power")

    def saveNetCDF(self, reference=None, save_dir=None, filename=None):
        da = self._build_xarray()
        da = da.transpose("pump_power", "frequency", "pump_freq", "ifbl")
        ds = xr.Dataset(
            {"magnitude": xr.apply_ufunc(np.abs, da).assign_attrs(long_name="|S21| linear"),
             "phase": xr.apply_ufunc(np.angle, da).assign_attrs(long_name="arg(S21)", units="rad")},
            attrs={"pump_state": 1, "yoko_mode": self._yoko_mode or "current",
                   "frequency_unit": "Hz", "flux_unit": "V" if self._yoko_mode == "voltage" else "A"},
        )
        root = save_dir or BaseExperiment._data_path or "."
        yy, mm, dd = datetime.datetime.today().strftime("%Y-%m-%d").split("-")
        out_dir = os.path.join(root, yy, mm, f"Data_{mm}{dd}")
        os.makedirs(out_dir, exist_ok=True)
        if filename is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"TWPA_gain_power_{ts}"
        gain_path = os.path.join(out_dir, filename + ".nc")
        ds.to_netcdf(gain_path)
        print(f"Gain-power data saved to {gain_path}")
        ref_path = None
        if reference is not None:
            ref_s21 = reference._build_s21_xarray()
            smallest_flux = float(np.abs(ref_s21["ifbl"]).min())
            ref_row = ref_s21.sel(ifbl=smallest_flux).drop_vars("ifbl")
            ds_ref = xr.Dataset(
                {"magnitude": xr.apply_ufunc(np.abs, ref_row).assign_attrs(long_name="|S21| linear"),
                 "phase": xr.apply_ufunc(np.angle, ref_row).assign_attrs(long_name="arg(S21)", units="rad")},
                attrs={"pump_state": 0, "ifbl": smallest_flux},
            )
            ds_ref = ds_ref.assign_coords(pump_state=0, ifbl=smallest_flux)
            ref_path = os.path.join(out_dir, filename + "_reference.nc")
            ds_ref.to_netcdf(ref_path)
            print(f"Reference saved to {ref_path}")
        return gain_path, ref_path

    def analyze(self, reference, gain_min=12, gain_median=15, ripple_max=5,
                f_min=4e9, f_max=8e9, exclusion_radius=20, freq_exclude=None):
        da = self._build_xarray()
        ref_s21 = reference._build_s21_xarray()
        smallest_flux = float(np.abs(ref_s21["ifbl"]).min())
        ref_row = ref_s21.sel(ifbl=smallest_flux).drop_vars("ifbl")
        results = []
        for pp in self.pump_powers:
            slice_3d = da.sel(pump_power=pp)
            gain_norm = (slice_3d / ref_row).transpose("frequency", "pump_freq", "ifbl")
            if freq_exclude:
                f = gain_norm["frequency"]
                mask = xr.ones_like(f, dtype=bool)
                for f_lo, f_hi in freq_exclude:
                    mask = mask & ~((f >= f_lo) & (f <= f_hi))
                gain_scored = gain_norm.where(mask)
            else:
                gain_scored = gain_norm
            total_score = score_ai_twpa_c_gain_data(
                gain_data=gain_scored, gain_min=gain_min, gain_median=gain_median,
                ripple_max=ripple_max, f_min=f_min, f_max=f_max,
            )
            best = find_best_operation_point(total_score, exclusion_radius=exclusion_radius)
            best_score = float(best) if best is not None else float("nan")
            results.append({"pump_power": pp, "best_point": best, "score": best_score,
                             "gain_normalized": gain_norm, "total_score": total_score})
        return results


# ── s002g: TWPAPowerScan ──────────────────────────────────────────────────────

class TWPAPowerScan:
    """TWPA gain vs pump power at a fixed flux and pump frequency."""

    YOKO_VOLTAGE_RAMP_STEP: float = 1e-5
    YOKO_CURRENT_RAMP_STEP: float = 1e-8
    YOKO_RAMP_INTERVAL: float = 0.01

    def __init__(self, run_cfg, pump_source, pump_freq, pump_powers, flux_value):
        self.run_cfg = run_cfg
        self.pump = pump_source
        self.pump_freq = float(pump_freq)
        self.pump_powers = np.asarray(pump_powers, dtype=float)
        self.flux_value = float(flux_value)
        self._slices = []
        self._collected_powers = []
        self._yoko_mode = None
        self._stop = False

    def stop(self):
        self._stop = True
        print("Stop requested.")

    def run(self, py_avg, yoko_inst, yoko_mode="current",
            temp_folder=None, reference=None, **kwargs):
        self._yoko_mode = yoko_mode
        self._slices = []
        self._collected_powers = []
        self._stop = False
        _rm = pyvisa.ResourceManager()
        _yoko = YOKOGS200(yoko_inst, _rm)
        _yoko.voltage_ramp_step = self.YOKO_VOLTAGE_RAMP_STEP
        _yoko.current_ramp_step = self.YOKO_CURRENT_RAMP_STEP
        _yoko.ramp_interval = self.YOKO_RAMP_INTERVAL
        if yoko_mode == "current":
            _yoko.SetMode("current")
            _yoko.SetCurrent(self.flux_value)
        else:
            _yoko.SetMode("voltage")
            _yoko.SetVoltage(self.flux_value)
        self.pump.frequency = self.pump_freq
        self.pump.on()
        print(f"Pump ON  | freq = {self.pump_freq / 1e9:.4f} GHz"
              f" | flux = {self.flux_value * 1e3:.4f} mA")
        _ref_row = None
        if reference is not None:
            try:
                _ref_s21 = reference._build_s21_xarray()
                _smallest = float(np.abs(_ref_s21["ifbl"]).min())
                _ref_row = _ref_s21.sel(ifbl=_smallest).drop_vars("ifbl")
            except Exception:
                _ref_row = None
        _cmap = plt.cm.viridis
        _cnorm = plt.Normalize(self.pump_powers.min(), self.pump_powers.max())
        _fig_live, _ax_live = plt.subplots(figsize=(9, 5))
        _ax_live.set_xlabel("Frequency (GHz)")
        _ax_live.set_ylabel("Gain (dB)" if _ref_row is not None else "Amplitude")
        _ax_live.set_title(f"TWPA Power Scan\npump_freq = {self.pump_freq / 1e9:.4f} GHz  |  flux = {self.flux_value * 1e3:.4f} mA")
        _ax_live.grid(True, alpha=0.3)
        _sm = plt.cm.ScalarMappable(cmap=_cmap, norm=_cnorm)
        _sm.set_array([])
        _fig_live.colorbar(_sm, ax=_ax_live, label="Pump power (dBm)")
        _fig_live.tight_layout()
        _live_id = f"twpa-ps-live-{np.random.randint(int(1e9))}"
        if _HAS_IPY:
            ipy_display(_fig_live, display_id=_live_id)
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
                    tqdm.write("\nKeyboardInterrupt.")
                    self._stop = True
                    break
                iq_data = iq_list[0][0].dot([1, 1j])
                s21_1d = xr.DataArray(iq_data, dims=["frequency"],
                                      coords={"frequency": _freq_hz, "pump_power": pp})
                self._slices.append(s21_1d)
                self._collected_powers.append(pp)
                _ax_live.cla()
                _ax_live.set_xlabel("Frequency (GHz)")
                _ax_live.set_ylabel("Gain (dB)" if _ref_row is not None else "Amplitude")
                _ax_live.set_title(f"TWPA Power Scan  [{len(self._collected_powers)}/{len(self.pump_powers)}]  —  latest: {pp:+.1f} dBm\n"
                                   f"pump_freq = {self.pump_freq / 1e9:.4f} GHz  |  flux = {self.flux_value * 1e3:.4f} mA")
                _ax_live.grid(True, alpha=0.3)
                for _pp, _sl in zip(self._collected_powers, self._slices):
                    _freq = _sl["frequency"].values / 1e9
                    _y = 20 * np.log10(np.abs(_sl.values / _ref_row.values)) if _ref_row is not None else np.abs(_sl.values)
                    _is_latest = _pp == pp
                    _ax_live.plot(_freq, _y, color=_cmap(_cnorm(_pp)),
                                  lw=1.2 if _is_latest else 0.8, alpha=1.0 if _is_latest else 0.6)
                if _HAS_IPY:
                    ipy_update(_fig_live, display_id=_live_id)
                else:
                    plt.pause(0.01)
                if temp_folder is not None:
                    self.saveNetCDF(reference=reference, save_dir=temp_folder, filename="temp_power_scan")
        finally:
            if temp_folder is not None and self._slices:
                self.saveNetCDF(reference=reference, save_dir=temp_folder, filename="temp_power_scan")

    def _build_xarray(self):
        if not self._slices:
            raise RuntimeError("No data.")
        return xr.concat(self._slices, dim="pump_power")

    def saveNetCDF(self, reference=None, save_dir=None, filename=None):
        da = self._build_xarray()
        ds = xr.Dataset(
            {"magnitude": xr.apply_ufunc(np.abs, da).assign_attrs(long_name="|S21| linear"),
             "phase": xr.apply_ufunc(np.angle, da).assign_attrs(long_name="arg(S21)", units="rad")},
            attrs={"pump_freq": self.pump_freq, "flux_value": self.flux_value, "pump_state": 1,
                   "yoko_mode": self._yoko_mode or "current",
                   "frequency_unit": "Hz", "flux_unit": "V" if self._yoko_mode == "voltage" else "A"},
        )
        ds = ds.assign_coords(pump_freq=self.pump_freq, flux_value=self.flux_value)
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
                {"magnitude": xr.apply_ufunc(np.abs, ref_row).assign_attrs(long_name="|S21| linear"),
                 "phase": xr.apply_ufunc(np.angle, ref_row).assign_attrs(long_name="arg(S21)", units="rad")},
                attrs={"pump_state": 0, "ifbl": smallest_flux},
            )
            ds_ref = ds_ref.assign_coords(pump_state=0, ifbl=smallest_flux)
            ref_path = os.path.join(out_dir, filename + "_reference.nc")
            ds_ref.to_netcdf(ref_path)
            print(f"Reference saved to {ref_path}")
        return gain_path, ref_path

    def analyze(self, reference, f_min=4e9, f_max=8e9, gain_min=10, freq_exclude=None):
        da = self._build_xarray()
        ref_s21 = reference._build_s21_xarray()
        smallest_flux = float(np.abs(ref_s21["ifbl"]).min())
        ref_row = ref_s21.sel(ifbl=smallest_flux).drop_vars("ifbl")
        gain_norm = da / ref_row
        gain_db = 20 * np.log10(np.abs(gain_norm))
        freq_ghz = gain_db["frequency"].values / 1e9
        powers = gain_db["pump_power"].values
        freq = gain_db["frequency"].values
        band_mask = (freq >= f_min) & (freq <= f_max)
        medians = {}
        for pp in powers:
            g_band = gain_db.sel(pump_power=pp).values[band_mask]
            medians[pp] = float(np.nanmedian(g_band)) if g_band.size else float("nan")
        cmap = plt.cm.viridis
        cnorm = plt.Normalize(powers.min(), powers.max())
        fig, ax = plt.subplots(figsize=(9, 5))
        for pp in powers:
            curve = gain_db.sel(pump_power=pp).values
            ax.plot(freq_ghz, curve, color=cmap(cnorm(pp)), lw=0.9,
                    label=f"{pp:.0f} dBm  median {medians[pp]:.1f} dB")
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
        ax.set_title(f"TWPA Gain vs Pump Power\npump_freq = {self.pump_freq / 1e9:.4f} GHz  |  flux = {self.flux_value * 1e3:.4f} mA")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()
        return gain_db


__all__ = ["TWPAFluxProgram", "TWPAFlux", "TWPAGain", "TWPAGainPower", "TWPAPowerScan"]
