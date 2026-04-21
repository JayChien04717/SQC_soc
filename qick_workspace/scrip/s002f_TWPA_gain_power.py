"""
s002f — TWPA Gain vs Pump Power × Pump Frequency
==================================================
Outer loop : sweep pump power.
Middle loop: sweep pump frequency (via AnritsuMG3692).
Inner loop : TWPAFlux sweeps QICK frequency × Yoko flux.

Produces a 4-D xarray (pump_power × pump_freq × ifbl × frequency)
compatible with the scoring tools.

Typical usage
-------------
from qick_workspace.scrip.s002f_TWPA_gain_power import TWPAGainPower

power_sweep = TWPAGainPower(
    run_cfg, pump,
    pump_freqs  = np.linspace(11e9, 14e9, 11),
    pump_powers = np.arange(-20, 11, 1),   # -20 to +10 dBm, 1 dB step
)
power_sweep.YOKO_CURRENT_RAMP_STEP = 10e-6
power_sweep.run(1, yoko_inst=yoko_connect, yoko_value=yoko_range,
                temp_folder="temp", reference=ref)
power_sweep.analyze(reference=ref)
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


class TWPAGainPower:
    """
    TWPA gain scan: sweeps pump power × pump frequency × Yoko flux × QICK frequency.

    Parameters
    ----------
    run_cfg : dict-like
        Experiment config (same as TWPAFlux).
    pump_source : AnritsuMG3692
        Pump signal generator instance.
    pump_freqs : array-like
        Pump frequencies to sweep (Hz).
    pump_powers : array-like
        Pump power values to sweep (dBm).
    """

    YOKO_VOLTAGE_RAMP_STEP: float = 1e-5
    YOKO_CURRENT_RAMP_STEP: float = 1e-8
    YOKO_RAMP_INTERVAL: float = 0.01

    def __init__(self, run_cfg, pump_source, pump_freqs, pump_powers):
        self.run_cfg = run_cfg
        self.pump = pump_source
        self.pump_freqs = np.asarray(pump_freqs, dtype=float)
        self.pump_powers = np.asarray(pump_powers, dtype=float)

        self._power_slices = []  # list[list[DataArray]]: [power_idx][freq_idx]
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
        Sweep pump power × pump frequency and acquire TWPAFlux data at each step.

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
            If True, save each slice as a Labber HDF5 file.
        qb_idx : str
            Label used in the Labber filename.
        temp_folder : str, optional
            If provided, auto-save netCDF after run completes.
        reference : TWPAFlux, optional
            Unpumped reference passed to saveNetCDF when temp_folder is set.
        """
        self._yoko_mode = yoko_mode
        self._power_slices = []
        self._collected_powers = []   # tracks which pump_powers actually finished
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
                        exp.run(
                            py_avg,
                            yoko_inst=yoko_inst,
                            yoko_value=yoko_value,
                            yoko_mode=yoko_mode,
                        )

                        s21 = exp._build_s21_xarray()  # (ifbl, frequency)
                        s21 = s21.assign_coords(pump_freq=pf)
                        freq_slices.append(s21)

                        if save_raw:
                            title = f"power_{pp:+.1f}dBm_pumpfreq_{pf / 1e9:.3f}GHz"
                            exp.saveLabber(qb_idx, title=title)

                except KeyboardInterrupt:
                    tqdm.write("\nKeyboardInterrupt — saving collected data and stopping.")
                    self._stop = True

                # Save whatever freq_slices were collected for this power step
                if freq_slices:
                    self._power_slices.append(freq_slices)
                    self._collected_powers.append(pp)
                    if temp_folder is not None:
                        self.saveNetCDF(
                            reference=reference, save_dir=temp_folder,
                            filename="temp_gain_power",
                        )
                        tqdm.write(f"  [checkpoint] {len(self._collected_powers)}"
                                   f"/{len(self.pump_powers)} powers saved")

                if self._stop:
                    break

        finally:
            self.pump.off()
            print("Pump OFF")
            if temp_folder is not None and self._power_slices:
                self.saveNetCDF(
                    reference=reference, save_dir=temp_folder,
                    filename="temp_gain_power",
                )
                print(f"[final save] {len(self._collected_powers)}"
                      f"/{len(self.pump_powers)} pump_power steps saved"
                      f" → {temp_folder}")

    # ------------------------------------------------------------------
    # Build 4-D xarray
    # ------------------------------------------------------------------

    def _build_xarray(self):
        """Return 4-D DataArray (pump_power × pump_freq × ifbl × frequency)."""
        if not self._power_slices:
            raise RuntimeError("No data. Run the experiment first.")

        collected = self._collected_powers if self._collected_powers else self.pump_powers
        power_das = []
        for pp, freq_slices in zip(collected, self._power_slices):
            da_pf = xr.concat(freq_slices, dim="pump_freq")  # (pump_freq, ifbl, freq)
            da_pf = da_pf.assign_coords(pump_power=pp)
            power_das.append(da_pf)

        return xr.concat(
            power_das, dim="pump_power"
        )  # (pump_power, pump_freq, ifbl, freq)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def saveNetCDF(self, reference=None, save_dir=None, filename=None):
        """Save raw 4-D gain data as netCDF."""
        da = self._build_xarray()
        # scoring expects (frequency, pump_freq, ifbl) per power slice
        # store with pump_power as extra dim
        da = da.transpose("pump_power", "frequency", "pump_freq", "ifbl")

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
                "pump_state": 1,
                "yoko_mode": self._yoko_mode or "current",
                "frequency_unit": "Hz",
                "flux_unit": "V" if self._yoko_mode == "voltage" else "A",
            },
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
        gain_min=12,
        gain_median=15,
        ripple_max=5,
        f_min=4e9,
        f_max=8e9,
        exclusion_radius=20,
        freq_exclude=None,
    ):
        """
        For each pump power: normalize, score, find best operation point.
        Then plot score vs pump power and gain curves at best points.

        Parameters
        ----------
        reference : TWPAFlux
            Unpumped reference measurement.
        gain_min : float
            Minimum gain threshold in dB for scoring.
        gain_median : float
            Target median gain in dB for scoring.
        ripple_max : float
            Max allowed pk-to-pk ripple in dB for scoring.
        f_min, f_max : float
            Frequency range in Hz for scoring.
        exclusion_radius : float
            Exclusion radius for find_best_operation_point.
        freq_exclude : list of (float, float), optional
            Frequency bands to mask before scoring.

        Returns
        -------
        results : list of dict
            One entry per pump power:
            ``{"pump_power", "best_point", "score", "gain_normalized"}``.
        """
        da = self._build_xarray()  # (pump_power, pump_freq, ifbl, frequency)

        # Normalize by zero-flux reference
        ref_s21 = reference._build_s21_xarray()
        smallest_flux = float(np.abs(ref_s21["ifbl"]).min())
        ref_row = ref_s21.sel(ifbl=smallest_flux).drop_vars("ifbl")  # (frequency,)

        if freq_exclude:
            print(
                f"Excluded: {[f'{a / 1e9:.3f}–{b / 1e9:.3f} GHz' for a, b in freq_exclude]}"
            )

        results = []
        for pp in self.pump_powers:
            slice_3d = da.sel(pump_power=pp)  # (pump_freq, ifbl, frequency)
            gain_norm = (slice_3d / ref_row).transpose("frequency", "pump_freq", "ifbl")

            # Freq exclusion mask
            if freq_exclude:
                f = gain_norm["frequency"]
                mask = xr.ones_like(f, dtype=bool)
                for f_lo, f_hi in freq_exclude:
                    mask = mask & ~((f >= f_lo) & (f <= f_hi))
                gain_scored = gain_norm.where(mask)
            else:
                gain_scored = gain_norm

            total_score = score_ai_twpa_c_gain_data(
                gain_data=gain_scored,
                gain_min=gain_min,
                gain_median=gain_median,
                ripple_max=ripple_max,
                f_min=f_min,
                f_max=f_max,
            )

            best = find_best_operation_point(
                total_score, exclusion_radius=exclusion_radius
            )
            best_score = float(best) if best is not None else float("nan")

            results.append(
                {
                    "pump_power": pp,
                    "best_point": best,
                    "score": best_score,
                    "gain_normalized": gain_norm,
                    "total_score": total_score,
                }
            )

        # ── Plot 1: best score vs pump power ──────────────────────────
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        valid = [
            (r["pump_power"], r["score"]) for r in results if not np.isnan(r["score"])
        ]
        if valid:
            pp_arr, sc_arr = zip(*valid)
            ax1.plot(pp_arr, sc_arr, "o-", color="steelblue")
            best_idx = int(np.argmax(sc_arr))
            ax1.axvline(
                pp_arr[best_idx],
                color="red",
                ls="--",
                lw=1,
                label=f"best {pp_arr[best_idx]:+.0f} dBm",
            )
            ax1.legend()
        ax1.set_xlabel("Pump power (dBm)")
        ax1.set_ylabel("Best score")
        ax1.set_title("Best score vs Pump Power")
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()

        # ── Plot 2: gain curve at best point for each power (color by power) ─
        valid_results = [r for r in results if r["best_point"] is not None]
        if valid_results:
            fig2, ax2 = plt.subplots(figsize=(9, 5))
            cmap = plt.cm.viridis
            pvals = [r["pump_power"] for r in valid_results]
            cnorm = plt.Normalize(min(pvals), max(pvals))

            for r in valid_results:
                op = r["best_point"]
                gain = r["gain_normalized"]
                gain_at_op = 20 * np.log10(
                    np.abs(
                        gain.sel(
                            pump_freq=float(op["pump_freq"]),
                            ifbl=float(op["ifbl"]),
                            method="nearest",
                        )
                    )
                )
                ax2.plot(
                    gain["frequency"].values / 1e9,
                    gain_at_op.values,
                    color=cmap(cnorm(r["pump_power"])),
                    lw=0.9,
                )

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)
            fig2.colorbar(sm, ax=ax2, label="Pump power (dBm)")
            ax2.axvline(f_min / 1e9, color="gray", ls="--", lw=0.8)
            ax2.axvline(f_max / 1e9, color="gray", ls="--", lw=0.8)
            ax2.set_xlabel("Frequency (GHz)")
            ax2.set_ylabel("Gain (dB)")
            ax2.set_title("Gain at Best Operation Point vs Pump Power")
            set_units_on_plot_axis(ax2.xaxis, 1e9, "GHz", decimals=1)
            fig2.tight_layout()

        # ── Plot 3: score heatmap at power with highest score ─────────
        if valid_results:
            best_r = max(valid_results, key=lambda r: r["score"])
            fig3, ax3 = plt.subplots(figsize=(7, 5))
            plot_operation_point_parameters(
                best_r["total_score"], [best_r["best_point"]], ax=ax3
            )
            set_units_on_plot_axis(ax3.xaxis, 1e-6, "µA")
            set_units_on_plot_axis(ax3.yaxis, 1e9, "GHz", decimals=1)
            ax3.set_title(
                f"Score heatmap at best power = {best_r['pump_power']:+.0f} dBm"
            )
            fig3.tight_layout()

        plt.show()

        # Print summary
        print("\n=== Best operation point per pump power ===")
        print(
            f"  {'Power (dBm)':>12}  {'pump_freq (GHz)':>16}  "
            f"{'ifbl (µA)':>10}  {'score':>7}"
        )
        print("-" * 52)
        for r in results:
            op = r["best_point"]
            if op is not None:
                pf = float(op["pump_freq"]) / 1e9
                ib = float(op["ifbl"]) / 1e-6
                print(
                    f"  {r['pump_power']:>12.1f}  {pf:>16.4f}  "
                    f"{ib:>10.2f}  {r['score']:>7.3f}"
                )
            else:
                print(f"  {r['pump_power']:>12.1f}  {'—':>16}  {'—':>10}  {'—':>7}")

        return results
