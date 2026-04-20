"""
s016 — Single Qubit State Tomography
======================================
X/Y/Z measurement axes with calibration (|0⟩, |1⟩).
Uses setup_standard_gates() for gate pulses.
"""
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import matplotlib.colors as mcolors

from .base_program import BaseProgram, resolve_gate
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import hdf5_generator, get_next_filename_labber, config_to_yaml


# ── Program ──

class StateTomographyProgram(BaseProgram):
    """QICK program for a single tomography axis measurement with optional state preparation."""

    def _initialize(self, cfg):
        """Set up resonator, qubit generator, and standard ge gates."""
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, 'ge')
        self.setup_standard_gates(cfg, prefix='ge')

    def _body(self, cfg):
        """Apply optional calibration or prep pulse, tomography pre-rotation, then measure."""
        axis = cfg["tomo_axis"]
        cal_pulse = cfg.get("cal_pulse", None)
        prep_pulse = cfg.get("prep_pulse", None)

        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)

        # 1. Optional calibration pulse (x180 for |1⟩)
        if cal_pulse == "x180_ge":
            self.pulse(ch=cfg["qb_ch"], name="x180_ge", t=0)
            self.delay_auto(0.05)
        # 2. Optional state preparation pulse
        elif prep_pulse is not None and prep_pulse != "None":
            self.pulse(ch=cfg["qb_ch"], name=prep_pulse, t=0)
            self.delay_auto(0.05)

        # 3. Tomography pre-rotation
        if axis == "X":
            self.pulse(ch=cfg["qb_ch"], name="y90m_ge", t=0)
            self.delay_auto(0.01)
        elif axis == "Y":
            self.pulse(ch=cfg["qb_ch"], name="x90_ge", t=0)
            self.delay_auto(0.01)
        # Z: no rotation

        # 4. Readout
        self.delay_auto(0.05)
        self.measure(cfg)


# ── Experiment ──

class Tomography:
    """Single-qubit state tomography with X/Y/Z measurements and MLE reconstruction."""

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Experiment configuration dictionary.

        Raises
        ------
        RuntimeError
            If ``BaseExperiment.setup()`` has not been called first.
        """
        from .base_experiment import BaseExperiment
        if BaseExperiment._soc is None:
            raise RuntimeError("Call BaseExperiment.setup(soc, soccfg, data_path) first.")
        self.soc = BaseExperiment._soc
        self.soccfg = BaseExperiment._soccfg
        self.cfg = config
        self.iq_g = None
        self.iq_e = None
        self.tomo_data_raw = {}
        self.expect_values = {}
        self.rho_mle = None
        self.prep_pulse_name = None
        self._I = np.array([[1,0],[0,1]], dtype=complex)
        self._sx = np.array([[0,1],[1,0]], dtype=complex)
        self._sy = np.array([[0,-1j],[1j,0]], dtype=complex)
        self._sz = np.array([[1,0],[0,-1]], dtype=complex)

    def _run_calibration(self, pyavg):
        """Acquire IQ data for |0> and |1> calibration states."""
        print("Calibrating |0⟩ state...")
        cfg_g = self.cfg.copy()
        cfg_g.update({"tomo_axis": "Z", "cal_pulse": "None", "prep_pulse": None})
        prog_g = StateTomographyProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=cfg_g,
        )
        iq_g = prog_g.acquire(self.soc, rounds=pyavg, progress=False)[0][0].dot([1, 1j])

        print("Calibrating |1⟩ state...")
        cfg_e = self.cfg.copy()
        cfg_e.update({"tomo_axis": "Z", "cal_pulse": "x180_ge", "prep_pulse": None})
        prog_e = StateTomographyProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=cfg_e,
        )
        iq_e = prog_e.acquire(self.soc, rounds=pyavg, progress=False)[0][0].dot([1, 1j])

        print(f"IQ Ground: {iq_g}, IQ Excited: {iq_e}")
        return iq_g, iq_e

    def _run_tomography(self, pyavg, prep_pulse_name=None):
        """Acquire IQ data along X, Y, and Z measurement axes."""
        resolved = resolve_gate(prep_pulse_name) if prep_pulse_name else None
        tomo_data = {}
        for axis in tqdm(["X", "Y", "Z"], desc=f"Tomography ({prep_pulse_name})"):
            cfg = self.cfg.copy()
            cfg.update({"tomo_axis": axis, "cal_pulse": None, "prep_pulse": resolved})
            prog = StateTomographyProgram(
                self.soccfg, reps=self.cfg["reps"],
                final_delay=self.cfg["relax_delay"], cfg=cfg,
            )
            iq = prog.acquire(self.soc, rounds=pyavg, progress=False)[0][0].dot([1, 1j])
            tomo_data[axis] = iq
        return tomo_data

    def _project_to_expect(self, iq_data, iq_g, iq_e):
        """
        Project IQ data onto the |0>-|1> calibration axis to obtain an expectation value.

        Parameters
        ----------
        iq_data : complex
            Measured IQ value.
        iq_g : complex
            Calibration IQ for |0> state.
        iq_e : complex
            Calibration IQ for |1> state.

        Returns
        -------
        expect : float
            Expectation value clipped to [-1, 1].
        """
        cal_vector = iq_e - iq_g
        data_vector = iq_data - iq_g
        projection = np.real(data_vector * np.conj(cal_vector)) / np.abs(cal_vector)**2
        return np.clip((1 - projection) - projection, -1, 1)

    def _mle_reconstruction(self, rho_raw):
        """
        Project a raw density matrix to the nearest physical state via MLE.

        Parameters
        ----------
        rho_raw : ndarray of shape (2, 2)
            Raw density matrix (may be non-physical).

        Returns
        -------
        rho_mle : ndarray of shape (2, 2)
            Physical density matrix with non-negative eigenvalues and unit trace.
        """
        eig_vals, eig_vecs = np.linalg.eigh(rho_raw)
        eig_vals = np.maximum(0, eig_vals)
        trace = np.sum(eig_vals)
        eig_vals = eig_vals / trace if trace > 0 else eig_vals
        return eig_vecs @ np.diag(eig_vals) @ np.conj(eig_vecs.T)

    def _reconstruct_density_matrix(self):
        """
        Reconstruct the density matrix from stored X, Y, Z tomography data.

        Returns
        -------
        expect_values : dict
            Bloch vector components ``{"X": float, "Y": float, "Z": float}``.
        rho_mle : ndarray of shape (2, 2)
            MLE-reconstructed density matrix.
        """
        expect_values = {}
        for axis in ["X", "Y", "Z"]:
            expect_values[axis] = self._project_to_expect(
                self.tomo_data_raw[axis], self.iq_g, self.iq_e)
        r_x, r_y, r_z = expect_values["X"], expect_values["Y"], expect_values["Z"]
        rho_raw = 0.5 * (self._I + r_x*self._sx + r_y*self._sy + r_z*self._sz)
        rho_mle = self._mle_reconstruction(rho_raw)
        purity = np.real(np.trace(rho_mle @ rho_mle))
        print(f"\n<X>={r_x:.4f}, <Y>={r_y:.4f}, <Z>={r_z:.4f}")
        print(f"Purity: {purity:.5f}")
        return expect_values, rho_mle

    def run(self, py_avg, prep_pulse_name=None):
        """
        Run calibration and tomography, then reconstruct the density matrix.

        Parameters
        ----------
        py_avg : int
            Hardware averages (rounds) per measurement axis.
        prep_pulse_name : str or None, optional
            Gate name to prepare the state under study (e.g. ``"X"``).
            ``None`` prepares the ground state.
        """
        self.prep_pulse_name = str(prep_pulse_name)
        self.iq_g, self.iq_e = self._run_calibration(py_avg)
        self.tomo_data_raw = self._run_tomography(py_avg, prep_pulse_name)
        self.expect_values, self.rho_mle = self._reconstruct_density_matrix()

    def plot(self, plot_type="2d", qb_idx=None):
        """
        Plot the reconstructed density matrix as a 2D heatmap or 3D bar chart.

        Parameters
        ----------
        plot_type : str, optional
            ``"2d"`` for a 2D matrix heatmap or ``"3d"`` for a 3D bar chart.
        qb_idx : int or None, optional
            Qubit index prepended to the plot title when provided.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : tuple or None
            Axes objects; ``None`` for 3D plots.
        """
        if self.rho_mle is None:
            print("No data. Run first.")
            return None, None
        title_prefix = f"State: '{self.prep_pulse_name}'"
        if qb_idx is not None:
            title_prefix = f"Q{qb_idx} - {title_prefix}"

        rho_real = self.rho_mle.real
        rho_imag = self.rho_mle.imag
        cmap = plt.get_cmap("RdBu")
        vmax_r = max(np.max(np.abs(rho_real)), 1e-9)
        vmax_i = max(np.max(np.abs(rho_imag)), 1e-9)
        norm_r = mcolors.Normalize(vmin=-vmax_r, vmax=vmax_r)
        norm_i = mcolors.Normalize(vmin=-vmax_i, vmax=vmax_i)
        labels_k, labels_b = ["|0⟩", "|1⟩"], ["⟨0|", "⟨1|"]

        if plot_type == "2d":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle(title_prefix, fontsize=18, y=1.05)
            for ax, data, norm, title in [(ax1,rho_real,norm_r,"Real(ρ)"), (ax2,rho_imag,norm_i,"Imag(ρ)")]:
                im = ax.matshow(data, cmap=cmap, norm=norm)
                ax.set_title(title, fontsize=16)
                fig.colorbar(im, ax=ax, shrink=0.8)
                ax.set_xticks([0,1]); ax.set_yticks([0,1])
                ax.set_xticklabels(labels_k); ax.set_yticklabels(labels_b)
                ax.xaxis.set_ticks_position("bottom")
                for i in range(2):
                    for j in range(2):
                        val = data[i,j]
                        bg = cmap(norm(val))
                        tc = "black" if 0.299*bg[0]+0.587*bg[1]+0.114*bg[2] > 0.5 else "white"
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=tc, fontsize=12)
            plt.tight_layout()
            plt.show()
            return fig, (ax1, ax2)
        elif plot_type == "3d":
            fig = plt.figure(figsize=(12, 6))
            fig.suptitle(title_prefix, fontsize=18, y=1.0)
            for idx, (data, norm, title) in enumerate([(rho_real,norm_r,"Real(ρ)"), (rho_imag,norm_i,"Imag(ρ)")]):
                ax = fig.add_subplot(1, 2, idx+1, projection="3d")
                x_pos, y_pos = [0,0,1,1], [0,1,0,1]
                dz = data.flatten()
                colors = cmap(norm(dz))
                ax.bar3d(x_pos, y_pos, np.zeros(4), 0.8, 0.8, dz, color=colors, shade=True)
                ax.set_title(title, fontsize=16)
                ax.set_xticks([0.4,1.4]); ax.set_yticks([0.4,1.4])
                ax.set_xticklabels(labels_k); ax.set_yticklabels(labels_b)
                z_max = max(np.max(np.abs(data)), 1e-9)
                ax.set_zlim(-z_max, z_max)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
            return fig, None

    def saveLabber(self, qb_idx, yoko_value=None):
        """
        Save tomography data to an HDF5/Labber file.

        Parameters
        ----------
        qb_idx : int
            Qubit index appended to the experiment name.
        yoko_value : float or None, optional
            Yokogawa flux bias value embedded in the filename.
        """
        if not self.tomo_data_raw:
            print("No data. Run first.")
            return
        expt_name = f"s016_Tomography_ge_Q{qb_idx}"
        from .base_experiment import BaseExperiment
        save_dir = BaseExperiment._data_path or DATA_PATH
        file_path = get_next_filename_labber(save_dir, expt_name, yoko_value)
        dict_val = config_to_yaml(self.cfg)
        comment = (
            f"{dict_val}\n--- Tomography ---\n"
            f"Prepared: {self.prep_pulse_name}\n"
            f"<X>={self.expect_values['X']:.4f}, "
            f"<Y>={self.expect_values['Y']:.4f}, "
            f"<Z>={self.expect_values['Z']:.4f}\n"
            f"rho_mle:\n{self.rho_mle}"
        )
        x_vals = np.array([0, 1, 2])
        z_vals = np.array([self.tomo_data_raw["X"], self.tomo_data_raw["Y"], self.tomo_data_raw["Z"]])
        hdf5_generator(
            filepath=file_path,
            x_info={"name": "Axis", "unit": "None (0=X, 1=Y, 2=Z)", "values": x_vals},
            z_info={"name": "Signal", "unit": "ADC unit", "values": z_vals},
            comment=comment, tag="Tomography",
        )
        print(f"Data save to {file_path}")
