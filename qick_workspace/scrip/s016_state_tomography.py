# ===================================================================
# 1. Standard & Third-Party Scientific Libraries
# ===================================================================
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from IPython.display import display, clear_output
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

# ===================================================================
# 2. QICK Libraries
# ===================================================================
from qick import *
from qick.pyro import make_proxy
from qick.asm_v2 import AveragerProgramV2, QickSpan, QickSweep1D

# ===================================================================
# 3. User/Local Libraries
# ===================================================================
from ..tools.system_cfg import *
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import get_next_filename_labber, hdf5_generator
from ..tools.fitting import *
from ..tools.yamltool import yml_comment


class StateTomography(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        qb_ch = cfg["qb_ch"]

        # --- Standardized Generator Declaration (from AllXY) ---
        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])

        if self.soccfg["gens"][qb_ch]["type"] == "axis_sg_int4_v2":
            self.declare_gen(ch=qb_ch, nqz=cfg["nqz_qb"], mixer_freq=cfg["qb_mixer"])
        else:
            self.declare_gen(ch=qb_ch, nqz=cfg["nqz_qb"])

        # --- Standardized Readout Declaration (from AllXY) ---
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])
        self.add_readoutconfig(
            ch=ro_ch, name="myro", freq=cfg["res_freq_ge"], gen_ch=res_ch
        )

        # --- Standardized Readout Pulse (from AllXY) ---
        self.add_gauss(
            ch=res_ch,
            name="readout",
            sigma=cfg["res_sigma"],
            length=5 * cfg["res_sigma"],
            even_length=True,
        )
        self.add_pulse(
            ch=res_ch,
            name="res_pulse",
            ro_ch=ro_ch,
            style="flat_top",
            envelope="readout",
            length=cfg["res_length"],
            freq=cfg["res_freq_ge"],
            phase=cfg["res_phase"],
            gain=cfg["res_gain_ge"],
        )

        # --- Standardized qb Pulses (from AllXY) ---
        self.add_gauss(
            ch=qb_ch,
            name="ramp",
            sigma=cfg["sigma"],
            length=cfg["sigma"] * 5,
            even_length=True,
        )
        if cfg["pulse_type"] == "arb":
            for name, phase, gain in [
                ("x180", 0, cfg["pi_gain_ge"]),
                ("y180", 90, cfg["pi_gain_ge"]),
                ("x90", 0, cfg["pi2_gain_ge"]),
                ("y90", 90, cfg["pi2_gain_ge"]),
            ]:
                self.add_pulse(
                    ch=qb_ch,
                    name=name,
                    style="arb",
                    envelope="ramp",
                    freq=cfg["qb_freq_ge"],
                    phase=phase,
                    gain=gain,
                )

        elif cfg["pulse_type"] == "flat_top":
            for name, phase, gain in [
                ("x180", 0, cfg["pi_gain_ge"]),
                ("y180", 90, cfg["pi_gain_ge"]),
                ("x90", 0, cfg["pi2_gain_ge"]),
                ("y90", 90, cfg["pi2_gain_ge"]),
            ]:
                self.add_pulse(
                    ch=qb_ch,
                    name=name,
                    style="flat_top",
                    envelope="ramp",
                    freq=cfg["qb_freq_ge"],
                    phase=phase,
                    gain=gain,
                    length=cfg["flat_top_len"],
                )

        # --- Tomography-Specific Pulse (y90m) ---
        # (Assuming 'arb' style for simplicity, add flat_top if needed)
        self.add_pulse(
            ch=qb_ch,
            name="y90m",
            style="arb",  # or cfg["pulse_type"]
            envelope="ramp",
            freq=cfg["qb_freq_ge"],
            phase=-90,
            gain=cfg["pi2_gain_ge"],
            # length=cfg.get("flat_top_len", 10), # Add if using flat_top
        )

    def apply_cool(self, cfg):
        # --- Cooling Method (Copied from AllXY) ---
        cool_ch1 = cfg["cool_ch1"]
        cool_ch2 = cfg["cool_ch2"]
        if self.soccfg["gens"][cool_ch1]["type"] == "axis_sg_int4_v2":
            self.declare_gen(
                ch=cool_ch1, nqz=cfg["nqz_cool_ch1"], mixer_freq=cfg["cool_mixer1"]
            )
        else:
            self.declare_gen(ch=cool_ch1, nqz=cfg["nqz_cool_ch1"])

        if self.soccfg["gens"][cool_ch2]["type"] == "axis_sg_int4_v2":
            self.declare_gen(
                ch=cool_ch2, nqz=cfg["nqz_cool_ch2"], mixer_freq=cfg["cool_mixer2"]
            )
        else:
            self.declare_gen(ch=cool_ch2, nqz=cfg["nqz_cool_ch2"])
        self.add_gauss(
            ch=cool_ch1,
            name="cooling1",
            sigma=cfg["res_sigma"],
            length=5 * cfg["res_sigma"],
            even_length=True,
        )
        self.add_pulse(
            ch=cool_ch1,
            name="cool_pulse1",
            style="flat_top",
            envelope="cooling1",
            length=cfg["cool_length"],
            freq=cfg["cool_freq_1"],
            phase=0,
            gain=cfg["cool_gain_1"],
        )
        self.add_gauss(
            ch=cool_ch2,
            name="cooling2",
            sigma=cfg["res_sigma"],
            length=5 * cfg["res_sigma"],
            even_length=True,
        )
        self.add_pulse(
            ch=cool_ch2,
            name="cool_pulse2",
            style="flat_top",
            envelope="cooling2",
            length=cfg["cool_length"],
            freq=cfg["cool_freq_2"],
            phase=0,
            gain=cfg["cool_gain_2"],
        )

    def _body(self, cfg):
        axis = cfg["tomo_axis"]
        cal_pulse = cfg.get("cal_pulse", None)
        prep_pulse = cfg.get("prep_pulse", None)

        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)

        # --- Cooling (from AllXY) ---
        if cfg["cooling"] is True:
            self.apply_cool(cfg)
            self.pulse(ch=self.cfg["cool_ch1"], name="cool_pulse1", t=0)
            self.pulse(ch=self.cfg["cool_ch2"], name="cool_pulse2", t=0)
            self.delay_auto(0.5, tag="Ring down")
        else:
            pass

        # 1. (Optional) Apply calibration pulse
        if cal_pulse == "x180":
            self.pulse(ch=cfg["qb_ch"], name="x180", t=0)
            self.delay_auto(0.05)

        # 2. (Optional) Apply state preparation pulse
        elif prep_pulse is not None and prep_pulse != "None":
            self.pulse(ch=cfg["qb_ch"], name=prep_pulse, t=0)
            self.delay_auto(0.05)

        # 3. Apply tomography pre-rotation
        if axis == "X":
            self.pulse(ch=cfg["qb_ch"], name="y90m", t=0)
            self.delay_auto(0.01)
        elif axis == "Y":
            self.pulse(ch=cfg["qb_ch"], name="x90", t=0)
            self.delay_auto(0.01)
        elif axis == "Z":
            pass  # No rotation for Z measurement

        # 4. Readout
        self.delay_auto(0.05)
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


# ######################################################
# ### Tomography Controller Class (Matplotlib 3D) ###
# ######################################################


class Tomography:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

        # Initialize result containers
        self.iq_g = None
        self.iq_e = None
        self.tomo_data_raw = {}
        self.expect_values = {}
        self.rho_mle = None  # This will be a numpy array
        self.prep_pulse_name = None

        # --- Define Pauli matrices as numpy arrays ---
        self._I = np.array([[1, 0], [0, 1]], dtype=complex)
        self._sx = np.array([[0, 1], [1, 0]], dtype=complex)
        self._sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self._sz = np.array([[1, 0], [0, -1]], dtype=complex)

    # ... (所有 _run_calibration, _run_tomography, ... _reconstruct_density_matrix
    #      等內部方法保持不變，複製您上一版本 (no qutip) 的即可) ...

    def _run_calibration(self, pyavg):
        """Internal method to calibrate |0> and |1> states."""
        print("Calibrating |0> state...")
        cfg_g = self.cfg.copy()
        cfg_g["tomo_axis"] = "Z"
        cfg_g["cal_pulse"] = "None"
        cfg_g["prep_pulse"] = None
        prog_g = StateTomography(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=cfg_g,
        )
        iq_list_g = prog_g.acquire(self.soc, rounds=pyavg, progress=False)
        iq_g = iq_list_g[0][0].dot([1, 1j])

        print("Calibrating |1> state...")
        cfg_e = self.cfg.copy()
        cfg_e["tomo_axis"] = "Z"
        cfg_e["cal_pulse"] = "x180"
        cfg_e["prep_pulse"] = None
        prog_e = StateTomography(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=cfg_e,
        )
        iq_list_e = prog_e.acquire(self.soc, rounds=pyavg, progress=False)
        iq_e = iq_list_e[0][0].dot([1, 1j])

        print(f"IQ Ground (|0>): {iq_g}")
        print(f"IQ Excited (|1>): {iq_e}")
        return iq_g, iq_e

    def _run_tomography(self, pyavg, prep_pulse_name=None):
        """Internal method to run X, Y, Z measurements."""
        if prep_pulse_name:
            print(f"Running tomography for state prepared by: {prep_pulse_name}")
        else:
            print("Running tomography for ground state |0>")

        tomo_data = {}
        for axis in tqdm(
            ["X", "Y", "Z"], desc=f"Tomography (State: {prep_pulse_name})"
        ):
            cfg = self.cfg.copy()
            cfg["tomo_axis"] = axis
            cfg["cal_pulse"] = None
            cfg["prep_pulse"] = prep_pulse_name
            prog = StateTomography(
                self.soccfg,
                reps=self.cfg["reps"],
                final_delay=self.cfg["relax_delay"],
                cfg=cfg,
            )
            iq_list = prog.acquire(self.soc, rounds=pyavg, progress=False)
            iq = iq_list[0][0].dot([1, 1j])
            tomo_data[axis] = iq
        print(f"Raw Tomography IQ data: {tomo_data}")
        return tomo_data

    def _project_to_expect(self, iq_data, iq_g, iq_e):
        """Internal method for IQ projection."""
        cal_vector = iq_e - iq_g
        data_vector = iq_data - iq_g
        projection = (
            np.real(data_vector * np.conj(cal_vector)) / np.abs(cal_vector) ** 2
        )
        expect_value = (1 - projection) - projection
        return np.clip(expect_value, -1, 1)

    def _mle_reconstruction(self, rho_raw):
        """
        Internal method for Maximum Likelihood Estimation using NumPy.
        rho_raw is a numpy array.
        """
        eig_vals, eig_vecs = np.linalg.eigh(rho_raw)
        eig_vals_clipped = np.maximum(0, eig_vals)
        trace = np.sum(eig_vals_clipped)
        if trace > 0:
            eig_vals_norm = eig_vals_clipped / trace
        else:
            eig_vals_norm = eig_vals_clipped
        rho_mle = eig_vecs @ np.diag(eig_vals_norm) @ np.conj(eig_vecs.T)
        return rho_mle

    def _reconstruct_density_matrix(self):
        """Internal method to reconstruct rho from stored data using NumPy."""
        expect_values = {}
        for axis in ["X", "Y", "Z"]:
            iq_measured = self.tomo_data_raw[axis]
            expect_values[axis] = self._project_to_expect(
                iq_measured, self.iq_g, self.iq_e
            )

        print("\nMeasured Expectation Values:")
        print(f"  <X> = {expect_values['X']:.4f}")
        print(f"  <Y> = {expect_values['Y']:.4f}")
        print(f"  <Z> = {expect_values['Z']:.4f}")

        r_x, r_y, r_z = expect_values["X"], expect_values["Y"], expect_values["Z"]
        rho_raw = 0.5 * (self._I + r_x * self._sx + r_y * self._sy + r_z * self._sz)

        print("\n--- Raw Density Matrix (before MLE) ---")
        print(rho_raw)
        rho_mle = self._mle_reconstruction(rho_raw)
        print("\n--- Final MLE Reconstructed Density Matrix ---")
        print(rho_mle)

        purity = np.real(np.trace(rho_mle @ rho_mle))
        print(f"\nPurity: {purity:.5f} (1.0 = pure state)")

        return expect_values, rho_mle

    def run(self, py_avg, prep_pulse_name=None):
        """
        Run the full tomography experiment.
        """
        self.prep_pulse_name = str(prep_pulse_name)
        self.iq_g, self.iq_e = self._run_calibration(py_avg)
        self.tomo_data_raw = self._run_tomography(py_avg, prep_pulse_name)
        self.expect_values, self.rho_mle = self._reconstruct_density_matrix()

    # ##############################################
    # ### --- MODIFIED PLOT METHOD HERE --- ###
    # ##############################################
    def plot(self, plot_type="2d", qb_idx=None):
        """
        Plot the reconstructed density matrix.

        Args:
            plot_type (str): '2d' for Real/Imaginary matrix plots (default),
                             '3d' for a 3D bar chart (using Matplotlib).
            qb_idx (int, optional): The qubit index, used for the title.
        """
        if self.rho_mle is None:
            print("No data to plot. Run the experiment first using .run()")
            return None, None

        # 建立標題
        title_prefix = f"State: (prepared by '{self.prep_pulse_name}')"
        if qb_idx is not None:
            title_prefix = f"Q{qb_idx} - {title_prefix}"

        rho_np = self.rho_mle
        rho_real = rho_np.real
        rho_imag = rho_np.imag

        # --- 共用的顏色設定 ---
        cmap = plt.get_cmap("RdBu")
        vmax_real = np.max(np.abs(rho_real))
        vmax_imag = np.max(np.abs(rho_imag))
        if vmax_real < 1e-9:
            vmax_real = 1.0
        if vmax_imag < 1e-9:
            vmax_imag = 1.0

        norm_real = mcolors.Normalize(vmin=-vmax_real, vmax=vmax_real)
        norm_imag = mcolors.Normalize(vmin=-vmax_imag, vmax=vmax_imag)

        # --- Option 1: 2D Matrix Plot ---
        if plot_type == "2d":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            if title_prefix:
                fig.suptitle(title_prefix, fontsize=18, y=1.05)

            im1 = ax1.matshow(rho_real, cmap=cmap, norm=norm_real)
            ax1.set_title("Real($\\rho$)", fontsize=16)
            fig.colorbar(im1, ax=ax1, shrink=0.8)

            im2 = ax2.matshow(rho_imag, cmap=cmap, norm=norm_imag)
            ax2.set_title("Imag($\\rho$)", fontsize=16)
            fig.colorbar(im2, ax=ax2, shrink=0.8)

            labels_kets = ["|0>", "|1>"]
            labels_bras = ["<0|", "<1|"]

            def get_text_color(bg_color_rgb):
                luminance = (
                    0.299 * bg_color_rgb[0]
                    + 0.587 * bg_color_rgb[1]
                    + 0.114 * bg_color_rgb[2]
                )
                return "black" if luminance > 0.5 else "white"

            for ax, data, norm in [
                (ax1, rho_real, norm_real),
                (ax2, rho_imag, norm_imag),
            ]:
                ax.set_xticks(np.arange(len(labels_kets)))
                ax.set_yticks(np.arange(len(labels_bras)))
                ax.set_xticklabels(labels_kets)
                ax.set_yticklabels(labels_bras)
                ax.xaxis.set_ticks_position("bottom")
                for i in range(2):
                    for j in range(2):
                        val = data[i, j]
                        bg_color = cmap(norm(val))
                        text_color = get_text_color(bg_color[:3])
                        ax.text(
                            j,
                            i,
                            f"{val:.2f}",
                            ha="center",
                            va="center",
                            color=text_color,
                            fontsize=12,
                        )

            plt.tight_layout()
            plt.show()
            return fig, (ax1, ax2)

        # --- Option 2: 3D Bar Chart (Matplotlib) ---
        elif plot_type == "3d":
            fig = plt.figure(figsize=(12, 6))
            if title_prefix:
                fig.suptitle(title_prefix, fontsize=18, y=1.0)

            # --- 3D 繪圖輔助函式 ---
            def plot_3d_bar(ax, data, norm, title):
                x_pos = [0, 0, 1, 1]
                y_pos = [0, 1, 0, 1]

                z_pos = np.zeros(4)

                dx = dy = 0.8

                dz = data.flatten()

                colors = cmap(norm(dz))

                ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, shade=True)

                ax.set_title(title, fontsize=16)

                labels_kets = ["|0>", "|1>"]
                labels_bras = ["<0|", "<1|"]
                ax.set_xticks([0.4, 1.4])
                ax.set_yticks([0.4, 1.4])
                ax.set_xticklabels(labels_kets)
                ax.set_yticklabels(labels_bras)
                ax.set_zlabel("Value")

                z_max = np.max(np.abs(data))
                if z_max < 1e-9:
                    z_max = 1.0
                ax.set_zlim(-z_max, z_max)

                ax.plot_surface(
                    np.array([[0, 2], [0, 2]]),
                    np.array([[0, 0], [2, 2]]),
                    np.zeros((2, 2)),
                    alpha=0.2,
                    color="k",
                )

            ax1 = fig.add_subplot(1, 2, 1, projection="3d")
            plot_3d_bar(ax1, rho_real, norm_real, "Real($\\rho$)")

            ax2 = fig.add_subplot(1, 2, 2, projection="3d")
            plot_3d_bar(ax2, rho_imag, norm_imag, "Imag($\\rho$)")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
            return fig, (ax1, ax2)

        else:
            print(f"錯誤: 未知的 plot_type '{plot_type}'. 請選擇 '2d' 或 '3d'.")
            return None, None

    # ... (saveLabber 方法保持不變) ...
    def saveLabber(self, qb_idx, yoko_value=None):
        """
        Save the tomography data to Labber HDF5 file.
        """
        if self.tomo_data_raw is None:
            print("No data to save. Run the experiment first using .run()")
            return

        expt_name = "s015_Tomography_ge" + f"_Q{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_value)

        # Create a detailed comment
        dict_val = yml_comment(self.cfg)
        comment = (
            f"{dict_val}\n"
            f"--- Tomography Results ---\n"
            f"Prepared State Pulse: {self.prep_pulse_name}\n"
            f"IQ Ground (|0>): {self.iq_g}\n"
            f"IQ Excited (|1>): {self.iq_e}\n"
            f"--- Expectation Values ---\n"
            f"<X>: {self.expect_values['X']}\n"
            f"<Y>: {self.expect_values['Y']}\n"
            f"<Z>: {self.expect_values['Z']}\n"
            f"--- Reconstructed Density Matrix (MLE) ---\n"
            f"{self.rho_mle}"  # This will print the numpy array
        )

        # Prepare data for HDF5
        x_vals = np.array([0, 1, 2])  # Representing X, Y, Z
        z_vals = np.array(
            [self.tomo_data_raw["X"], self.tomo_data_raw["Y"], self.tomo_data_raw["Z"]]
        )

        hdf5_generator(
            filepath=file_path,
            x_info={"name": "Axis", "unit": "None (0=X, 1=Y, 2=Z)", "values": x_vals},
            z_info={"name": "Signal", "unit": "ADC unit", "values": z_vals},
            comment=comment,
            tag="Tomography",
        )
        print(f"Data save to {file_path}")
