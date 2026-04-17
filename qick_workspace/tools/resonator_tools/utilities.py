from __future__ import annotations
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]


def Watt2dBm(x: float | FloatArray) -> float | FloatArray:
    """
    converts from units of watts to dBm
    """
    return 10.0 * np.log10(x * 1000.0)


def dBm2Watt(x: float | FloatArray) -> float | FloatArray:
    """
    converts from units of dBm to watts
    """
    return 10 ** (x / 10.0) / 1000.0


class plotting(object):
    """
    some helper functions for plotting
    """

    # TODO: refactor architecture using composition instead of inheritance, so that plotting is a separate class that can be used by any port type without needing to inherit from it
    def plotall(self) -> None:
        # remove electrical delay for a cleaner visualization of the raw vs fit
        if hasattr(self, '_cancel_delay_phase') and self._cancel_delay_phase is not None:
            delay_phase = self._cancel_delay_phase
        else:
            delay = getattr(self, '_delay', 0.0)
            delay_phase = np.exp(2j * np.pi * delay * np.array(self.f_data))  # type: ignore
        
        z_raw_nodelay = self.z_data_raw * delay_phase  # type: ignore
        z_sim_nodelay = self.z_data_sim * delay_phase  # type: ignore
        
        real = z_raw_nodelay.real
        imag = z_raw_nodelay.imag
        real2 = z_sim_nodelay.real
        imag2 = z_sim_nodelay.imag
        
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(2, 2)
        
        ax_amp = fig.add_subplot(gs[:, 0])
        ax_iq = fig.add_subplot(gs[0, 1])
        ax_phase = fig.add_subplot(gs[1, 1])
        
        # Plot Amplitude (Left bridging 2 rows)
        ax_amp.plot(self.f_data * 1e-9, np.absolute(z_raw_nodelay), label="rawdata")  # type: ignore
        ax_amp.plot(self.f_data * 1e-9, np.absolute(z_sim_nodelay), label="fit")  # type: ignore
        ax_amp.set_xlabel("f (GHz)")
        ax_amp.set_ylabel("|S21|")
        ax_amp.legend()

        # Plot IQ circle (Right Top)
        ax_iq.plot(real, imag, label="rawdata")
        ax_iq.plot(real2, imag2, label="fit")
        # Ensure IQ plot is somewhat square to not distort the circle visualization
        ax_iq.set_aspect('equal', 'datalim') 
        ax_iq.set_xlabel("Re(S21)")
        ax_iq.set_ylabel("Im(S21)")
        ax_iq.legend()
        
        # Plot Phase (Right Bottom)
        ax_phase.plot(self.f_data * 1e-9, np.angle(z_raw_nodelay), label="rawdata")  # type: ignore
        ax_phase.plot(self.f_data * 1e-9, np.angle(z_sim_nodelay), label="fit")  # type: ignore
        ax_phase.set_xlabel("f (GHz)")
        ax_phase.set_ylabel("arg(|S21|)")
        ax_phase.legend()

        # Add Title with fit parameters if available
        if hasattr(self, 'fitresults') and self.fitresults and self.fitresults.get('fr') is not None:
            fr = self.fitresults.get('fr', 0.0)
            Ql = self.fitresults.get('Ql', 0.0)
            Qc = self.fitresults.get('Qc_dia_corr', self.fitresults.get('absQc', self.fitresults.get('Qc', 0.0)))
            Qi = self.fitresults.get('Qi_dia_corr', self.fitresults.get('Qi', 0.0))
            kappa = fr / Ql if Ql else 0.0
            
            title_str = f"fr = {fr*1e-9:.5f} GHz  |  $\kappa/2\pi$ = {kappa*1e-6:.2f} MHz\nQi = {Qi:.1f}  |  Qc = {Qc:.1f}"
            fig.suptitle(title_str, fontsize=12)

        plt.tight_layout()
        plt.show()

    def plotcalibrateddata(self) -> None:
        real = self.z_data.real  # type: ignore
        imag = self.z_data.imag  # type: ignore
        plt.subplot(221)
        plt.plot(real, imag, label="rawdata")
        plt.xlabel("Re(S21)")
        plt.ylabel("Im(S21)")
        plt.legend()
        plt.subplot(222)
        plt.plot(self.f_data * 1e-9, np.absolute(self.z_data), label="rawdata")  # type: ignore
        plt.xlabel("f (GHz)")
        plt.ylabel("|S21|")
        plt.legend()
        plt.subplot(223)
        plt.plot(self.f_data * 1e-9, np.angle(self.z_data), label="rawdata")  # type: ignore
        plt.xlabel("f (GHz)")
        plt.ylabel("arg(|S21|)")
        plt.legend()
        plt.show()

    def plotrawdata(self) -> None:
        real = self.z_data_raw.real  # type: ignore
        imag = self.z_data_raw.imag  # type: ignore
        plt.subplot(221)
        plt.plot(real, imag, label="rawdata")
        plt.xlabel("Re(S21)")
        plt.ylabel("Im(S21)")
        plt.legend()
        plt.subplot(222)
        plt.plot(self.f_data * 1e-9, np.absolute(self.z_data_raw), label="rawdata")  # type: ignore
        plt.xlabel("f (GHz)")
        plt.ylabel("|S21|")
        plt.legend()
        plt.subplot(223)
        plt.plot(self.f_data * 1e-9, np.angle(self.z_data_raw), label="rawdata")  # type: ignore
        plt.xlabel("f (GHz)")
        plt.ylabel("arg(|S21|)")
        plt.legend()
        plt.show()


class save_load(object):
    """
    procedures for loading and saving data used by other classes
    """

    def _ConvToCompl(
        self,
        x: FloatArray,
        y: FloatArray,
        dtype: str,
    ) -> ComplexArray:
        """
        dtype = 'realimag', 'dBmagphaserad', 'linmagphaserad', 'dBmagphasedeg', 'linmagphasedeg'
        """
        if dtype == "realimag":
            return x + 1j * y
        elif dtype == "linmagphaserad":
            return (x * np.exp(1j * y)).astype(np.complex128)
        elif dtype == "dBmagphaserad":
            return (10 ** (x / 20.0) * np.exp(1j * y)).astype(np.complex128)
        elif dtype == "linmagphasedeg":
            return (x * np.exp(1j * y / 180.0 * np.pi)).astype(np.complex128)
        elif dtype == "dBmagphasedeg":
            return (10 ** (x / 20.0) * np.exp(1j * y / 180.0 * np.pi)).astype(
                np.complex128
            )
        else:
            raise ValueError(
                "Undefined input type! Use 'realimag', 'dBmagphaserad', 'linmagphaserad', 'dBmagphasedeg' or 'linmagphasedeg'."
            )

    def add_data(self, f_data: FloatArray, z_data: ComplexArray) -> None:
        self.f_data = np.array(f_data)
        self.z_data_raw = np.array(z_data)

    def cut_data(self, f1: float, f2: float) -> None:
        def findpos(f_data: FloatArray, val: float) -> int:
            pos = 0
            for i in range(len(f_data)):
                if f_data[i] < val:
                    pos = i
            return pos

        pos1 = findpos(self.f_data, f1)
        pos2 = findpos(self.f_data, f2)
        self.f_data = self.f_data[pos1:pos2]
        self.z_data_raw = self.z_data_raw[pos1:pos2]  # type: ignore

    def add_fromtxt(
        self,
        fname: str,
        dtype: str,
        header_rows: int,
        usecols: tuple[int, int, int] = (0, 1, 2),
        fdata_unit: float = 1.0,
        delimiter: str | None = None,
    ) -> None:
        """
        dtype = 'realimag', 'dBmagphaserad', 'linmagphaserad', 'dBmagphasedeg', 'linmagphasedeg'
        """
        data = np.loadtxt(
            fname, usecols=usecols, skiprows=header_rows, delimiter=delimiter
        )
        self.f_data = data[:, 0] * fdata_unit
        self.z_data_raw = self._ConvToCompl(data[:, 1], data[:, 2], dtype=dtype)

    def add_fromhdf(self) -> None:
        pass

    def add_froms2p(
        self,
        fname: str,
        y1_col: int,
        y2_col: int,
        dtype: str,
        fdata_unit: float = 1.0,
        delimiter: str | None = None,
    ) -> None:
        """
        dtype = 'realimag', 'dBmagphaserad', 'linmagphaserad', 'dBmagphasedeg', 'linmagphasedeg'
        """
        if dtype == "dBmagphasedeg" or dtype == "linmagphasedeg":
            phase_conversion = 1.0 / 180.0 * np.pi
        else:
            phase_conversion = 1.0
        with open(fname) as f:
            lines = f.readlines()
        z_data_raw = []
        f_data = []
        if dtype == "realimag":
            for line in lines:
                if (line != "\n") and (line[0] != "#") and (line[0] != "!"):
                    lineinfo = line.split(delimiter)
                    f_data.append(float(lineinfo[0]) * fdata_unit)
                    z_data_raw.append(
                        complex(float(lineinfo[y1_col]), float(lineinfo[y2_col]))
                    )
        elif dtype == "linmagphaserad" or dtype == "linmagphasedeg":
            for line in lines:
                if (
                    (line != "\n")
                    and (line[0] != "#")
                    and (line[0] != "!")
                    and (line[0] != "M")
                    and (line[0] != "P")
                ):
                    lineinfo = line.split(delimiter)
                    f_data.append(float(lineinfo[0]) * fdata_unit)
                    z_data_raw.append(
                        float(lineinfo[y1_col])
                        * np.exp(
                            complex(0.0, phase_conversion * float(lineinfo[y2_col]))
                        )
                    )
        elif dtype == "dBmagphaserad" or dtype == "dBmagphasedeg":
            for line in lines:
                if (
                    (line != "\n")
                    and (line[0] != "#")
                    and (line[0] != "!")
                    and (line[0] != "M")
                    and (line[0] != "P")
                ):
                    lineinfo = line.split(delimiter)
                    f_data.append(float(lineinfo[0]) * fdata_unit)
                    linamp = 10 ** (float(lineinfo[y1_col]) / 20.0)
                    z_data_raw.append(
                        linamp
                        * np.exp(
                            complex(0.0, phase_conversion * float(lineinfo[y2_col]))
                        )
                    )
        else:
            raise ValueError(
                "Undefined input type! Use 'realimag', 'dBmagphaserad', 'linmagphaserad', 'dBmagphasedeg' or 'linmagphasedeg'."
            )
        self.f_data = np.array(f_data)
        self.z_data_raw = np.array(z_data_raw)

    def save_fitresults(self, fname: str) -> None:
        pass
