import qutip as qt
import numpy as np


def flux_to_phi(x, flux_range, phi_range):
    """
    Map an external flux value to a reduced flux bias phi in [0, 0.5].

    Applies a linear interpolation from *flux_range* to *phi_range*, then
    wraps the result into the fundamental domain ``[0, 0.5]`` using the
    symmetry of the cosine potential.

    Parameters
    ----------
    x : float
        External flux value (in whatever units *flux_range* uses).
    flux_range : tuple of float
        (min, max) bounds of the external flux axis.
    phi_range : tuple of float
        (min, max) bounds of the reduced flux axis (phi units).

    Returns
    -------
    phi : float
        Reduced flux bias in [0, 0.5].
    """
    src_min, src_max = flux_range
    tgt_min, tgt_max = phi_range
    val = tgt_min + (x - src_min) / (src_max - src_min) * (tgt_max - tgt_min)
    # wrap into [0, 0.5]
    val = val % 1
    return val if val <= 0.5 else 1 - val


def phi_to_flux(phi, flux_range, phi_range):
    """
    Map a reduced flux bias phi back to an external flux value.

    Assumes *phi* is already in [0, 0.5] and applies the inverse linear
    interpolation from *phi_range* to *flux_range*.

    Parameters
    ----------
    phi : float
        Reduced flux bias in [0, 0.5].
    flux_range : tuple of float
        (min, max) bounds of the external flux axis (output units).
    phi_range : tuple of float
        (min, max) bounds of the reduced flux axis (input units).

    Returns
    -------
    flux : float
        External flux value corresponding to *phi*.
    """
    src_min, src_max = phi_range
    tgt_min, tgt_max = flux_range
    # Since phi is in [0, 0.5], we need to choose which side of the symmetry to map to.
    # We'll pick the lower side (<= 0.5).
    val = phi  # already in [0, 0.5]
    # Map from phi_range to flux_range
    flux_val = tgt_min + (val - src_min) / (src_max - src_min) * (tgt_max - tgt_min)
    return flux_val


class Fluxonium:
    """
    Fluxonium qubit Hamiltonian and transition frequency calculator.

    Constructs the circuit Hamiltonian in the harmonic oscillator (Fock)
    basis and provides methods to compute energy levels and transition
    frequencies for a given external flux bias.

    Parameters
    ----------
    EJ : float
        Josephson energy in GHz.
    EC : float
        Charging energy in GHz.
    EL : float
        Inductive energy in GHz.
    dimention : int
        Hilbert space dimension (number of Fock states to include).
    flux : float
        Reduced external flux bias phi (in units of flux quantum, range [0, 0.5]).
    """

    def __init__(self, EJ, EC, EL, dimention, flux) -> None:
        self.EJ = EJ
        self.EC = EC
        self.EL = EL
        self.dim = dimention
        self.phi = flux

    @property
    def phi_osc(self):
        """Oscillator length (zero-point flux fluctuation amplitude)."""
        return ((8 * self.EC) / self.EL) ** (1 / 4)  # oscillator length

    @property
    def creation(self):
        """Bosonic creation operator in the Fock basis."""
        return qt.create(self.dim)

    @property
    def destroy(self):
        """Bosonic annihilation operator in the Fock basis."""
        return qt.destroy(self.dim)

    @property
    def n_op(self):
        """Charge (number) operator in the Fock basis."""
        return (-1j / (np.sqrt(2) * self.phi_osc)) * (
            self.creation - self.destroy
        )  # charge operator

    @property
    def phi_op(self):
        """Phase (flux) operator in the Fock basis."""
        return (self.phi_osc / np.sqrt(2)) * (
            self.creation + self.destroy
        )  # flux operator

    def J_term(self):
        """Return the Josephson cosine term of the Hamiltonian."""
        phi_ext_op = qt.qeye(self.dim) * (2 * np.pi * self.phi)
        return -self.EJ * (self.phi_op - phi_ext_op).cosm()

    def C_term(self):
        """Return the charging energy term of the Hamiltonian."""
        return 4 * self.EC * self.n_op**2

    def L_term(self):
        """Return the inductive energy term of the Hamiltonian."""
        return 0.5 * self.EL * self.phi_op**2

    def hamiltonian(self):
        """Return the full Fluxonium Hamiltonian operator."""
        return self.J_term() + self.C_term() + self.L_term()

    def f01(self):
        """
        Compute the |0> → |1> transition frequency.

        Returns
        -------
        f01 : float
            Transition frequency in GHz.
        """
        ham = self.hamiltonian()
        energies = ham.eigenenergies()
        return energies[1] - energies[0]

    def f12(self):
        """
        Compute the |1> → |2> transition frequency.

        Returns
        -------
        f12 : float
            Transition frequency in GHz.
        """
        ham = self.hamiltonian()
        energies = ham.eigenenergies()
        return energies[2] - energies[1]

    def f02(self):
        """
        Compute the |0> → |2> transition frequency.

        Returns
        -------
        f02 : float
            Transition frequency in GHz.
        """
        ham = self.hamiltonian()
        energies = ham.eigenenergies()
        return energies[2] - energies[0]

    def f03(self):
        """
        Compute the |0> → |3> transition frequency.

        Returns
        -------
        f03 : float
            Transition frequency in GHz.
        """
        ham = self.hamiltonian()
        energies = ham.eigenenergies()
        return energies[3] - energies[0]

    def fij(self, i, j):
        """
        Compute the transition frequency between energy levels *j* and *i*.

        Parameters
        ----------
        i : int
            Upper energy level index (must satisfy i >= j).
        j : int
            Lower energy level index.

        Returns
        -------
        fij : float
            Transition frequency E_i - E_j in GHz.

        Raises
        ------
        ValueError
            If *i* < *j*.
        """
        if i < j:
            raise ValueError(f"Invalid transition: i={i} must be >= j={j}")

        ham = self.hamiltonian()
        energies = ham.eigenenergies()
        return energies[i] - energies[j]

    def cooling_to_g(self, fr):
        """
        Compute drive frequencies needed to cool the qubit to the ground state.

        Returns the sideband cooling frequencies assuming a resonator at *fr*
        mediates the qubit–resonator interaction for the |f,0> ↔ |g,1> swap.

        Parameters
        ----------
        fr : float
            Readout resonator frequency in GHz.

        Returns
        -------
        freqs : dict
            Dictionary with keys:

            ``"f12"`` : float
                Qubit |1> → |2> transition frequency (GHz).
            ``"f0g1"`` : float
                Resonator-mediated |g,1> sideband frequency fr - f02 (GHz).
        """
        dict = {"f12": self.fij(2, 1), "f0g1": fr - self.fij(2, 0)}
        return dict

    def cooling_to_e(self, fr):
        """
        Compute drive frequencies needed to prepare the qubit in the excited state.

        Returns the sideband cooling frequencies for the |h,0> ↔ |e,1> swap
        assuming a resonator at *fr*.

        Parameters
        ----------
        fr : float
            Readout resonator frequency in GHz.

        Returns
        -------
        freqs : dict
            Dictionary with keys:

            ``"f03"`` : float
                Qubit |0> → |3> transition frequency (GHz).
            ``"fhe1"`` : float
                Resonator-mediated |e,1> sideband frequency fr - f13 (GHz).
        """
        dict = {"f03": self.fij(3, 0), "fhe1": fr - self.fij(3, 1)}
        return dict
