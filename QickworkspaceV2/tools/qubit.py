import qutip as qt
import numpy as np


def flux_to_phi(x, flux_range, phi_range):
    """Map an external flux value to a reduced flux bias phi in [0, 0.5]."""
    src_min, src_max = flux_range
    tgt_min, tgt_max = phi_range
    val = tgt_min + (x - src_min) / (src_max - src_min) * (tgt_max - tgt_min)
    val = val % 1
    return val if val <= 0.5 else 1 - val


def phi_to_flux(phi, flux_range, phi_range):
    """Map a reduced flux bias phi back to an external flux value."""
    src_min, src_max = phi_range
    tgt_min, tgt_max = flux_range
    flux_val = tgt_min + (phi - src_min) / (src_max - src_min) * (tgt_max - tgt_min)
    return flux_val


class Fluxonium:
    """
    Fluxonium qubit Hamiltonian and transition frequency calculator.

    Parameters
    ----------
    EJ : float
        Josephson energy in GHz.
    EC : float
        Charging energy in GHz.
    EL : float
        Inductive energy in GHz.
    dimention : int
        Hilbert space dimension (number of Fock states).
    flux : float
        Reduced external flux bias phi in [0, 0.5].
    """

    def __init__(self, EJ, EC, EL, dimention, flux) -> None:
        self.EJ = EJ
        self.EC = EC
        self.EL = EL
        self.dim = dimention
        self.phi = flux

    @property
    def phi_osc(self):
        return ((8 * self.EC) / self.EL) ** (1 / 4)

    @property
    def creation(self):
        return qt.create(self.dim)

    @property
    def destroy(self):
        return qt.destroy(self.dim)

    @property
    def n_op(self):
        return (-1j / (np.sqrt(2) * self.phi_osc)) * (self.creation - self.destroy)

    @property
    def phi_op(self):
        return (self.phi_osc / np.sqrt(2)) * (self.creation + self.destroy)

    def J_term(self):
        phi_ext_op = qt.qeye(self.dim) * (2 * np.pi * self.phi)
        return -self.EJ * (self.phi_op - phi_ext_op).cosm()

    def C_term(self):
        return 4 * self.EC * self.n_op**2

    def L_term(self):
        return 0.5 * self.EL * self.phi_op**2

    def hamiltonian(self):
        return self.J_term() + self.C_term() + self.L_term()

    def f01(self):
        energies = self.hamiltonian().eigenenergies()
        return energies[1] - energies[0]

    def f12(self):
        energies = self.hamiltonian().eigenenergies()
        return energies[2] - energies[1]

    def f02(self):
        energies = self.hamiltonian().eigenenergies()
        return energies[2] - energies[0]

    def f03(self):
        energies = self.hamiltonian().eigenenergies()
        return energies[3] - energies[0]

    def fij(self, i, j):
        if i < j:
            raise ValueError(f"Invalid transition: i={i} must be >= j={j}")
        energies = self.hamiltonian().eigenenergies()
        return energies[i] - energies[j]

    def cooling_to_g(self, fr):
        return {"f12": self.fij(2, 1), "f0g1": fr - self.fij(2, 0)}

    def cooling_to_e(self, fr):
        return {"f03": self.fij(3, 0), "fhe1": fr - self.fij(3, 1)}
