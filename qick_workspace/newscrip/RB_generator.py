"""
single_qubit_rb.py
==================
Single-qubit Randomized Benchmarking — Clifford sequence generator
-------------------------------------------------------------------
Gates : I, X, Y, X/2, Y/2, -X/2, -Y/2  (numpy / scipy only)

Public API
----------
    rb = SingleQubitRB()

    # Standard RB — one sample
    seq = rb.generate_single_qb_rb(n_cliffords=10, n_samples=30)

    # Interleaved RB — interleave_gate ∈ {"X", "Y", "X/2", "Y/2"}
    seq = rb.generate_single_qb_rb(n_cliffords=10, n_samples=30,
                                   interleave_gate="X")

    Each element of `seq` is an RBSequence dataclass:
        .clifford_indices  list[int]       random Clifford indices
        .recovery_index    int             inverse / recovery Clifford
        .pulse_sequence    list[str]       flat primitive-gate list
                                           (includes recovery gate)
        .U_accumulated     np.ndarray 2×2  product of random (+ interleaved) gates
        .U_recovery        np.ndarray 2×2  recovery gate matrix

IRB analysis
------------
    epc, f_gate, bound = SingleQubitRB.epc_from_rb_irb(p_ref, p_irb)

    where p_ref / p_irb are the depolarizing parameters from fitting
        P(m) = A · p^m + B
    to the standard-RB and IRB survival-probability curves respectively.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Union


# ══════════════════════════════════════════════════════════════════
# 0.  Return-value container
# ══════════════════════════════════════════════════════════════════

@dataclass
class RBSequence:
    """One complete RB (or IRB) gate sequence including the recovery gate."""
    clifford_indices : List[int]        # random Clifford indices
    recovery_index   : int              # index of recovery Clifford
    pulse_sequence   : List[str]        # flat primitive list (random + recovery)
    U_accumulated    : np.ndarray       # product of random (+ interleaved) gates
    U_recovery       : np.ndarray       # recovery gate matrix


# ══════════════════════════════════════════════════════════════════
# 1.  SingleQubitRB
# ══════════════════════════════════════════════════════════════════

class SingleQubitRB:
    """
    Single-qubit Clifford-group RB / IRB sequence generator.

    The Clifford group (24 elements) is parametrised as SU(2) matrices.
    Each element is pre-decomposed into a short pulse sequence drawn from
    the gate set {I, X, Y, X/2, Y/2, -X/2, -Y/2}.

    Convention
    ----------
    pulse_sequence = [g1, g2, …, gN]  means  g1 is applied FIRST to |ψ⟩,
    i.e.  U_total = U_gN · … · U_g2 · U_g1.

    Allowed interleave gates
    ------------------------
    "X", "Y", "X/2", "Y/2"  (matches typical superconducting qubit gate sets)
    """

    # ── 1a. Primitive gate matrices ───────────────────────────────

    @staticmethod
    def _Rx(theta: float) -> np.ndarray:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

    @staticmethod
    def _Ry(theta: float) -> np.ndarray:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    # ── 1b. Gate-name → matrix map ────────────────────────────────

    @classmethod
    def _build_gate_matrix(cls) -> dict:
        return {
            "I"    : np.eye(2, dtype=complex),
            "X"    : cls._Rx(np.pi),
            "Y"    : cls._Ry(np.pi),
            "X/2"  : cls._Rx( np.pi / 2),
            "Y/2"  : cls._Ry( np.pi / 2),
            "-X/2" : cls._Rx(-np.pi / 2),
            "-Y/2" : cls._Ry(-np.pi / 2),
        }

    # ── 1c. Allowed interleave gates ──────────────────────────────

    ALLOWED_INTERLEAVE = {"X", "Y", "X/2", "Y/2"}

    # ── 1d. 24 Clifford decompositions ───────────────────────────
    #
    #  Source / convention:
    #    Barends et al. (2014)  /  Eppstein (2005)
    #
    #  Every Clifford is written as a product of primitives.
    #  Reading the list left-to-right = gate applied first to |ψ⟩.
    #
    #  The 24 elements cover all 6 stabiliser states × 4 Pauli phases.
    #  Grouped by the axis they map |0⟩ → {±X, ±Y, ±Z}.

    _RAW_CLIFFORDS = [
        # ── Identity / Paulis ──────────────────────────────────────────────
        ("C0",  ["I"]),                       # I
        ("C1",  ["X"]),                       # X
        ("C2",  ["Y"]),                       # Y
        ("C3",  ["Y", "X"]),                  # Z  (= X·Y up to phase; apply Y first)

        # ── π/2 rotations ─────────────────────────────────────────────────
        ("C4",  ["X/2"]),
        ("C5",  ["-X/2"]),
        ("C6",  ["Y/2"]),
        ("C7",  ["-Y/2"]),

        # ── Hadamard-like (swap axes) ──────────────────────────────────────
        # H = Y/2 · X  (apply X first, then Y/2)
        ("C8",  ["X",  "Y/2"]),               # H  (Z↔X)
        ("C9",  ["X",  "-Y/2"]),              # H† (alt)
        ("C10", ["Y",  "X/2"]),               # (Z↔Y variant)
        ("C11", ["Y",  "-X/2"]),

        # ── 2π/3 face-diagonal (S-like) ───────────────────────────────────
        ("C12", ["X/2",  "Y/2"]),
        ("C13", ["X/2",  "-Y/2"]),
        ("C14", ["-X/2", "Y/2"]),
        ("C15", ["-X/2", "-Y/2"]),

        # ── remaining face-diagonal rotations ─────────────────────────────
        ("C16", ["Y/2",  "X/2"]),
        ("C17", ["-Y/2", "X/2"]),
        ("C18", ["Y/2",  "-X/2"]),
        ("C19", ["-Y/2", "-X/2"]),

        # ── edge-midpoint rotations ────────────────────────────────────────
        ("C20", ["X/2",  "Y/2",  "X/2"]),
        ("C21", ["X/2",  "-Y/2", "X/2"]),
        ("C22", ["-X/2", "Y/2",  "X/2"]),
        ("C23", ["-X/2", "-Y/2", "X/2"]),
    ]

    # ─────────────────────────────────────────────────────────────
    def __init__(self) -> None:
        self._gate_matrix   = self._build_gate_matrix()
        self._clifford_table= self._build_clifford_table()
        self._inverse_table = self._build_inverse_table()   # pre-computed
        self._verify_group_closure()                         # sanity check

    # ── 2. Build Clifford table ───────────────────────────────────

    def _seq_to_matrix(self, pulses: List[str]) -> np.ndarray:
        """
        pulses = [g1, g2, …]  →  U = U_gN · … · U_g1
        (first gate in list applied first to state)
        """
        U = np.eye(2, dtype=complex)
        for g in pulses:
            U = self._gate_matrix[g] @ U
        return U

    def _build_clifford_table(self) -> List[dict]:
        table = []
        for name, pulses in self._RAW_CLIFFORDS:
            U = self._seq_to_matrix(pulses)
            table.append({"name": name, "pulses": pulses, "matrix": U})
        self._verify_24_distinct(table)
        return table

    # ── 3. Verification helpers ───────────────────────────────────

    @staticmethod
    def _matrix_distance(A: np.ndarray, B: np.ndarray) -> float:
        """
        Distance between two U(2) matrices up to global phase.
        Checks both +phase and −phase (covers SU(2) ↔ U(2) ambiguity).
        """
        return min(np.linalg.norm(A - B, 'fro'),
                   np.linalg.norm(A + B, 'fro'))

    def _verify_24_distinct(self, table: List[dict]) -> None:
        assert len(table) == 24, f"Expected 24 Cliffords, got {len(table)}"
        for i in range(24):
            for j in range(i + 1, 24):
                d = self._matrix_distance(table[i]["matrix"], table[j]["matrix"])
                assert d > 1e-6, (
                    f"Duplicate Cliffords: {table[i]['name']} and "
                    f"{table[j]['name']}  (dist={d:.2e})\n"
                    f"Check _RAW_CLIFFORDS decompositions."
                )

    def _verify_group_closure(self) -> None:
        """
        Verify that the 24 elements are closed under multiplication,
        i.e. they really form the single-qubit Clifford group.
        This catches any wrong decomposition in _RAW_CLIFFORDS.
        """
        for i, ci in enumerate(self._clifford_table):
            for j, cj in enumerate(self._clifford_table):
                prod = ci["matrix"] @ cj["matrix"]
                _, dist = self._find_closest_clifford(prod)
                assert dist < 1e-6, (
                    f"Group closure FAILED: C{i} · C{j} not in Clifford group "
                    f"(dist={dist:.2e}).\n"
                    f"One or more entries in _RAW_CLIFFORDS is wrong."
                )

    # ── 4. Inverse lookup table (pre-computed, O(1) access) ───────

    def _build_inverse_table(self) -> dict:
        """
        CLIFFORD_INVERSE[i] = j  such that  C_j · C_i ≈ I  (up to global phase).
        Pre-computed once at construction; used by all sequence generators.
        """
        inv = {}
        I2  = np.eye(2, dtype=complex)
        for i, ci in enumerate(self._clifford_table):
            for j, cj in enumerate(self._clifford_table):
                if self._matrix_distance(cj["matrix"] @ ci["matrix"], I2) < 1e-6:
                    inv[i] = j
                    break
            assert i in inv, (
                f"Could not find inverse for Clifford C{i} — "
                "group may be incomplete."
            )
        assert len(inv) == 24
        return inv

    # ── 5. Closest-Clifford lookup ────────────────────────────────

    def _find_closest_clifford(self, U: np.ndarray):
        """Return (index, distance) of the nearest Clifford to U."""
        best_idx, best_dist = 0, np.inf
        for idx, cliff in enumerate(self._clifford_table):
            d = self._matrix_distance(U, cliff["matrix"])
            if d < best_dist:
                best_dist = d
                best_idx  = idx
        return best_idx, best_dist

    # ── 6. SU(2) normalisation ────────────────────────────────────

    @staticmethod
    def _normalize_su2(U: np.ndarray) -> np.ndarray:
        """
        Project U onto SU(2) by removing the global phase so that det = +1.

        Uses  U / sqrt(det U).  Because det(U) for a unitary is e^{iφ},
        sqrt gives e^{iφ/2}, which is well-defined on the principal branch.
        The ±1 phase ambiguity that remains is handled by _matrix_distance.
        """
        det = np.linalg.det(U)
        assert abs(abs(det) - 1.0) < 1e-8, (
            f"Matrix is not unitary: |det| = {abs(det):.6f}"
        )
        return U / np.sqrt(det)

    # ── 7. Standard RB sequence generator ────────────────────────

    def _generate_rb_sequence_single(
        self,
        n_cliffords : int,
        rng         : np.random.Generator,
    ) -> RBSequence:
        """
        Generate one standard-RB sequence of length n_cliffords + 1.

        Returns
        -------
        RBSequence  with pulse_sequence = [random_gates …, recovery_gate]
        """
        indices = rng.integers(0, 24, size=n_cliffords).tolist()

        # Accumulate U = C_m · … · C_1
        U_acc = np.eye(2, dtype=complex)
        for idx in indices:
            U_acc = self._clifford_table[idx]["matrix"] @ U_acc

        # Recovery = (U_acc)^{-1}  via pre-computed inverse table
        # First find which Clifford U_acc is, then look up its inverse.
        acc_idx, dist = self._find_closest_clifford(U_acc)
        assert dist < 1e-6, (
            f"Accumulated unitary not in Clifford group (dist={dist:.2e}).\n"
            "This should never happen — check _verify_group_closure output."
        )
        recovery_index = self._inverse_table[acc_idx]

        # Build flat pulse list
        pulses = []
        for idx in indices:
            pulses.extend(self._clifford_table[idx]["pulses"])
        pulses.extend(self._clifford_table[recovery_index]["pulses"])

        return RBSequence(
            clifford_indices = indices,
            recovery_index   = recovery_index,
            pulse_sequence   = pulses,
            U_accumulated    = U_acc,
            U_recovery       = self._clifford_table[recovery_index]["matrix"],
        )

    # ── 8. Interleaved RB sequence generator ─────────────────────

    def _generate_irb_sequence_single(
        self,
        n_cliffords    : int,
        interleave_gate: str,
        rng            : np.random.Generator,
    ) -> RBSequence:
        """
        Generate one IRB sequence.

        Sequence structure:
            C_1 · G · C_2 · G · … · C_m · G · C_recovery

        C_recovery inverts the entire accumulated product (G·C_m)·…·(G·C_1).

        Parameters
        ----------
        interleave_gate : str — must be one of ALLOWED_INTERLEAVE
        """
        if interleave_gate not in self.ALLOWED_INTERLEAVE:
            raise ValueError(
                f"interleave_gate={interleave_gate!r} is not allowed.\n"
                f"Choose from: {self.ALLOWED_INTERLEAVE}"
            )

        U_G      = self._gate_matrix[interleave_gate]
        G_pulses = [interleave_gate]

        indices = rng.integers(0, 24, size=n_cliffords).tolist()

        # Accumulate U = (G·C_m) · … · (G·C_1)
        U_acc = np.eye(2, dtype=complex)
        for idx in indices:
            U_C   = self._clifford_table[idx]["matrix"]
            U_acc = U_G @ U_C @ U_acc

        # Find recovery via inverse lookup
        # U_acc may not be in the Clifford group by itself (G is not necessarily
        # a Clifford), but (G·C) is always a Clifford if G ∈ gate set that
        # generates Clifford group. We still use find_closest_clifford here.
        U_inv = self._normalize_su2(U_acc.conj().T)
        recovery_index, dist = self._find_closest_clifford(U_inv)
        assert dist < 1e-6, (
            f"IRB recovery Clifford not found (dist={dist:.2e}).\n"
            "U_acc may not lie in the Clifford group for this interleave gate."
        )

        # Build flat pulse list: C_i pulses → G pulses → … → recovery pulses
        pulses = []
        for idx in indices:
            pulses.extend(self._clifford_table[idx]["pulses"])
            pulses.extend(G_pulses)
        pulses.extend(self._clifford_table[recovery_index]["pulses"])

        return RBSequence(
            clifford_indices = indices,
            recovery_index   = recovery_index,
            pulse_sequence   = pulses,
            U_accumulated    = U_acc,
            U_recovery       = self._clifford_table[recovery_index]["matrix"],
        )

    # ── 9. Public API ─────────────────────────────────────────────

    def generate_single_qb_rb(
        self,
        n_cliffords    : int,
        n_samples      : int,
        interleave_gate: Optional[str] = None,
        seed           : Optional[int] = None,
    ) -> List[RBSequence]:
        """
        Generate n_samples independent RB (or IRB) sequences.

        Parameters
        ----------
        n_cliffords : int
            Number of random Clifford gates per sequence (NOT counting recovery).
        n_samples : int
            Number of independent random sequences to generate.
        interleave_gate : str or None
            If None → standard RB.
            Otherwise → IRB with this gate interleaved after each Clifford.
            Allowed values: "X", "Y", "X/2", "Y/2".
        seed : int or None
            Master seed for reproducibility.  Each of the n_samples sequences
            uses a child seed derived from this master, so results are
            fully reproducible regardless of n_samples.

        Returns
        -------
        List[RBSequence]
            Length = n_samples.  Each RBSequence contains:
              .clifford_indices  — list[int], length n_cliffords
              .recovery_index    — int
              .pulse_sequence    — list[str], full flat gate list
                                   (random gates [+ interleave] + recovery)
              .U_accumulated     — 2×2 complex ndarray
              .U_recovery        — 2×2 complex ndarray

        Notes
        -----
        pulse_sequence convention:
            Standard RB  : [C1_pulses, C2_pulses, …, Cm_pulses, recovery_pulses]
            IRB          : [C1_pulses, G_pulses, C2_pulses, G_pulses, …,
                            Cm_pulses, G_pulses, recovery_pulses]
        """
        rng = np.random.default_rng(seed)

        sequences = []
        for _ in range(n_samples):
            child_seed = int(rng.integers(0, 2**31))
            child_rng  = np.random.default_rng(child_seed)

            if interleave_gate is None:
                seq = self._generate_rb_sequence_single(n_cliffords, child_rng)
            else:
                seq = self._generate_irb_sequence_single(
                    n_cliffords, interleave_gate, child_rng
                )
            sequences.append(seq)

        return sequences

    # ── 10. Sequence verification ─────────────────────────────────

    def verify_sequence(
        self,
        seq     : RBSequence,
        verbose : bool = True,
    ) -> float:
        """
        Verify that the full sequence (random + recovery) returns to identity.

        Returns residual Frobenius norm (should be < 1e-10 for correct sequences).
        """
        U = np.eye(2, dtype=complex)
        for idx in seq.clifford_indices:
            U = self._clifford_table[idx]["matrix"] @ U
        U = self._clifford_table[seq.recovery_index]["matrix"] @ U

        phase    = U[0, 0] / abs(U[0, 0]) if abs(U[0, 0]) > 1e-10 else 1.0
        residual = np.linalg.norm(U / phase - np.eye(2), 'fro')

        if verbose:
            print(f"  Full sequence matrix (phase-normalised):\n{np.round(U / phase, 6)}")
            print(f"  Residual from Identity: {residual:.2e}")

        return residual

    def verify_irb_sequence(
        self,
        seq             : RBSequence,
        interleave_gate : str,
        verbose         : bool = True,
    ) -> float:
        """
        Verify that the full IRB sequence returns to identity.

        Returns residual Frobenius norm.
        """
        if interleave_gate not in self.ALLOWED_INTERLEAVE:
            raise ValueError(f"interleave_gate must be one of {self.ALLOWED_INTERLEAVE}")

        U_G = self._gate_matrix[interleave_gate]

        U = np.eye(2, dtype=complex)
        for idx in seq.clifford_indices:
            U = U_G @ self._clifford_table[idx]["matrix"] @ U
        U = self._clifford_table[seq.recovery_index]["matrix"] @ U

        phase    = U[0, 0] / abs(U[0, 0]) if abs(U[0, 0]) > 1e-10 else 1.0
        residual = np.linalg.norm(U / phase - np.eye(2), 'fro')

        if verbose:
            print(f"  Full IRB matrix (phase-normalised):\n{np.round(U / phase, 6)}")
            print(f"  Residual from Identity: {residual:.2e}")

        return residual

    # ── 11. EPC / gate fidelity calculation ──────────────────────

    @staticmethod
    def epc_from_rb_irb(
        p_ref : float,
        p_irb : float,
        d     : int = 2,
    ) -> tuple:
        """
        Compute Error Per Clifford (EPC) of the interleaved gate.

        Parameters
        ----------
        p_ref : float
            Depolarizing parameter from standard-RB fit:  P(m) = A · p_ref^m + B
        p_irb : float
            Depolarizing parameter from IRB fit:          P(m) = A · p_irb^m + B
        d : int
            Hilbert-space dimension (d = 2 for single qubit).

        Returns
        -------
        epc       : float   error per gate  =  (d-1)/d · (1 − p_irb/p_ref)
        f_gate    : float   average gate fidelity  =  1 − epc
        epc_bound : float   1-sigma upper bound (Magesan et al. 2012)

        Notes
        -----
        DO NOT confuse p_ref with the error rate r_ref = (d-1)/d · (1 − p_ref).
        The fitting model is  P(m) = A · p^m + B,  NOT  A · r^m + B.
        """
        epc       = (d - 1) / d * (1 - p_irb / p_ref)
        f_gate    = 1.0 - epc
        epc_bound = (d - 1) / d * (abs(p_ref - p_irb) / p_ref + (1 - p_ref))
        return epc, f_gate, epc_bound


# ══════════════════════════════════════════════════════════════════
# Quick self-test
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rb = SingleQubitRB()
    print("Clifford group verified (24 distinct, group-closed).\n")

    # ── Standard RB ──────────────────────────────────────────────
    print("=== Standard RB (m=10, n_samples=5) ===")
    seqs = rb.generate_single_qb_rb(n_cliffords=10, n_samples=5, seed=42)
    for i, s in enumerate(seqs):
        res = rb.verify_sequence(s, verbose=False)
        print(f"  Sample {i}: {len(s.pulse_sequence):3d} pulses | "
              f"residual = {res:.2e} | "
              f"pulses = {s.pulse_sequence}")

    print()

    # ── IRB (interleave X) ────────────────────────────────────────
    print("=== IRB  (m=5, n_samples=3, gate='X') ===")
    iseqs = rb.generate_single_qb_rb(
        n_cliffords=5, n_samples=3, interleave_gate="X", seed=0
    )
    for i, s in enumerate(iseqs):
        res = rb.verify_irb_sequence(s, interleave_gate="X", verbose=False)
        print(f"  Sample {i}: {len(s.pulse_sequence):3d} pulses | "
              f"residual = {res:.2e} | "
              f"pulses = {s.pulse_sequence}")

    print()

    # ── IRB allowed gates ─────────────────────────────────────────
    print("=== IRB gate set check ===")
    for g in ["X", "Y", "X/2", "Y/2"]:
        s = rb.generate_single_qb_rb(8, 1, interleave_gate=g, seed=7)[0]
        res = rb.verify_irb_sequence(s, g, verbose=False)
        print(f"  gate={g!r:5s}  residual={res:.2e}  {'✓' if res < 1e-9 else '✗ FAIL'}")

    print()

    # ── EPC example ──────────────────────────────────────────────
    print("=== EPC demo ===")
    epc, f_gate, bound = SingleQubitRB.epc_from_rb_irb(p_ref=0.99, p_irb=0.98)
    print(f"  p_ref=0.99, p_irb=0.98")
    print(f"  EPC     = {epc:.4f}")
    print(f"  F_gate  = {f_gate:.4f}")
    print(f"  Bound   = {bound:.4f}")