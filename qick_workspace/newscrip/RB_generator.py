"""
Single-qubit Randomized Benchmarking — Clifford sequence generator
===================================================================
Gates: I, X, Y, X/2, Y/2, -X/2, -Y/2  (no external packages beyond numpy/scipy)

Clifford group (24 elements) is parametrised as SU(2) matrices and each element
is pre-decomposed into a short pulse sequence drawn from the gate set above.

Supports
--------
  Standard RB  : generate_rb_sequence(n_cliffords, seed)
  Interleaved RB: generate_irb_sequence(n_cliffords, interleave_gate, seed)

IRB workflow
------------
  1. Run standard RB  → fit decay  → get reference fidelity r_ref
  2. Run IRB with gate G interleaved → fit decay → get r_irb
  3. Gate fidelity of G:
       F_gate = 1 - (d-1)/d * (r_irb/r_ref - 1)   d=2 for 1 qubit
       EPC     = (d-1)/d * (1 - r_irb/r_ref)
"""

import numpy as np
from scipy.linalg import expm

# ─────────────────────────────────────────────
# 1. Primitive gate matrices  (SU(2) or U(2))
# ─────────────────────────────────────────────

def _Rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s],
                     [-1j * s,  c]], dtype=complex)

def _Ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s],
                     [s,  c]], dtype=complex)

# Named primitives
GATE_MATRIX = {
    "I"   : np.eye(2, dtype=complex),
    "X"   : _Rx(np.pi),
    "Y"   : _Ry(np.pi),
    "X/2" : _Rx(np.pi / 2),
    "Y/2" : _Ry(np.pi / 2),
    "-X/2": _Rx(-np.pi / 2),
    "-Y/2": _Ry(-np.pi / 2),
}

def seq_to_matrix(pulse_list):
    """Multiply a list of gate name strings left-to-right (first applied first)."""
    U = np.eye(2, dtype=complex)
    for g in pulse_list:
        U = GATE_MATRIX[g] @ U
    return U

# ─────────────────────────────────────────────
# 2. All 24 single-qubit Cliffords
#    Each entry: {"matrix": U, "pulses": [...]}
#    Pulses are the physical sequence that realises the Clifford.
# ─────────────────────────────────────────────

# We generate all 24 by exhaustive composition and then assign pulse sequences.
# The canonical decomposition follows the Eppstein / Barends convention:
#   every Clifford can be written as (optional Y/2 or -Y/2) * (optional X/2 or X or -X/2) * (optional Y/2 or -Y/2)
# We just hard-code the 24 decompositions for clarity and reproducibility.

_RAW_CLIFFORDS = [
    # ── Pauli-like ──────────────────────────────────────────────
    ("C0",  ["I"]),
    ("C1",  ["X"]),
    ("C2",  ["Y"]),
    ("C3",  ["X", "Y"]),          # = Z  (up to phase)

    # ── π/2 around X / Y ────────────────────────────────────────
    ("C4",  ["X/2"]),
    ("C5",  ["-X/2"]),
    ("C6",  ["Y/2"]),
    ("C7",  ["-Y/2"]),

    # ── Hadamard-like ────────────────────────────────────────────
    ("C8",  ["X", "Y/2"]),
    ("C9",  ["X", "-Y/2"]),
    ("C10", ["Y", "X/2"]),
    ("C11", ["Y", "-X/2"]),

    # ── 2π/3 rotations (face diagonals of the octahedron) ───────
    ("C12", ["X/2",  "Y/2"]),
    ("C13", ["X/2",  "-Y/2"]),
    ("C14", ["-X/2", "Y/2"]),
    ("C15", ["-X/2", "-Y/2"]),

    # ── remaining face-diagonal rotations ───────────────────────
    ("C16", ["Y/2",  "X/2"]),
    ("C17", ["Y/2",  "-X/2"]),
    ("C18", ["-Y/2", "X/2"]),
    ("C19", ["-Y/2", "-X/2"]),

    # ── edge-midpoint rotations ──────────────────────────────────
    ("C20", ["X/2",  "Y/2",  "X/2"]),
    ("C21", ["X/2",  "Y/2", "-X/2"]),
    ("C22", ["X/2",  "-Y/2", "X/2"]),
    ("C23", ["-X/2", "Y/2",  "X/2"]),
]

# Build Clifford table: list of dicts with matrix + pulses
CLIFFORD_TABLE = []
for name, pulses in _RAW_CLIFFORDS:
    U = seq_to_matrix(pulses)
    CLIFFORD_TABLE.append({"name": name, "pulses": pulses, "matrix": U})

# Quick sanity: we should have exactly 24 distinct elements
def _matrix_distance(A, B):
    """
    Distance between two SU(2) matrices up to global phase.
    Returns minimum over the two possible phase ambiguities.
    """
    diff1 = np.linalg.norm(A - B, 'fro')
    diff2 = np.linalg.norm(A + B, 'fro')   # global phase flip
    return min(diff1, diff2)

def _verify_24():
    n = len(CLIFFORD_TABLE)
    assert n == 24, f"Expected 24 Cliffords, got {n}"
    for i in range(n):
        for j in range(i + 1, n):
            d = _matrix_distance(CLIFFORD_TABLE[i]["matrix"],
                                 CLIFFORD_TABLE[j]["matrix"])
            assert d > 1e-6, (
                f"Duplicate Cliffords: {CLIFFORD_TABLE[i]['name']} and "
                f"{CLIFFORD_TABLE[j]['name']}  (dist={d:.2e})"
            )

_verify_24()

# ─────────────────────────────────────────────
# 3. Closest-Clifford lookup
# ─────────────────────────────────────────────

def find_closest_clifford(U):
    """
    Given an arbitrary U(2) matrix, return the index of the Clifford in
    CLIFFORD_TABLE that is closest (up to global phase).
    """
    best_idx  = 0
    best_dist = np.inf
    for idx, cliff in enumerate(CLIFFORD_TABLE):
        d = _matrix_distance(U, cliff["matrix"])
        if d < best_dist:
            best_dist = d
            best_idx  = idx
    return best_idx, best_dist

# ─────────────────────────────────────────────
# 4. RB sequence generation
# ─────────────────────────────────────────────

def generate_rb_sequence(n_cliffords, seed=None):
    """
    Generate one RB sequence of length n_cliffords + 1 (the last gate is the
    recovery / inverse Clifford).

    Parameters
    ----------
    n_cliffords : int
        Number of random Clifford gates (not counting the recovery gate).
    seed : int or None
        NumPy random seed for reproducibility.

    Returns
    -------
    clifford_indices : list[int]
        Indices into CLIFFORD_TABLE for the n_cliffords random gates.
    recovery_index : int
        Index into CLIFFORD_TABLE for the recovery gate.
    pulse_sequence : list[str]
        Flat list of primitive gate names for the full experiment
        (random gates + recovery gate, in order).
    U_accumulated : np.ndarray (2×2 complex)
        Product of the n_cliffords random gates (before recovery).
    U_recovery : np.ndarray (2×2 complex)
        The recovery gate matrix.
    """
    rng = np.random.default_rng(seed)

    # Draw n_cliffords random Clifford indices
    clifford_indices = rng.integers(0, 24, size=n_cliffords).tolist()

    # Accumulate U = C_{m} · … · C_{1}  (first gate applied first → right side)
    U_acc = np.eye(2, dtype=complex)
    for idx in clifford_indices:
        U_acc = CLIFFORD_TABLE[idx]["matrix"] @ U_acc

    # Recovery gate = U_acc†  (in SU(2) sense)
    U_inv = U_acc.conj().T
    # Make sure it's SU(2): fix determinant phase
    det   = np.linalg.det(U_inv)
    U_inv = U_inv / np.sqrt(det)

    recovery_index, dist = find_closest_clifford(U_inv)
    assert dist < 1e-6, (
        f"Recovery Clifford not found — dist = {dist:.2e}. "
        "Check that CLIFFORD_TABLE spans the full group."
    )

    # Build flat pulse sequence
    pulse_sequence = []
    for idx in clifford_indices:
        pulse_sequence.extend(CLIFFORD_TABLE[idx]["pulses"])
    pulse_sequence.extend(CLIFFORD_TABLE[recovery_index]["pulses"])

    U_recovery = CLIFFORD_TABLE[recovery_index]["matrix"]

    return clifford_indices, recovery_index, pulse_sequence, U_acc, U_recovery


# ─────────────────────────────────────────────
# 5. Verification helper
# ─────────────────────────────────────────────

def verify_sequence(clifford_indices, recovery_index, verbose=True):
    """
    Check that the full sequence (random + recovery) returns to identity.
    Returns the residual Frobenius norm (should be < 1e-10).
    """
    U = np.eye(2, dtype=complex)
    for idx in clifford_indices:
        U = CLIFFORD_TABLE[idx]["matrix"] @ U
    U = CLIFFORD_TABLE[recovery_index]["matrix"] @ U

    # Up to global phase, should be identity
    phase = U[0, 0] / abs(U[0, 0]) if abs(U[0, 0]) > 1e-10 else 1.0
    residual = np.linalg.norm(U / phase - np.eye(2), 'fro')

    if verbose:
        print(f"  Full sequence matrix (normalised):\n{U / phase}")
        print(f"  Residual from Identity: {residual:.2e}")

    return residual


# ─────────────────────────────────────────────
# 6. Interleaved RB (IRB)
# ─────────────────────────────────────────────

# Built-in interleave targets: any named gate OR a custom SU(2) matrix
# The interleave gate is inserted after every random Clifford:
#   C_1 · G · C_2 · G · … · C_m · G · C_recovery
# where C_recovery inverts the entire accumulated product.

def _gate_to_matrix(gate):
    """
    Accept either:
      - str  : key in GATE_MATRIX  (e.g. "X", "Y/2")
      - list : pulse sequence      (e.g. ["X/2", "Y/2"])
      - ndarray : raw 2×2 SU(2) matrix
    Returns a 2×2 complex numpy array.
    """
    if isinstance(gate, np.ndarray):
        assert gate.shape == (2, 2), "Custom gate must be 2×2"
        return gate.astype(complex)
    if isinstance(gate, str):
        assert gate in GATE_MATRIX, f"Unknown gate name: {gate!r}"
        return GATE_MATRIX[gate]
    if isinstance(gate, (list, tuple)):
        return seq_to_matrix(gate)
    raise TypeError(f"gate must be str / list / ndarray, got {type(gate)}")


def _gate_to_pulses(gate):
    """Return the pulse list representation of gate (for flat sequence output)."""
    if isinstance(gate, np.ndarray):
        # Try to match to a known Clifford; if not possible return placeholder
        idx, dist = find_closest_clifford(gate)
        if dist < 1e-6:
            return list(CLIFFORD_TABLE[idx]["pulses"])
        raise ValueError(
            "Custom matrix gate does not match any Clifford — "
            "provide a pulse list instead."
        )
    if isinstance(gate, str):
        return [gate]
    if isinstance(gate, (list, tuple)):
        return list(gate)
    raise TypeError(f"gate must be str / list / ndarray, got {type(gate)}")


def generate_irb_sequence(n_cliffords, interleave_gate, seed=None):
    """
    Generate one Interleaved RB sequence.

    Structure:
        C_1 · G · C_2 · G · … · C_m · G · C_recovery

    where G is the interleaved gate and C_recovery satisfies:
        C_recovery · (G · C_m) · … · (G · C_1) = I

    Parameters
    ----------
    n_cliffords : int
        Number of random Clifford gates (not counting recovery).
    interleave_gate : str | list[str] | np.ndarray
        The gate G to interleave. Accepts:
          - str       : single primitive name, e.g. "X", "Y/2"
          - list[str] : pulse sequence,        e.g. ["X/2", "Y/2"]
          - ndarray   : raw 2×2 SU(2) matrix
    seed : int or None

    Returns
    -------
    clifford_indices : list[int]
        Random Clifford indices (length n_cliffords).
    recovery_index : int
        Index of the recovery Clifford.
    pulse_sequence : list[str]
        Flat pulse list for the full IRB sequence including recovery.
        Format: [C1_pulses, G_pulses, C2_pulses, G_pulses, …, recovery_pulses]
    U_accumulated : np.ndarray (2×2)
        Product (G·C_m)·…·(G·C_1) before recovery.
    U_recovery : np.ndarray (2×2)
        Recovery gate matrix.
    """
    rng = np.random.default_rng(seed)

    U_G       = _gate_to_matrix(interleave_gate)
    G_pulses  = _gate_to_pulses(interleave_gate)

    # Normalise G to SU(2)
    det = np.linalg.det(U_G)
    U_G = U_G / np.sqrt(det)

    clifford_indices = rng.integers(0, 24, size=n_cliffords).tolist()

    # Accumulate U = (G·C_m)·…·(G·C_1)
    U_acc = np.eye(2, dtype=complex)
    for idx in clifford_indices:
        U_C    = CLIFFORD_TABLE[idx]["matrix"]
        U_acc  = U_G @ U_C @ U_acc   # G applied after each C

    # Recovery: inverts U_acc
    U_inv = U_acc.conj().T
    det   = np.linalg.det(U_inv)
    U_inv = U_inv / np.sqrt(det)

    recovery_index, dist = find_closest_clifford(U_inv)
    assert dist < 1e-6, (
        f"IRB recovery Clifford not found — dist={dist:.2e}."
    )

    # Build flat pulse sequence: C_i pulses then G pulses, then recovery
    pulse_sequence = []
    for idx in clifford_indices:
        pulse_sequence.extend(CLIFFORD_TABLE[idx]["pulses"])
        pulse_sequence.extend(G_pulses)
    pulse_sequence.extend(CLIFFORD_TABLE[recovery_index]["pulses"])

    U_recovery = CLIFFORD_TABLE[recovery_index]["matrix"]

    return clifford_indices, recovery_index, pulse_sequence, U_acc, U_recovery


def verify_irb_sequence(clifford_indices, recovery_index, interleave_gate,
                         verbose=True):
    """
    Check IRB sequence returns to identity. Returns residual Frobenius norm.
    """
    U_G  = _gate_to_matrix(interleave_gate)
    det  = np.linalg.det(U_G)
    U_G  = U_G / np.sqrt(det)

    U = np.eye(2, dtype=complex)
    for idx in clifford_indices:
        U = U_G @ CLIFFORD_TABLE[idx]["matrix"] @ U
    U = CLIFFORD_TABLE[recovery_index]["matrix"] @ U

    phase    = U[0, 0] / abs(U[0, 0]) if abs(U[0, 0]) > 1e-10 else 1.0
    residual = np.linalg.norm(U / phase - np.eye(2), 'fro')

    if verbose:
        print(f"  Full IRB matrix (normalised):\n{U / phase}")
        print(f"  Residual from Identity: {residual:.2e}")

    return residual


def epc_from_rb_irb(r_ref, r_irb, d=2):
    """
    Compute Error Per Clifford (EPC) of the interleaved gate.

    Parameters
    ----------
    r_ref : float   decay rate from standard RB fit  (A·r_ref^m + B)
    r_irb : float   decay rate from IRB fit
    d     : int     Hilbert space dimension (2 for 1 qubit)

    Returns
    -------
    epc       : float   error per gate
    f_gate    : float   gate fidelity  = 1 - epc
    epc_bound : float   upper bound from Magesan et al.
    """
    epc       = (d - 1) / d * (1 - r_irb / r_ref)
    f_gate    = 1 - epc
    epc_bound = (d - 1) / d * (abs(r_ref - r_irb) / r_ref + (1 - r_ref))
    return epc, f_gate, epc_bound


# ─────────────────────────────────────────────
# 7. Demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 60)
    print("  Single-qubit RB + IRB — Clifford sequence generator")
    print("=" * 60)

    # ── Example 1: standard RB short sequence ───────────────────
    N = 8
    seed = 2025
    c_idx, r_idx, pulses, U_acc, U_rec = generate_rb_sequence(N, seed=seed)

    print(f"\n[1] Standard RB  n_cliffords={N}  seed={seed}")
    for i, idx in enumerate(c_idx):
        c = CLIFFORD_TABLE[idx]
        print(f"    Gate {i+1:2d}: {c['name']:4s}  {c['pulses']}")
    print(f"    Recovery: {CLIFFORD_TABLE[r_idx]['name']:4s}  "
          f"{CLIFFORD_TABLE[r_idx]['pulses']}")
    print(f"    Flat pulses ({len(pulses)}): {pulses}")
    res = verify_sequence(c_idx, r_idx)
    print(f"    → {'PASS ✓' if res < 1e-8 else 'FAIL ✗'}")

    # ── Example 2: IRB — interleave X gate ──────────────────────
    print(f"\n[2] IRB  interleave_gate='X'  n_cliffords={N}  seed={seed}")
    c_idx, r_idx, pulses, U_acc, _ = generate_irb_sequence(
        N, interleave_gate="X", seed=seed)

    for i, idx in enumerate(c_idx):
        c = CLIFFORD_TABLE[idx]
        print(f"    Gate {i+1:2d}: {c['name']:4s}  {c['pulses']}  →  X")
    print(f"    Recovery: {CLIFFORD_TABLE[r_idx]['name']:4s}  "
          f"{CLIFFORD_TABLE[r_idx]['pulses']}")
    print(f"    Flat pulses ({len(pulses)}): {pulses}")
    res = verify_irb_sequence(c_idx, r_idx, "X")
    print(f"    → {'PASS ✓' if res < 1e-8 else 'FAIL ✗'}")

    # ── Example 3: IRB — interleave X/2 (pulse list form) ───────
    print(f"\n[3] IRB  interleave_gate='X/2'  n_cliffords={N}  seed={seed}")
    c_idx, r_idx, pulses, _, _ = generate_irb_sequence(
        N, interleave_gate="X/2", seed=seed)
    print(f"    Flat pulses ({len(pulses)}): {pulses}")
    res = verify_irb_sequence(c_idx, r_idx, "X/2")
    print(f"    → {'PASS ✓' if res < 1e-8 else 'FAIL ✗'}")

    # ── Example 4: IRB — interleave custom pulse list ────────────
    # e.g. your real X gate might be ["X/2", "X/2"] due to hardware decomp
    print(f"\n[4] IRB  interleave_gate=['X/2','X/2']  (custom pulse list)")
    c_idx, r_idx, pulses, _, _ = generate_irb_sequence(
        N, interleave_gate=["X/2", "X/2"], seed=seed)
    res = verify_irb_sequence(c_idx, r_idx, ["X/2", "X/2"])
    print(f"    Flat pulses ({len(pulses)}): {pulses}")
    print(f"    → {'PASS ✓' if res < 1e-8 else 'FAIL ✗'}")

    # ── Example 5: stress test both RB and IRB ───────────────────
    print("\n[5] Stress test — 300 RB + 300 IRB, lengths 1–40")
    gates_to_test = ["X", "Y", "X/2", "Y/2", "-X/2", "-Y/2"]
    rng = np.random.default_rng(42)
    fail_rb = fail_irb = 0

    for _ in range(300):
        n  = int(rng.integers(1, 41))
        sd = int(rng.integers(0, 2**31))

        c_idx, r_idx, _, _, _ = generate_rb_sequence(n, seed=sd)
        if verify_sequence(c_idx, r_idx, verbose=False) > 1e-8:
            fail_rb += 1

        ig = gates_to_test[int(rng.integers(0, len(gates_to_test)))]
        c_idx, r_idx, _, _, _ = generate_irb_sequence(n, ig, seed=sd)
        if verify_irb_sequence(c_idx, r_idx, ig, verbose=False) > 1e-8:
            fail_irb += 1

    print(f"    RB  failures: {fail_rb}/300  "
          f"{'ALL PASS ✓' if fail_rb==0 else 'FAIL ✗'}")
    print(f"    IRB failures: {fail_irb}/300  "
          f"{'ALL PASS ✓' if fail_irb==0 else 'FAIL ✗'}")

    # ── Example 6: EPC formula demo (simulated decay rates) ─────
    print("\n[6] EPC formula demo (simulated r_ref / r_irb)")
    r_ref  = 0.98    # typical reference RB decay per Clifford
    r_irb  = 0.975   # IRB decay with gate G interleaved
    epc, f_gate, bound = epc_from_rb_irb(r_ref, r_irb)
    print(f"    r_ref   = {r_ref}")
    print(f"    r_irb   = {r_irb}")
    print(f"    EPC     = {epc:.5f}   (error per gate)")
    print(f"    F_gate  = {f_gate:.5f}  (gate fidelity)")
    print(f"    bound   = {bound:.5f}  (Magesan upper bound)")

    # ── Example 7: typical IRB experiment layout ─────────────────
    print("\n[7] Typical IRB experiment structure")
    print("    interleave_gate = 'X'")
    lengths = [1, 2, 4, 8, 16, 32]
    n_seeds = 2
    for L in lengths:
        for s in range(n_seeds):
            # --- reference RB ---
            c_idx, r_idx, pulses_ref, _, _ = generate_rb_sequence(L, seed=s)
            res_ref = verify_sequence(c_idx, r_idx, verbose=False)
            # --- IRB ---
            c_idx, r_idx, pulses_irb, _, _ = generate_irb_sequence(
                L, "X", seed=s)
            res_irb = verify_irb_sequence(c_idx, r_idx, "X", verbose=False)
            ok = "✓" if (res_ref < 1e-8 and res_irb < 1e-8) else "✗"
            print(f"    L={L:3d} seed={s}  "
                  f"ref_pulses={len(pulses_ref):3d}  "
                  f"irb_pulses={len(pulses_irb):3d}  {ok}")
