import numpy as np
import random


def rx(theta):
    """
    Construct the single-qubit X-rotation matrix R_x(theta).

    Parameters
    ----------
    theta : float
        Rotation angle in radians.

    Returns
    -------
    ndarray of shape (2, 2)
        Unitary rotation matrix about the X axis.
    """
    return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                     [-1j*np.sin(theta/2), np.cos(theta/2)]])


def ry(theta):
    """
    Construct the single-qubit Y-rotation matrix R_y(theta).

    Parameters
    ----------
    theta : float
        Rotation angle in radians.

    Returns
    -------
    ndarray of shape (2, 2)
        Unitary rotation matrix about the Y axis.
    """
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2), np.cos(theta/2)]])


I = np.eye(2, dtype=complex)
X_half  = rx( np.pi/2)
mX_half = rx(-np.pi/2)
Y_half  = ry( np.pi/2)
mY_half = ry(-np.pi/2)
X = X_half @ X_half
Y = Y_half @ Y_half

gate_map = {
    "I":    I,
    "X/2":  X_half,
    "-X/2": mX_half,
    "Y/2":  Y_half,
    "-Y/2": mY_half,
    "X":    X,
    "Y":    Y,
}

clifford_decompositions = [
    ["I"],
    ["X/2", "X/2"],
    ["Y/2", "Y/2"],
    ["Y/2", "Y/2", "X/2", "X/2"],
    ["X/2"],
    ["-X/2"],
    ["Y/2"],
    ["-Y/2"],
    ["-X/2", "Y/2", "X/2"],
    ["-X/2", "-Y/2", "X/2"],
    ["X/2", "X/2", "Y/2"],
    ["X/2", "X/2", "-Y/2"],
    ["Y/2", "Y/2", "X/2"],
    ["Y/2", "Y/2", "-X/2"],
    ["X/2", "Y/2", "X/2"],
    ["-X/2", "Y/2", "-X/2"],
    ["Y/2", "X/2"],
    ["Y/2", "-X/2"],
    ["-Y/2", "X/2"],
    ["-Y/2", "-X/2"],
    ["X/2", "-Y/2"],
    ["X/2", "Y/2"],
    ["-X/2", "-Y/2"],
    ["-X/2", "Y/2"],
]

# ── Build Clifford matrices ──────────────────────────────────────
# Convention: list left-to-right = gate applied first to |ψ⟩
# U_total = U_last @ ... @ U_first
clifford_matrices = []
for decomp in clifford_decompositions:
    mat = I.copy()
    for g in decomp:
        mat = gate_map[g] @ mat
    clifford_matrices.append(mat)

assert len(clifford_matrices) == 24

# ── Verify 24 distinct elements ──────────────────────────────────
def _matrix_distance(A, B):
    """Distance up to global phase: checks both +phase and -phase."""
    return min(np.linalg.norm(A - B, 'fro'),
               np.linalg.norm(A + B, 'fro'))

for i in range(24):
    for j in range(i+1, 24):
        d = _matrix_distance(clifford_matrices[i], clifford_matrices[j])
        assert d > 1e-6, f"Duplicate Cliffords at index {i} and {j}"

# ── Pre-build inverse lookup table ──────────────────────────────
# inverse_table[i] = j  such that  C_j @ C_i ≈ I  (up to global phase)
# This is computed ONCE using matrix distance (no threshold fragility).
inverse_table = {}
I2 = np.eye(2, dtype=complex)
for i, ci in enumerate(clifford_matrices):
    for j, cj in enumerate(clifford_matrices):
        if _matrix_distance(cj @ ci, I2) < 1e-6:
            inverse_table[i] = j
            break
    assert i in inverse_table, f"Could not find inverse for Clifford {i}"

assert len(inverse_table) == 24

# ── Find closest Clifford (used for accumulated matrix) ──────────
def find_clifford_index(U):
    """
    Return index of the Clifford closest to U (up to global phase).

    Parameters
    ----------
    U : ndarray of shape (2, 2)
        Unitary matrix to look up in the Clifford group.

    Returns
    -------
    best_idx : int
        Index into ``clifford_matrices`` of the closest element.

    Raises
    ------
    AssertionError
        If no match is found within tolerance 1e-6.
    """
    best_idx, best_dist = 0, np.inf
    for idx, m in enumerate(clifford_matrices):
        d = _matrix_distance(U, m)
        if d < best_dist:
            best_dist = d
            best_idx = idx
    assert best_dist < 1e-6, (
        f"Accumulated matrix not in Clifford group (dist={best_dist:.2e}).\n"
        "Check clifford_decompositions for errors."
    )
    return best_idx

# ── INTERLEAVE gate definitions ──────────────────────────────────
# Each entry: gate_name -> (matrix, [pulse_list])
# The pulse_list is a SINGLE atomic pulse — NOT the clifford_decompositions
# entry (which may expand X into ["X/2","X/2"]).
# This ensures IRB characterizes exactly one physical gate operation.
INTERLEAVE_GATES = {
    "X":   (gate_map["X"],   ["X"]),
    "Y":   (gate_map["Y"],   ["Y"]),
    "X/2": (gate_map["X/2"], ["X/2"]),
    "Y/2": (gate_map["Y/2"], ["Y/2"]),
}

# ── Main RB sequence generator ───────────────────────────────────
def single_qb_rb(n_clifford, n_sample, interleave=None, seed=None, debug=False):
    """
    Generate n_sample RB (or IRB) sequences of length n_clifford.

    Parameters
    ----------
    n_clifford : int
        Number of random Cliffords per sequence.
    n_sample : int
        Number of independent sequences to generate.
    interleave : str or None, optional
        Gate to interleave after each random Clifford.  Must be one of
        ``"X"``, ``"Y"``, ``"X/2"``, ``"Y/2"``.  None produces standard RB.
    seed : int or None, optional
        Random seed for reproducibility.
    debug : bool, optional
        If True, print each sample's RB list and recovery gate.

    Returns
    -------
    results : list of list of str
        Length ``n_sample``.  Each element is a flat list of gate strings::

            [random_gates..., (interleaved_gates...,) recovery_gates...]

        The recovery gate is already appended at the end.

    Raises
    ------
    ValueError
        If ``interleave`` is not in ``INTERLEAVE_GATES``.
    """
    rng = random.Random(seed)

    if interleave is not None:
        if interleave not in INTERLEAVE_GATES:
            raise ValueError(
                f"Unsupported interleave gate {interleave!r}. "
                f"Choose from: {list(INTERLEAVE_GATES.keys())}"
            )
        interleave_mat    = INTERLEAVE_GATES[interleave][0]   # matrix for accumulation
        interleave_pulses = INTERLEAVE_GATES[interleave][1]   # single atomic pulse list
    else:
        interleave_mat    = None
        interleave_pulses = None

    results = []

    for sample_idx in range(n_sample):
        sequence_indices = []
        current_mat = I.copy()

        interleave_pulse_log = []   # records the interleaved pulses in order

        for _ in range(n_clifford):
            idx = rng.randint(0, 23)
            sequence_indices.append(idx)
            current_mat = clifford_matrices[idx] @ current_mat

            if interleave_mat is not None:
                # Accumulate using the exact gate matrix (not a decomposition)
                current_mat = interleave_mat @ current_mat
                interleave_pulse_log.append(interleave_pulses)

        # ── Recovery gate via pre-built inverse table ────────────
        acc_idx = find_clifford_index(current_mat)
        inv_idx = inverse_table[acc_idx]

        # ── Build flat pulse list ────────────────────────────────
        # Standard RB: just expand each clifford decomposition
        # IRB: interleave the atomic gate pulse after each random Clifford
        rb_pulse_list = []
        ilv_iter = iter(interleave_pulse_log) if interleave_pulses is not None else None

        for k, idx in enumerate(sequence_indices):
            rb_pulse_list.extend(clifford_decompositions[idx])
            # After every random Clifford (not after interleaved ones),
            # insert the atomic interleave pulse.
            # sequence_indices for IRB is: [rand, rand, rand, ...] — no interleave idx
            if ilv_iter is not None:
                rb_pulse_list.extend(next(ilv_iter))

        recovery_gates = clifford_decompositions[inv_idx]

        # ── Debug print ──────────────────────────────────────────
        if debug:
            print(f"Sample {sample_idx + 1}:")
            print(f"  RB list  : {rb_pulse_list}")
            print(f"  Recovery : {recovery_gates}")
            print()

        # ── 1D flat list: rb gates + recovery gate ───────────────
        full_sequence = rb_pulse_list + recovery_gates
        results.append(full_sequence)

    return results

# ── Verification (updated for 1D flat list) ──────────────────────
def verify_sequence(full_sequence):
    """
    Verify that the full 1D flat sequence (rb gates + recovery) returns to Identity.

    Parameters
    ----------
    full_sequence : list of str
        Flat gate list including the recovery gate, as returned by
        ``single_qb_rb``.

    Returns
    -------
    bool
        True if the residual matrix distance from Identity is less than 1e-6.
    """
    mat = I.copy()
    for g in full_sequence:
        mat = gate_map[g] @ mat
    return _matrix_distance(mat, I2) < 1e-6

# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Standard RB (n=10, samples=3, debug=True) ===")
    rb_samples = single_qb_rb(n_clifford=10, n_sample=3, seed=42, debug=True)
    for i, seq in enumerate(rb_samples):
        ok = verify_sequence(seq)
        print(f"  Sample {i+1}: {len(seq):3d} total gates | Identity: {ok}")

    print()
    print("=== IRB (n=5, samples=3, gate='X', debug=True) ===")
    irb_samples = single_qb_rb(n_clifford=5, n_sample=3, interleave="X", seed=42, debug=True)
    for i, seq in enumerate(irb_samples):
        ok = verify_sequence(seq)
        print(f"  Sample {i+1}: {len(seq):3d} total gates | Identity: {ok}")

    print()
    print("=== All interleave gates ===")
    for g in ["X", "Y", "X/2", "Y/2"]:
        seq = single_qb_rb(8, 1, interleave=g, seed=7)[0]
        ok = verify_sequence(seq)
        print(f"  gate={g!r:5s} Identity: {ok} {'✓' if ok else '✗ FAIL'}")
