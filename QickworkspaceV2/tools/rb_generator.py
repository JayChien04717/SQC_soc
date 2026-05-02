import numpy as np
import random


def rx(theta):
    """Construct the single-qubit X-rotation matrix R_x(theta)."""
    return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                     [-1j*np.sin(theta/2), np.cos(theta/2)]])


def ry(theta):
    """Construct the single-qubit Y-rotation matrix R_y(theta)."""
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

# Build Clifford matrices — list left-to-right = gate applied first to |ψ⟩
clifford_matrices = []
for decomp in clifford_decompositions:
    mat = I.copy()
    for g in decomp:
        mat = gate_map[g] @ mat
    clifford_matrices.append(mat)

assert len(clifford_matrices) == 24


def _matrix_distance(A, B):
    """Distance up to global phase: checks both +phase and -phase."""
    return min(np.linalg.norm(A - B, 'fro'),
               np.linalg.norm(A + B, 'fro'))


for i in range(24):
    for j in range(i+1, 24):
        d = _matrix_distance(clifford_matrices[i], clifford_matrices[j])
        assert d > 1e-6, f"Duplicate Cliffords at index {i} and {j}"

# Pre-build inverse lookup table: inverse_table[i] = j such that C_j @ C_i ≈ I
inverse_table = {}
I2 = np.eye(2, dtype=complex)
for i, ci in enumerate(clifford_matrices):
    for j, cj in enumerate(clifford_matrices):
        if _matrix_distance(cj @ ci, I2) < 1e-6:
            inverse_table[i] = j
            break
    assert i in inverse_table, f"Could not find inverse for Clifford {i}"

assert len(inverse_table) == 24


def find_clifford_index(U):
    """Return index of the Clifford closest to U (up to global phase)."""
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


INTERLEAVE_GATES = {
    "X":   (gate_map["X"],   ["X"]),
    "Y":   (gate_map["Y"],   ["Y"]),
    "X/2": (gate_map["X/2"], ["X/2"]),
    "Y/2": (gate_map["Y/2"], ["Y/2"]),
}


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
        Gate to interleave after each random Clifford. Must be one of
        "X", "Y", "X/2", "Y/2". None produces standard RB.
    seed : int or None, optional
        Random seed for reproducibility.
    debug : bool, optional
        If True, print each sample's RB list and recovery gate.

    Returns
    -------
    results : list of list of str
        Length n_sample. Each element is a flat list of gate strings.
    """
    rng = random.Random(seed)

    if interleave is not None:
        if interleave not in INTERLEAVE_GATES:
            raise ValueError(
                f"Unsupported interleave gate {interleave!r}. "
                f"Choose from: {list(INTERLEAVE_GATES.keys())}"
            )
        interleave_mat    = INTERLEAVE_GATES[interleave][0]
        interleave_pulses = INTERLEAVE_GATES[interleave][1]
    else:
        interleave_mat    = None
        interleave_pulses = None

    results = []

    for sample_idx in range(n_sample):
        sequence_indices = []
        current_mat = I.copy()
        interleave_pulse_log = []

        for _ in range(n_clifford):
            idx = rng.randint(0, 23)
            sequence_indices.append(idx)
            current_mat = clifford_matrices[idx] @ current_mat

            if interleave_mat is not None:
                current_mat = interleave_mat @ current_mat
                interleave_pulse_log.append(interleave_pulses)

        acc_idx = find_clifford_index(current_mat)
        inv_idx = inverse_table[acc_idx]

        rb_pulse_list = []
        ilv_iter = iter(interleave_pulse_log) if interleave_pulses is not None else None

        for k, idx in enumerate(sequence_indices):
            rb_pulse_list.extend(clifford_decompositions[idx])
            if ilv_iter is not None:
                rb_pulse_list.extend(next(ilv_iter))

        recovery_gates = clifford_decompositions[inv_idx]

        if debug:
            print(f"Sample {sample_idx + 1}:")
            print(f"  RB list  : {rb_pulse_list}")
            print(f"  Recovery : {recovery_gates}")
            print()

        full_sequence = rb_pulse_list + recovery_gates
        results.append(full_sequence)

    return results


def verify_sequence(full_sequence):
    """
    Verify that the full 1D flat sequence (rb gates + recovery) returns to Identity.

    Returns True if residual matrix distance from Identity is less than 1e-6.
    """
    mat = I.copy()
    for g in full_sequence:
        mat = gate_map[g] @ mat
    return _matrix_distance(mat, I2) < 1e-6
