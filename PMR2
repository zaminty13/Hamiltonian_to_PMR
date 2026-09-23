import numpy as np


PAULI_MATRICES = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_terms_to_matrix(pauli_terms):
    """Build H = sum(coefficient * Pauli_string) from Pauli terms.

    The leftmost character is the most-significant qubit. For example,
    ``"XI"`` means ``X kron I``.
    """
    if not pauli_terms:
        raise ValueError("At least one Pauli term is required.")

    number_of_qubits = len(pauli_terms[0][1])
    if number_of_qubits == 0:
        raise ValueError("Pauli strings cannot be empty.")

    dimension = 2**number_of_qubits
    H = np.zeros((dimension, dimension), dtype=complex)

    for coefficient, pauli_string in pauli_terms:
        pauli_string = pauli_string.upper()

        if len(pauli_string) != number_of_qubits:
            raise ValueError("All Pauli strings must have the same length.")
        if any(symbol not in PAULI_MATRICES for symbol in pauli_string):
            raise ValueError(
                f"Invalid Pauli string {pauli_string!r}; use only I, X, Y, and Z."
            )

        term_matrix = np.array([[1]], dtype=complex)
        for symbol in pauli_string:
            term_matrix = np.kron(term_matrix, PAULI_MATRICES[symbol])

        H += complex(coefficient) * term_matrix

    return H


def validate_pauli_terms(pauli_terms):
    """Validate the input and return the number of qubits."""
    if not pauli_terms:
        raise ValueError("At least one Pauli term is required.")

    number_of_qubits = len(pauli_terms[0][1])
    if number_of_qubits == 0:
        raise ValueError("Pauli strings cannot be empty.")

    for _, pauli_string in pauli_terms:
        pauli_string = pauli_string.upper()
        if len(pauli_string) != number_of_qubits:
            raise ValueError("All Pauli strings must have the same length.")
        if any(symbol not in PAULI_MATRICES for symbol in pauli_string):
            raise ValueError(
                f"Invalid Pauli string {pauli_string!r}; use only I, X, Y, and Z."
            )

    return number_of_qubits


def pauli_terms_to_pmr(pauli_terms, tolerance=1e-12, materialize=False):
    """Convert Pauli terms directly to PMR form.

    This version supports any number of qubits and does not first construct the
    full dense Hamiltonian. In compact mode (the default), only each diagonal
    and XOR mask are stored. Set ``materialize=True`` to also construct D and P.
    """
    number_of_qubits = validate_pauli_terms(pauli_terms)
    dimension = 2**number_of_qubits
    states = np.arange(dimension, dtype=np.int64)
    diagonals_by_mask = {}

    for coefficient, pauli_string in pauli_terms:
        pauli_string = pauli_string.upper()
        flip_mask = 0

        # The leftmost Pauli symbol acts on the most-significant bit.
        for position, symbol in enumerate(pauli_string):
            if symbol in "XY":
                flip_mask |= 1 << (number_of_qubits - position - 1)

        # D[row] is H[row, row XOR flip_mask]. Compute the phase of the
        # Pauli string on that column state without constructing its matrix.
        columns = states ^ flip_mask
        diagonal = np.full(dimension, complex(coefficient), dtype=complex)

        for position, symbol in enumerate(pauli_string):
            bit_position = number_of_qubits - position - 1
            bits = (columns >> bit_position) & 1

            if symbol == "Z":
                diagonal *= 1 - 2 * bits
            elif symbol == "Y":
                diagonal *= 1j * (1 - 2 * bits)

        if flip_mask in diagonals_by_mask:
            diagonals_by_mask[flip_mask] += diagonal
        else:
            diagonals_by_mask[flip_mask] = diagonal

    pmr_terms = []
    for flip_mask in sorted(diagonals_by_mask):
        diagonal = diagonals_by_mask[flip_mask]
        if np.all(np.abs(diagonal) <= tolerance):
            continue

        D = None
        P = None
        if materialize:
            D = np.diag(diagonal)
            P = np.zeros((dimension, dimension), dtype=complex)
            P[states ^ flip_mask, states] = 1.0

        pmr_terms.append(
            {
                "index": len(pmr_terms),
                "flip_mask": flip_mask,
                "diagonal": diagonal,
                "D": D,
                "P": P,
            }
        )

    return pmr_terms, dimension


def matrix_to_pmr(H, tolerance=1e-12):
    """Decompose H as H = sum_k D_k @ P_k.

    D_k is diagonal and P_k maps |s> to |s XOR k>.
    """
    H = np.asarray(H, dtype=complex)

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be a square matrix.")

    dimension = H.shape[0]
    if dimension == 0 or dimension & (dimension - 1):
        raise ValueError("The dimension of H must be a power of two.")

    states = np.arange(dimension)
    pmr_terms = []

    for flip_mask in range(dimension):
        diagonal = H[states, states ^ flip_mask].copy()

        if np.all(np.abs(diagonal) <= tolerance):
            continue

        P = np.zeros((dimension, dimension), dtype=complex)
        P[states ^ flip_mask, states] = 1.0

        pmr_terms.append(
            {
                "index": len(pmr_terms),
                "flip_mask": flip_mask,
                "diagonal": diagonal,
                "D": np.diag(diagonal),
                "P": P,
            }
        )

    return pmr_terms


def reconstruct_pmr(pmr_terms, dimension):
    """Reconstruct H from its PMR terms."""
    H_reconstructed = np.zeros((dimension, dimension), dtype=complex)
    states = np.arange(dimension)
    for term in pmr_terms:
        # This works in compact mode and avoids multiplying two dense matrices.
        H_reconstructed[states, states ^ term["flip_mask"]] += term["diagonal"]
    return H_reconstructed


def print_pmr(pmr_terms):
    """Print the decomposition and its matrices."""
    expression = " + ".join(
        f"D{term['index']} @ P{term['index']}" for term in pmr_terms
    )
    print("H =", expression if expression else "0")

    for term in pmr_terms:
        index = term["index"]
        mask = term["flip_mask"]
        print(f"\nTerm {index}, XOR flip mask = {mask}:")
        print(f"diagonal(D{index}) =")
        print(term["diagonal"])
        print(f"D{index} =")
        print(np.diag(term["diagonal"]))
        if term["D"] is not None:
            print(f"D{index} =")
            print(term["D"])
            print(f"P{index} =")
            print(term["P"])


if __name__ == "__main__":
    # Use strings of any common length. Their length sets the dimension:
    # 2 symbols -> 2**2 = 4, 3 symbols -> 2**3 = 8, etc.
    pauli_terms = [
    (2, "XIII"),
    (3, "XZII"),
    (4, "YIIZ"),
    (5, "IZZZ"),
    (-1.5, "XXYY"),
    (2j, "ZYXI"),
]

    # Compact PMR conversion avoids allocating a dense matrix for every D and P.
    pmr_terms, dimension = pauli_terms_to_pmr(pauli_terms)

    print("Input Pauli terms:")
    for coefficient, pauli_string in pauli_terms:
        print(f"  {coefficient} * {pauli_string}")

    print(f"\nNumber of qubits: {len(pauli_terms[0][1])}")
    print(f"Matrix dimension: {dimension} x {dimension}")
    print("\nCompact PMR decomposition:")
    print_pmr(pmr_terms)

    # Dense matrices are created here only for demonstration and verification.
    H = pauli_terms_to_matrix(pauli_terms)
    H_reconstructed = reconstruct_pmr(pmr_terms, dimension)
    print("\nMatrix H =")
    print(H)
    print("\nReconstructed H =")
    print(H_reconstructed)
    print("\nDecomposition is correct:", np.allclose(H, H_reconstructed))
