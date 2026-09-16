import numpy as np

I = np.array([
    [1, 0],
    [0, 1]
], dtype=complex)

X = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

Y = np.array([
    [0, -1j],
    [1j, 0]
], dtype=complex)

Z = np.array([
    [1, 0],
    [0, -1]
], dtype=complex)

PAULI_MATRICES = {
    "I": I,
    "X": X,
    "Y": Y,
    "Z": Z
}


def pauli_string_matrix(pauli_string):
    matrix = np.array([[1]], dtype=complex)

    for pauli in pauli_string:
        if pauli not in PAULI_MATRICES:
            raise ValueError(f"Invalid Pauli operator: {pauli}")

        matrix = np.kron(matrix, PAULI_MATRICES[pauli])

    return matrix


def original_hamiltonian_matrix(hamiltonian):
    n_qubits = len(hamiltonian[0][1])
    dimension = 2 ** n_qubits

    H = np.zeros((dimension, dimension), dtype=complex)

    for coefficient, pauli_string in hamiltonian:
        H += coefficient * pauli_string_matrix(pauli_string)

    return H


def pmr_hamiltonian_matrix(pmr_terms):
    dimension = pmr_terms[0][0].shape[0]

    H = np.zeros((dimension, dimension), dtype=complex)

    for D, P in pmr_terms:
        H += D @ P

    return H


def verify(hamiltonian, pmr_terms):
    H_original = original_hamiltonian_matrix(hamiltonian)
    H_pmr = pmr_hamiltonian_matrix(pmr_terms)

    return np.allclose(H_original, H_pmr)