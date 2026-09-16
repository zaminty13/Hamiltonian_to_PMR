import numpy as np

I = np.array([
    [1, 0],
    [0, 1]
], dtype=complex)

X = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)


def permutation_matrix(pauli_string):
    P = np.array([[1]], dtype=complex)

    for pauli in pauli_string:
        if pauli in ("I", "Z"):
            local_P = I
        elif pauli in ("X", "Y"):
            local_P = X
        else:
            raise ValueError(f"Invalid Pauli operator: {pauli}")

        P = np.kron(P, local_P)

    return P