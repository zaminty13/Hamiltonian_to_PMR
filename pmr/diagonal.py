import numpy as np

I = np.array([
    [1, 0],
    [0, 1]
], dtype=complex)

Z = np.array([
    [1, 0],
    [0, -1]
], dtype=complex)


def diagonal_matrix(pauli_string, coefficient):
    D = np.array([[1]], dtype=complex)
    phase = 1

    for pauli in pauli_string:
        if pauli in ("I", "X"):
            local_D = I
        elif pauli == "Z":
            local_D = Z
        elif pauli == "Y":
            local_D = Z
            phase *= -1j
        else:
            raise ValueError(f"Invalid Pauli operator: {pauli}")

        D = np.kron(D, local_D)

    return coefficient * phase * D