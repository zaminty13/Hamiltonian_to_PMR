def get_diagonal(pauli_string):
    D = ""
    phase = 1

    for pauli in pauli_string:
        if pauli == "I":
            D += "I"
        elif pauli == "X":
            D += "I"
        elif pauli == "Z":
            D += "Z"
        elif pauli == "Y":
            D += "Z"
            phase *= -1j
        else:
            raise ValueError(f"Invalid Pauli operator: {pauli}")

    return D, phase