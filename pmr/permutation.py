def get_permutation(pauli_string):
    P = ""

    for pauli in pauli_string:
        if pauli == "I":
            P += "I"
        elif pauli == "X":
            P += "X"
        elif pauli == "Z":
            P += "I"
        elif pauli == "Y":
            P += "X"
        else:
            raise ValueError(f"Invalid Pauli operator: {pauli}")

    return P