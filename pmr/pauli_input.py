import numpy as np


def validate_pauli_string(pauli):
    """Check that a Pauli string contains only I, X, Y, and Z."""
    pauli = pauli.upper().strip()

    if not pauli:
        raise ValueError("Pauli string cannot be empty.")

    if any(operator not in "IXYZ" for operator in pauli):
        raise ValueError(
            "Pauli strings may contain only I, X, Y, and Z."
        )

    return pauli


def pauli_string_to_matrix(pauli):
    """Convert a Pauli string into its matrix representation."""
    pauli = validate_pauli_string(pauli)

    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    matrices = {
        "I": I,
        "X": X,
        "Y": Y,
        "Z": Z,
    }

    result = np.array([[1]], dtype=complex)

    for operator in pauli:
        result = np.kron(result, matrices[operator])

    return result


def build_hamiltonian_from_paulis(pauli_terms):
    """
    Build H from input such as:

        [(2, "XI"), (3, "XZ"), (4, "YI")]
    """
    if not pauli_terms:
        raise ValueError("At least one Pauli term is required.")

    first_pauli = validate_pauli_string(pauli_terms[0][1])

    num_qubits = len(first_pauli)
    dimension = 2 ** num_qubits

    H = np.zeros((dimension, dimension), dtype=complex)

    for coefficient, pauli in pauli_terms:
        pauli = validate_pauli_string(pauli)

        if len(pauli) != num_qubits:
            raise ValueError(
                "All Pauli strings must have the same length."
            )

        H += coefficient * pauli_string_to_matrix(pauli)

    return H


def read_pauli_terms_from_terminal():
    """Read Pauli terms from the terminal."""

    while True:
        try:
            number_of_terms = int(
                input("Number of Pauli terms: ")
            )

            if number_of_terms <= 0:
                raise ValueError

            break

        except ValueError:
            print("Please enter a positive integer.")

    pauli_terms = []
    num_qubits = None

    for term_index in range(number_of_terms):

        print(f"\nTerm {term_index + 1}")

        while True:
            try:
                coefficient = complex(
                    input("Coefficient: ")
                    .lower()
                    .replace("i", "j")
                )
                break

            except ValueError:
                print("Please enter a valid coefficient.")

        while True:
            try:
                pauli = validate_pauli_string(
                    input("Pauli string: ")
                )

                if num_qubits is None:
                    num_qubits = len(pauli)

                elif len(pauli) != num_qubits:
                    raise ValueError(
                        "All Pauli strings must have the same length."
                    )

                break

            except ValueError as error:
                print(error)

        pauli_terms.append((coefficient, pauli))

    return pauli_terms
