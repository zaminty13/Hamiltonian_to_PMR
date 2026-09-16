from pmr.diagonal import diagonal_matrix
from pmr.permutation import permutation_matrix


def convert_to_pmr(hamiltonian):
    pmr_terms = []

    for coefficient, pauli_string in hamiltonian:
        D = diagonal_matrix(pauli_string, coefficient)
        P = permutation_matrix(pauli_string)

        pmr_terms.append((D, P))

    return pmr_terms