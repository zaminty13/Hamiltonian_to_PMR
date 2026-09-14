from .permutation import get_permutation
from .diagonal import get_diagonal


def convert_term(coefficient, pauli_string):
    P = get_permutation(pauli_string)
    D, phase = get_diagonal(pauli_string)

    diagonal_coefficient = coefficient * phase

    return diagonal_coefficient, D, P