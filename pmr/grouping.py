from .converter import convert_term


def group_terms(hamiltonian):
    groups = {}

    for coefficient, pauli_string in hamiltonian:
        diagonal_coefficient, D, P = convert_term(
            coefficient,
            pauli_string
        )

        if P not in groups:
            groups[P] = {}

        if D not in groups[P]:
            groups[P][D] = 0

        groups[P][D] += diagonal_coefficient

    return groups


def hamiltonian_to_pmr(hamiltonian):
    return group_terms(hamiltonian)