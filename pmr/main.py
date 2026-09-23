import numpy as np

from pauli_input import (
    read_pauli_terms_from_terminal,
    pauli_terms_to_pmr,
)

from matrix_pmr import (
    reconstruct_pmr,
    print_pmr,
)


# Dense 2^n x 2^n matrices are only printed up to this many qubits.
MAX_DISPLAY_QUBITS = 4

_SINGLE_QUBIT_OPERATORS = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def apply_pauli_terms(pauli_terms, psi):
    """
    Compute H|psi> straight from the Pauli terms by applying the 2x2
    matrices one qubit at a time.

    This is independent of the PMR code and never builds a dense matrix,
    so it makes a meaningful, cheap correctness check.
    """
    num_qubits = len(pauli_terms[0][1])
    result = np.zeros_like(psi)

    for coefficient, pauli in pauli_terms:
        # Axis 0 is the left-most character (most significant bit).
        tensor = psi.reshape((2,) * num_qubits)

        for axis, operator in enumerate(pauli):
            if operator != "I":
                tensor = np.moveaxis(
                    np.tensordot(
                        _SINGLE_QUBIT_OPERATORS[operator],
                        tensor,
                        axes=([1], [axis]),
                    ),
                    0,
                    axis,
                )

        result += coefficient * tensor.reshape(-1)

    return result


def apply_pmr(pmr_terms, psi):
    """Compute H|psi> from PMR terms: (H psi)[s] = sum_k D_k[s] * psi[s ^ k]."""
    states = np.arange(psi.size)
    result = np.zeros_like(psi)

    for term in pmr_terms:
        result += term["diagonal"] * psi[states ^ term["flip_mask"]]

    return result


def main():
    pauli_terms = read_pauli_terms_from_terminal()

    num_qubits = len(pauli_terms[0][1])
    dimension = 1 << num_qubits

    print("\nPauli Hamiltonian:")
    for coefficient, pauli in pauli_terms:
        print(f"  ({coefficient}) * {pauli}")

    pmr_terms = pauli_terms_to_pmr(pauli_terms)

    print("\nPMR decomposition:")

    if num_qubits <= MAX_DISPLAY_QUBITS:
        print_pmr(pmr_terms)

        print("\nH =")
        print(reconstruct_pmr(pmr_terms, dimension))

    else:
        print(
            f"  {len(pmr_terms)} terms "
            f"(dense matrices not printed for {num_qubits} qubits)"
        )

        for term in pmr_terms:
            print(
                f"  D{term['index']} @ P{term['index']}, "
                f"flip mask = {term['flip_mask']:0{num_qubits}b}"
            )

    # Check H|psi> on a random state instead of comparing dense matrices.
    rng = np.random.default_rng()
    psi = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)

    print(
        "\nDecomposition is correct:",
        np.allclose(
            apply_pmr(pmr_terms, psi),
            apply_pauli_terms(pauli_terms, psi),
        ),
    )


if __name__ == "__main__":
    main()