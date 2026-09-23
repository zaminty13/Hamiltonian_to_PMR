import numpy as np

from pmr.pauli_input import (
    read_pauli_terms_from_terminal,
    build_hamiltonian_from_paulis,
)

from pmr.matrix_pmr import (
    matrix_to_pmr,
    reconstruct_pmr,
    print_pmr,
)


def main():

    # Your Pauli-string input
    pauli_terms = read_pauli_terms_from_terminal()

    print("\nPauli Hamiltonian:")

    for coefficient, pauli in pauli_terms:
        print(f"  ({coefficient}) * {pauli}")

    # Convert your Pauli input into matrix H
    H = build_hamiltonian_from_paulis(pauli_terms)

    print("\nHamiltonian matrix H =")
    print(H)

    # PI's matrix-to-PMR algorithm
    pmr_terms = matrix_to_pmr(H)

    print("\nPMR decomposition:")
    print_pmr(pmr_terms)

    # Reconstruct H from PMR
    H_reconstructed = reconstruct_pmr(
        pmr_terms,
        dimension=H.shape[0],
    )

    print("\nReconstructed H =")
    print(H_reconstructed)

    # Verify
    print(
        "\nDecomposition is correct:",
        np.allclose(H, H_reconstructed),
    )


if __name__ == "__main__":
    main()
