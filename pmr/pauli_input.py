import numpy as np

from matrix_pmr import reconstruct_pmr


_I_POWERS = (
    1 + 0j,
    1j,
    -1 + 0j,
    -1j,
)


def validate_pauli_string(pauli):
    """Validate a Pauli string."""
    pauli = pauli.upper().strip()

    if not pauli:
        raise ValueError(
            "Pauli string cannot be empty."
        )

    if any(
        operator not in "IXYZ"
        for operator in pauli
    ):
        raise ValueError(
            "Pauli strings may contain only I, X, Y, and Z."
        )

    return pauli


def _pauli_masks(pauli):
    """
    Return (x_mask, z_mask, num_y).

    P|s> = i^num_y * (-1)^popcount(s & z_mask) * |s XOR x_mask>
    The left-most character is the most significant bit (np.kron ordering).
    """
    n = len(pauli)

    x_mask = 0
    z_mask = 0
    num_y = 0

    for position, operator in enumerate(pauli):
        bit = 1 << (n - 1 - position)

        if operator in "XY":
            x_mask |= bit

        if operator in "YZ":
            z_mask |= bit

        if operator == "Y":
            num_y += 1

    return x_mask, z_mask, num_y


def _parity(values):
    """Return the parity of each integer."""
    values = np.array(
        values,
        dtype=np.int64,
    )

    for shift in (32, 16, 8, 4, 2, 1):
        values ^= values >> shift

    return values & 1


def pauli_terms_to_pmr(
    pauli_terms,
    tolerance=1e-12,
):
    """Convert Pauli terms directly into grouped PMR terms."""
    if not pauli_terms:
        raise ValueError(
            "At least one Pauli term is required."
        )

    paulis = [
        validate_pauli_string(pauli)
        for _, pauli in pauli_terms
    ]

    num_qubits = len(paulis[0])

    if any(
        len(pauli) != num_qubits
        for pauli in paulis
    ):
        raise ValueError(
            "All Pauli strings must have the same length."
        )

    dimension = 1 << num_qubits
    states = np.arange(
        dimension,
        dtype=np.int64,
    )

    diagonals = {}

    for (coefficient, _), pauli in zip(
        pauli_terms,
        paulis,
    ):
        x_mask, z_mask, num_y = _pauli_masks(
            pauli
        )

        phase = coefficient * _I_POWERS[num_y % 4]

        if z_mask:
            # <s| P |s ^ x_mask> = phase * (-1)^popcount((s ^ x_mask) & z_mask)
            sign = 1 - 2 * _parity((states ^ x_mask) & z_mask)
            contribution = phase * sign
        else:
            contribution = phase

        diagonal = diagonals.get(x_mask)

        if diagonal is None:
            diagonal = np.zeros(
                dimension,
                dtype=complex,
            )

            diagonals[x_mask] = diagonal

        diagonal += contribution

    kept_masks = [
        flip_mask
        for flip_mask in sorted(diagonals)
        if np.abs(
            diagonals[flip_mask]
        ).max() > tolerance
    ]

    return [
        {
            "index": index,
            "flip_mask": flip_mask,
            "diagonal": diagonals[flip_mask],
        }
        for index, flip_mask
        in enumerate(kept_masks)
    ]


def pauli_string_to_matrix(pauli):
    """Convert a Pauli string into its dense matrix representation."""
    pauli = validate_pauli_string(pauli)

    pmr_terms = pauli_terms_to_pmr(
        [(1, pauli)],
        tolerance=0.0,
    )

    dimension = 1 << len(pauli)

    return reconstruct_pmr(
        pmr_terms,
        dimension,
    )


def build_hamiltonian_from_paulis(
    pauli_terms,
):
    """Build a dense Hamiltonian from Pauli terms."""
    if not pauli_terms:
        raise ValueError(
            "At least one Pauli term is required."
        )

    num_qubits = len(
        validate_pauli_string(
            pauli_terms[0][1]
        )
    )

    pmr_terms = pauli_terms_to_pmr(
        pauli_terms,
        tolerance=0.0,
    )

    return reconstruct_pmr(
        pmr_terms,
        1 << num_qubits,
    )


def read_pauli_terms_from_terminal():
    """Read Pauli terms from the terminal."""
    while True:
        try:
            number_of_terms = int(
                input(
                    "Number of Pauli terms: "
                )
            )

            if number_of_terms <= 0:
                raise ValueError

            break

        except ValueError:
            print(
                "Please enter a positive integer."
            )

    pauli_terms = []
    num_qubits = None

    for term_index in range(
        number_of_terms
    ):
        print(
            f"\nTerm {term_index + 1}"
        )

        while True:
            try:
                coefficient = complex(
                    input(
                        "Coefficient: "
                    )
                    .lower()
                    .replace("i", "j")
                )

                break

            except ValueError:
                print(
                    "Please enter a valid coefficient."
                )

        while True:
            try:
                pauli = validate_pauli_string(
                    input(
                        "Pauli string: "
                    )
                )

                if num_qubits is None:
                    num_qubits = len(pauli)

                elif len(pauli) != num_qubits:
                    raise ValueError(
                        "All Pauli strings must "
                        "have the same length."
                    )

                break

            except ValueError as error:
                print(error)

        pauli_terms.append(
            (coefficient, pauli)
        )

    return pauli_terms