import numpy as np


def matrix_to_pmr(H, tolerance=1e-12):
    """
    Decompose H as

        H = sum_k D_k @ P_k

    where D_k is diagonal and P_k maps |s> to |s XOR k>.

    Terms whose diagonal entries are all <= tolerance are omitted.
    The dimension of H must be a power of two.
    """
    H = np.asarray(H, dtype=complex)

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be a square matrix.")

    dim = H.shape[0]

    if dim == 0 or dim & (dim - 1):
        raise ValueError(
            "The dimension of H must be a power of two."
        )

    states = np.arange(dim)
    pmr_terms = []

    for flip_mask in range(dim):
        diagonal = H[states, states ^ flip_mask]

        if np.abs(diagonal).max() > tolerance:
            pmr_terms.append(
                {
                    "index": len(pmr_terms),
                    "flip_mask": flip_mask,
                    "diagonal": diagonal,
                }
            )

    return pmr_terms


def reconstruct_pmr(pmr_terms, dimension):
    """Reconstruct H from its PMR terms."""
    H = np.zeros(
        (dimension, dimension),
        dtype=complex,
    )

    states = np.arange(dimension)

    for term in pmr_terms:
        H[
            states,
            states ^ term["flip_mask"]
        ] += term["diagonal"]

    return H


def pmr_term_matrices(term):
    """Construct dense D and P matrices for one PMR term."""
    diagonal = term["diagonal"]
    dim = diagonal.size
    states = np.arange(dim)

    P = np.zeros(
        (dim, dim),
        dtype=complex,
    )

    P[
        states ^ term["flip_mask"],
        states
    ] = 1.0

    D = np.diag(diagonal)

    return D, P


def print_pmr(pmr_terms):
    """Print the PMR decomposition."""
    expression = " + ".join(
        f"D{term['index']} @ P{term['index']}"
        for term in pmr_terms
    )

    print(
        "H =",
        expression if expression else "0",
    )

    for term in pmr_terms:
        index = term["index"]
        D, P = pmr_term_matrices(term)

        print(
            f"\nTerm {index}, "
            f"XOR flip mask = {term['flip_mask']}:"
        )

        print(f"D{index} =")
        print(D)

        print(f"P{index} =")
        print(P)