import numpy as np


def matrix_to_pmr(H, tolerance=1e-12):
    """
    Decompose H as:

        H = sum_k D_k @ P_k

    where:
      - D_k is diagonal.
      - P_k maps |s> to |s XOR k>.
      - Terms whose diagonal coefficients are all smaller than
        tolerance are omitted.

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

        diagonal = H[
            states,
            states ^ flip_mask
        ].copy()

        if np.all(np.abs(diagonal) <= tolerance):
            continue

        P = np.zeros((dim, dim), dtype=complex)

        P[
            states ^ flip_mask,
            states
        ] = 1.0

        D = np.diag(diagonal)

        pmr_terms.append(
            {
                "index": len(pmr_terms),
                "flip_mask": flip_mask,
                "diagonal": diagonal,
                "D": D,
                "P": P,
            }
        )

    return pmr_terms


def reconstruct_pmr(pmr_terms, dimension):
    """Reconstruct H from its PMR terms."""

    H_reconstructed = np.zeros(
        (dimension, dimension),
        dtype=complex,
    )

    for term in pmr_terms:
        H_reconstructed += term["D"] @ term["P"]

    return H_reconstructed


def print_pmr(pmr_terms):
    """Print the PMR decomposition."""

    expression = " + ".join(
        f"D{term['index']} @ P{term['index']}"
        for term in pmr_terms
    )

    print("H =", expression if expression else "0")

    for term in pmr_terms:

        index = term["index"]
        mask = term["flip_mask"]

        print(
            f"\nTerm {index}, XOR flip mask = {mask}:"
        )

        print(f"D{index} =")
        print(term["D"])

        print(f"P{index} =")
        print(term["P"])
