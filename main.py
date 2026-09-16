from pmr.converter import convert_to_pmr
from pmr.verification import (
    original_hamiltonian_matrix,
    pmr_hamiltonian_matrix,
    verify
)

hamiltonian = [
    (1.0, "XI"),
    (0.5, "ZY"),
    (-0.25, "IZ")
]

pmr_terms = convert_to_pmr(hamiltonian)

for i, (D, P) in enumerate(pmr_terms):
    print(f"\nTerm {i + 1}")

    print("\nD =")
    print(D)

    print("\nP =")
    print(P)

    print("\nD @ P =")
    print(D @ P)

print("\nOriginal Hamiltonian:")
print(original_hamiltonian_matrix(hamiltonian))

print("\nHamiltonian from PMR:")
print(pmr_hamiltonian_matrix(pmr_terms))

print("\nVerification:")
print(verify(hamiltonian, pmr_terms))