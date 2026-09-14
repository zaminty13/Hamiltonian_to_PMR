from pmr.grouping import hamiltonian_to_pmr


def print_pmr(groups):
    print("\nPMR Representation")
    print("------------------")

    for P, diagonal_terms in groups.items():

        print(f"\nP = {P}")
        print("D =", end=" ")

        first_term = True

        for D, coefficient in diagonal_terms.items():

            if coefficient == 0:
                continue

            if not first_term:
                print(" + ", end="")

            print(f"({coefficient}){D}", end="")

            first_term = False

        print()


hamiltonian = [
    (2, "XI"),
    (3, "XZ"),
    (4, "YI"),
    (5, "IZ")
]


pmr = hamiltonian_to_pmr(hamiltonian)

print_pmr(pmr)