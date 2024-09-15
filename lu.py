
# remember UMFPACK

import numpy as np
import scipy.linalg as la

def gemm(A: np.ndarray) -> np.ndarray:
    n = A.shape[0] # A is a square matrix
    P = np.arange(n, dtype=np.int8)
    for j in range (n):
        # find the max elt in this column
        max_idx = np.argmax(A[j:, j].__abs__(), axis=0) 
        if j != max_idx:
            P[[j, max_idx]] = P[[max_idx, j]]
            A[[max_idx, j]] = A[[j, max_idx]]
        for i in range(j+1, n):
            A[i, j] /= A[j, j]
            for k in range(j+1, n):
                A[i, k] -= A[i, j] * A[j, k]
    return P


def main():
    n = 3
    A = np.random.random((n, n))
    A_cp = A.copy()
    print(f"matrix A:\n {A}")

    Perm = gemm(A)
    print(f"matrix A:\n {A}")
    print(f"array P: {Perm}")

    P, l, U = la.lu(A_cp)
    print(f"matrix L:\n {l}")
    print(f"array U: {U}")

    print(f"perm A:\n{A[Perm]}")

if __name__ == "__main__":
    main()
    
