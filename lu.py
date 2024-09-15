
# remember UMFPACK

import numpy as np
import scipy.linalg as la

def gemm(A: np.ndarray) -> np.ndarray:
    n = A.shape[0] # A is a square matrix
    P = np.zeros(n, dtype=np.int8)
    for j in range (n):
        # find the max elt in this column
        max_idx = np.argmax(A[:, j].__abs__(), axis=0) 
        P[j] = max_idx
        P[max_idx] = j
        temp = A[max_idx, :]
        A[max_idx, :] = A[j, :]
        A[j, :] = temp
        for i in range(j+1, n):
            A[i, j] /= A[j, j]
            for k in range(j+1, n):
                A[i, k] -= A[i, j] * A[j, k]
    return P


def main():
    n = 3
    A = np.random.random((n, n))
    print(f"matrix A:\n {A}")

    P = gemm(A)
    print(f"matrix A:\n {A}")
    print(f"array P: {P}")

    # Bsp, Psp = la.lu()

if __name__ == "__main__":
    main()
    
