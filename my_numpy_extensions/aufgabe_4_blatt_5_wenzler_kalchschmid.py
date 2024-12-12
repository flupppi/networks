# Wenzler, Kalchschmid - Numerische Mathematik - Blatt 5, Aufgabe 4

import numpy as np
import time
import matplotlib.pyplot as plt


def main():
    # Example usage of LU decomposition
    A = np.array([[4, 1, 0, 0],
                  [2, 3, 1, 0],
                  [0, 1, 3, 1],
                  [0, 0, 2, 4]])
    print("Input matrix A:")
    print(A)
    print("\nLU decomposition (L, U):")
    print(LU_tridiag(A))


def LU_tridiag(A):
    """
    Performs LU decomposition for a tridiagonal matrix A.

    Parameters:
    A (numpy.ndarray): The input matrix, assumed to be tridiagonal and square.

    Returns:
    L (numpy.ndarray): The lower triangular matrix with ones on the diagonal.
    U (numpy.ndarray): The upper triangular matrix.
    """
    # Ensure the input matrix is square
    if np.shape(A)[0] != np.shape(A)[1]:
        print("Can only process square matrices")
        return -1

    n = np.shape(A)[0]  # Size of the matrix
    L = np.eye(n)  # Initialize L as the identity matrix
    U = np.eye(n)  # Initialize U as the identity matrix

    # First diagonal element of U
    U[0][0] = A[0][0]

    for i in range(1, n):
        # Calculate the multipliers for L
        L[i][i - 1] = A[i][i - 1] / U[i - 1][i - 1]
        # Copy the off-diagonal elements to U
        U[i - 1][i] = A[i - 1][i]
        # Calculate the diagonal elements of U
        U[i][i] = A[i][i] - L[i][i - 1] * U[i - 1][i]

    return L, U


def LU_tridiag_solve(L, U, b):
    """
    Solves the system LUx = b using forward and backward substitution.

    Parameters:
    L (numpy.ndarray): The lower triangular matrix from LU decomposition.
    U (numpy.ndarray): The upper triangular matrix from LU decomposition.
    b (numpy.ndarray): The right-hand side vector.

    Returns:
    x (numpy.ndarray): The solution vector.
    """
    n = len(b)

    # Forward substitution to solve Ly = b
    y = np.zeros_like(b)
    y[0] = b[0]
    for i in range(1, n):
        y[i] = b[i] - L[i][i - 1] * y[i - 1]

    # Backward substitution to solve Ux = y
    x = np.zeros_like(b)
    x[-1] = y[-1] / U[-1][-1]
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - U[i][i + 1] * x[i + 1]) / U[i][i]

    return x


def generate_tridiagonal_matrix(n):
    """
    Generates a random symmetric, positive-definite tridiagonal matrix.

    Parameters:
    n (int): The size of the matrix.

    Returns:
    A (numpy.ndarray): The generated tridiagonal matrix.
    """
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = np.random.randint(2, 10)  # Main diagonal elements
        if i > 0:
            A[i, i - 1] = np.random.randint(1, 5)  # Subdiagonal elements
            A[i - 1, i] = A[i, i - 1]  # Superdiagonal elements (symmetric)
    return A


def compare_solvers():
    """
    Compares the runtime of the custom LU solver for tridiagonal matrices with
    numpy.linalg.solve on matrices of different sizes.

    The comparison is plotted using log-log plots.
    """
    sizes = [10, 100, 1000, 10000]  # Matrix sizes to test
    lu_times = []  # Runtime for LU solver
    numpy_times = []  # Runtime for numpy.linalg.solve

    for n in sizes:
        # Generate a random tridiagonal matrix and right-hand side vector
        A = generate_tridiagonal_matrix(n)
        b = np.random.rand(n)

        # LU decomposition and solving with custom solver
        start_time = time.time()
        L, U = LU_tridiag(A)
        x_lu = LU_tridiag_solve(L, U, b)
        lu_times.append(time.time() - start_time)

        # Solving directly with numpy.linalg.solve
        start_time = time.time()
        x_numpy = np.linalg.solve(A, b)
        numpy_times.append(time.time() - start_time)

    # Plot runtime comparison
    plt.loglog(sizes, lu_times, label="LU tridiag solve (custom)")
    plt.loglog(sizes, numpy_times, label="numpy.linalg.solve")
    plt.xlabel("Matrix size (n)")
    plt.ylabel("Runtime (s)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.title("Runtime Comparison of LU Solver and numpy.linalg.solve")
    plt.savefig("runtime_comparison.png")

    plt.show()


if __name__ == '__main__':
    compare_solvers()
