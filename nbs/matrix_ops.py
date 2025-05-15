# matrix_ops.py
# Matrix operations for numerical exercises, including matrix generation with loops and fast methods.
# Also includes runtime comparison of these methods.
import numpy as np
import timeit
import matplotlib.pyplot as plt


# %% Matrix Generation with Loops
def genmatrix(n, d, x):
    """
    Generates an n x n matrix with ones on the main diagonal and 'x' on the d-th
    sub-diagonal or super-diagonal.

    Input:
    - n (int): Matrix dimension, must be > 0.
    - d (int): Diagonal offset, valid values are within -n+1 to n-1.
    - x (float): Value to fill on the d-th diagonal.

    Output:
    - matrix (np.ndarray): The generated n x n matrix.
    """
    if not (0 < n and -(n - 1) <= d <= (n - 1)):
        raise ValueError("Invalid parameters for matrix generation.")

    matrix = np.zeros((n, n))
    np.fill_diagonal(matrix, 1)  # Main diagonal filled with ones

    # Loop to set d-th diagonal values
    for i in range(n):
        if 0 <= i + d < n:
            matrix[i, i + d] = x
    return matrix


def fastmatrix(n, d, x):
    """
    Quickly generates an n x n matrix with ones on the main diagonal and 'x' on the d-th
    sub-diagonal or super-diagonal, without loops using np.eye and np.diag.

    Input:
    - n (int): Matrix dimension, must be > 0.
    - d (int): Diagonal offset, valid values are within -n+1 to n-1.
    - x (float): Value to fill on the d-th diagonal.

    Output:
    - matrix (np.ndarray): The generated n x n matrix.
    """
    if not (0 < n and -(n - 1) <= d <= (n - 1)):
        raise ValueError("Invalid parameters for matrix generation.")

    matrix = np.eye(n)
    matrix += np.diag([x] * (n - abs(d)), k=d)  # Fill specified diagonal
    return matrix


def compare_matrices(n_values, d, x):
    """
    Compares the runtime of genmatrix and fastmatrix functions for a range of matrix sizes
    and plots the results on a semilogarithmic scale.

    Input:
    - n_values (list of int): List of matrix dimensions to test.
    - d (int): Diagonal offset for matrix generation.
    - x (float): Value to place on the d-th diagonal.

    Output:
    - A plot comparing execution times for each method at different matrix dimensions.
    """
    times_gen = []
    times_fast = []

    # Record execution times for each matrix size in n_values
    for n in n_values:
        time_gen = timeit.timeit(lambda: genmatrix(n, d, x), number=10)
        time_fast = timeit.timeit(lambda: fastmatrix(n, d, x), number=10)
        times_gen.append(time_gen)
        times_fast.append(time_fast)

    # Plotting the runtime comparison
    plt.semilogy(n_values, times_gen, label="genmatrix (with loops)")
    plt.semilogy(n_values, times_fast, label="fastmatrix (without loops)")
    plt.xlabel("Matrix Dimension (n)")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.savefig("matrix_comparison.pdf", format="pdf")

    plt.show()
