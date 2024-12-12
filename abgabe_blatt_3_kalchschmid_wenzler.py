#!/usr/bin/env python3
import numpy as np
from scipy.linalg import solve_triangular, lu

"""
Numerische Mathematik - Abgabe Blatt Nr. 03
Tutor(-in): Samuel Inca Pilco

Felix Kalchschmid, Leon Wenzler
"""


def trisolve(M: np.ndarray, b: np.array, threshold: float = 1e-8) -> np.ndarray:
    """
    Solves either a lower or upper triangular system of Mx = b.
    This is a minimized version of the `trisolve` function we submitted last sheet.

    :param A: triangular input matrix
    :param b: input vector of same size
    :return: solution vector x
    """

    # Initialize the solution vector x with zeros.
    n = len(M)
    x = np.zeros_like(b, dtype=float)

    # Check if M is a lower triangular matrix.
    if np.allclose(M, np.tril(M)):
        for i in range(n):
            if np.abs(M[i, i]) < threshold:
                raise ZeroDivisionError(f"Diagonal element at ({i},{i}) is zero")

            # This is the dot product of the i-th row of M up to column i (excluding M[i,i]) and the known x values.
            sum_ = np.dot(M[i, :i], x[:i])
            # Compute x[i] using the formula for forward substitution.
            x[i] = (b[i] - sum_) / M[i, i]
        return x

    # Check if M is an upper triangular matrix.
    if np.allclose(M, np.triu(M)):
        for i in reversed(range(n)):
            if np.abs(M[i, i]) < threshold:
                raise ZeroDivisionError(f"Diagonal element at ({i},{i}) is zero")

            # This is the dot product of the i-th row of M from column i+1 to end and the known x values.
            sum_ = np.dot(M[i, i + 1 :], x[i + 1 :])
            # Compute x[i] using the formula for backward substitution.
            x[i] = (b[i] - sum_) / M[i, i]
        return x

    # If M is neither lower nor upper triangular, we cannot solve using this method.
    raise ValueError("Input matrix is not triangular")


def gauss_decomposition_pivot(
    A: np.ndarray, threshold: float = 1e-8
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs the full Gauss elimination algorithm using scaling and pivoting.
    Implements algorithm 2.5 from the lecture script, with optimizations where
    possible by replacing loops with vectorized operations.

    :param threshold:
    :param A: quadratic input matrix
    :return: scaling D, permutation P, lower triangular L, and upper triangular U matrices
    """

    # %% Exercise 5.1
    # Using A to store LU-decomposition in-place, thus copy to avoid input modifications.
    A = A.copy()
    n = len(A)
    # Initialize the scaling and permutation matrices D,P as identities.
    D, P = np.eye(n), np.eye(n)

    # Scale i-th row in matrix A according to the inverse of one-norm.
    for i in range(n):
        scale = 1 / np.linalg.norm(A[i], ord=1)
        A[i] *= scale
        D[i] *= scale

    # For every column except the last, we find the largest absolute
    # value and swap its row to the j-th, adapting P accordingly.
    # This implements pivoting (i.e. permutation matrices) efficiently.
    for j in range(n - 1):
        p = j + np.argmax(np.abs(A[j:, j]))
        A[[j, p]] = A[[p, j]]
        P[[j, p]] = P[[p, j]]
        # If we are close or equal to zero, cannot proceed to decompose
        # into LU matrices, thus abort with exception.
        if np.abs(A[j, j]) < threshold:
            raise ValueError("Input matrix is (nearly) singular")

        # Solve row-wise by normalizing with diagonal element, according
        # to Gauss elimination algorithm.
        for i in range(j + 1, n):
            A[i, j] /= A[j, j]
            for k in range(j + 1, n):
                A[i, k] -= A[i, j] * A[j, k]

    # Retrieve lower left and upper right matrices from A.
    L = np.tril(A, k=-1) + np.eye(n)
    U = np.triu(A)
    return D, P, L, U


def gauss_elimination_pivot(
    A: np.ndarray, b: np.ndarray, threshold: float = 1e-8
) -> np.ndarray:
    """
    Uses the LU-decomposition with pivoting from `gauss_decomposition_pivot` to retrieve P, L, U.
    Then, solves the equation systems $Ly = Pb$ and $Ux = y$ to obtain the solutionv vector x.

    :param A: quadratic input matrix
    :param b: input vector of same size
    :return: solution vector x
    """

    D, P, L, U = gauss_decomposition_pivot(A, threshold=threshold)
    y = trisolve(L, P @ D @ b, threshold=threshold)
    return trisolve(U, y, threshold=threshold), P


# Aufgabe 4.1: Solution of linear systems
def solve_linear_system(M: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solves a linear system Mx = b using NumPy's built-in solver.

    :param M: Coefficient matrix
    :param b: Right-hand side vector
    :return: Solution vector x
    """
    print(f"Solving Mx = b with:\nM =\n{M}\nb = {b}\n")
    if np.isclose(np.linalg.det(M), 0):
        raise ValueError("Matrix is singular or nearly singular. Solution not possible.")
    x = np.linalg.solve(M, b)
    print(f"Solution:\n{x}\n")
    return x


def calculate_condition_numbers(M: np.ndarray) -> tuple[float, float]:
    """
    Calculates and returns the Frobenius and 2-norm condition numbers of the matrix M.

    :param M: Input matrix
    :return: Frobenius and 2-norm condition numbers
    """
    frobenius_cond = np.linalg.cond(M, 'fro')
    two_norm_cond = np.linalg.cond(M, 2)
    print(f"Frobenius norm condition number: {frobenius_cond}")
    print(f"2-norm condition number: {two_norm_cond}\n")
    return frobenius_cond, two_norm_cond


# Aufgabe 4.2: LU Decomposition with Forward/Backward Substitution
def lu_decomposition_solution(M: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solves a linear system Mx = b using LU decomposition and forward/backward substitution.

    :param M: Coefficient matrix
    :param b: Right-hand side vector
    :return: Solution vector x
    """
    print(f"Aufgabe 4.2: Performing LU decomposition on M:\n{M}\n")
    P, L, U = lu(M)
    print(f"Permutation matrix P:\n{P}")
    print(f"Lower triangular matrix L:\n{L}")
    print(f"Upper triangular matrix U:\n{U}\n")

    Pb = np.dot(P, b)
    y = solve_triangular(L, Pb, lower=True)
    print(f"Intermediate solution y (from Ly = Pb):\n{y}")
    x = solve_triangular(U, y, lower=False)
    print(f"Solution x (from Ux = y):\n{x}\n")
    return x


# Aufgabe 4.3: Comparison of Solutions
def compare_solutions(M: np.ndarray, b: np.ndarray):
    """
    Compares solutions obtained from direct solve and LU decomposition.

    :param M: Coefficient matrix
    :param b: Right-hand side vector
    """
    print("Aufgabe 4.3: Comparing solutions...\n")
    solution_direct = np.linalg.solve(M, b)
    solution_lu = lu_decomposition_solution(M, b)
    is_close = np.allclose(solution_direct, solution_lu)
    norm_diff = np.linalg.norm(solution_direct - solution_lu)
    print(f"Are the solutions close? {'Yes' if is_close else 'No'}")
    print(f"Norm of the difference between solutions: {norm_diff}\n")


if __name__ == "__main__":
    # %% Aufgabe 4.1: Solve Linear Systems
    print("Aufgabe 4.1a")
    A1 = np.array([[2, -1, -3, 3],
                   [4, 0, -3, 1],
                   [6, 1, -1, 6],
                   [-2, -5, 4, 1]], dtype=float)
    b1 = np.array([1, -8, -16, -12], dtype=float)

    try:
        solve_linear_system(A1, b1)
    except ValueError as e:
        print(e)
    calculate_condition_numbers(A1)
    compare_solutions(A1, b1)

    print("Aufgabe 4.1b")
    A2 = np.array([[1, 0, 6, 2],
                   [8, 0, -2, -2],
                   [2, 9, 1, 3],
                   [2, 1, -3, 10]], dtype=float)
    b2 = np.array([6, -2, -8, -4], dtype=float)

    try:
        solve_linear_system(A2, b2)
    except ValueError as e:
        print(e)
    calculate_condition_numbers(A2)
    compare_solutions(A2, b2)

    A = np.array([[1, 5, 0], [2, 2, 2], [-2, 0, 2]], dtype=float)
    D, P, L, U = gauss_decomposition_pivot(A)
    assert np.allclose(D, np.diag([1 / 6, 1 / 6, 1 / 4]))
    assert np.allclose(P, np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]))
    assert np.allclose(L, np.array([[1, 0, 0], [-1 / 3, 1, 0], [-2 / 3, 2 / 5, 1]]))
    assert np.allclose(
        U, np.array([[-1 / 2, 0, 1 / 2], [0, 5 / 6, 1 / 6], [0, 0, 3 / 5]])
    )
    print(
        "[\u2713] Solution from `gauss_decomposition_pivot` matches the example 2.23 from the script."
    )

    # %% Exercise 5.2
    A = np.array(
        [[2, -1, -3, 3], [4, 0, -3, 1], [6, 1, -1, 6], [-2, -5, 4, 1]], dtype=float
    )
    b = np.array([1, -8, -16, -12], dtype=float)
    x, _ = gauss_elimination_pivot(A, b)
    assert np.allclose(x, np.linalg.solve(A, b))
    print(
        "[\u2713] Solution from `gauss_elimination_pivot` matches the result from exercise 4.1a."
    )
    A = np.array(
        [[1, 0, 6, 2], [8, 0, -2, -2], [2, 9, 1, 3], [2, 1, -3, 10]], dtype=float
    )
    b = np.array([6, -2, -8, -4], dtype=float)
    x, _ = gauss_elimination_pivot(A, b)
    assert np.allclose(x, np.linalg.solve(A, b))
    print(
        "[\u2713] Solution from `gauss_elimination_pivot` matches the result from exercise 4.1b."
    )
