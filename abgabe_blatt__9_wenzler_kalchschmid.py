import time
import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt

def jacobi_lgs(A, b, x0, maxiter=500, tol=1e-4):
    """
    Solves the linear system Ax = b using the Jacobi iterative method.

    Parameters:
    - A (ndarray): Coefficient matrix of size n x n.
    - b (ndarray): Right-hand side vector of size n.
    - x0 (ndarray): Initial guess for the solution vector of size n.
    - maxiter (int): Maximum number of iterations allowed (default: 500).
    - tol (float): Convergence tolerance for the relative residual (default: 1e-4).

    Returns:
    - x_new (ndarray): Approximate solution vector of size n.
    - iterates (list): List of solution vectors at each iteration.
    - k (int): Number of iterations performed.
    - elapsed_time (float): Total runtime of the method in seconds.

    Raises:
    - ValueError: If the method does not converge within the maximum number of iterations.

    Notes:
    - The method splits A into D (diagonal) and R (off-diagonal).
    - The stopping criterion is based on the relative residual norm ||rk|| <= tol * ||r0||.
    """
    n = len(b)
    x = x0.copy()
    iterates = [x.copy()]

    # Extract diagonal and off-diagonal components
    D = np.diag(A)  # Diagonal elements of A
    R = A - np.diagflat(D)  # Off-diagonal elements of A

    r0_norm = np.linalg.norm(b - np.dot(A, x0))  # Initial residual norm
    start_time = time.time()

    for k in range(maxiter):
        # Update the solution vector
        x_new = (b - np.dot(R, x)) / D
        iterates.append(x_new.copy())

        # Compute the residual norm
        r_norm = np.linalg.norm(b - np.dot(A, x_new))
        if r_norm <= tol * r0_norm:
            elapsed_time = time.time() - start_time
            return x_new, iterates, k + 1, elapsed_time
        x = x_new

    elapsed_time = time.time() - start_time
    raise ValueError(f"Jacobi method did not converge within {maxiter} iterations")


def gauss_seidel_lgs(A, b, x0, maxiter=500, tol=1e-4):
    """
    Solves the linear system Ax = b using the Gauss-Seidel iterative method.

    Parameters:
    - A (ndarray): Coefficient matrix of size n x n.
    - b (ndarray): Right-hand side vector of size n.
    - x0 (ndarray): Initial guess for the solution vector of size n.
    - maxiter (int): Maximum number of iterations allowed (default: 500).
    - tol (float): Convergence tolerance for the relative residual (default: 1e-4).

    Returns:
    - x_new (ndarray): Approximate solution vector of size n.
    - iterates (list): List of solution vectors at each iteration.
    - k (int): Number of iterations performed.
    - elapsed_time (float): Total runtime of the method in seconds.

    Raises:
    - ValueError: If the method does not converge within the maximum number of iterations.

    Notes:
    - The method splits A into (D + L) (lower triangular with diagonal) and U (strictly upper triangular).
    - The stopping criterion is based on the relative residual norm ||rk|| <= tol * ||r0||.
    """
    n = len(b)
    x = x0.copy()
    iterates = [x.copy()]

    # Split A into lower triangular and strictly upper triangular parts
    L_plus_D = np.tril(A)  # Lower triangular part including diagonal
    U = np.triu(A, k=1)  # Strictly upper triangular part (k=1 excludes the diagonal)

    r0_norm = np.linalg.norm(b - np.dot(A, x0))  # Initial residual norm
    start_time = time.time()

    for k in range(maxiter):
        # Compute the right-hand side
        rhs = b - np.dot(U, x)
        # Solve the triangular system (L + D)x_new = rhs
        x_new = np.linalg.solve(L_plus_D, rhs)

        iterates.append(x_new.copy())

        # Compute the residual norm
        r_norm = np.linalg.norm(b - np.dot(A, x_new))
        if r_norm <= tol * r0_norm:
            elapsed_time = time.time() - start_time
            return x_new, iterates, k + 1, elapsed_time
        x = x_new

    elapsed_time = time.time() - start_time
    raise ValueError(f"Gauss-Seidel method did not converge within {maxiter} iterations")


def construct_system(n):
    """
    Constructs the matrix A and vector b for the discretized integral equation.

    Parameters:
    - n (int): The size of the system (number of discretization points).

    Returns:
    - A (ndarray): Coefficient matrix of size n x n.
    - b (ndarray): Right-hand side vector of size n.

    Notes:
    - The discretization is based on points t_i = (i - 0.5) * h where h = 1 / n.
    - The diagonal of A is adjusted by adding 1/h.
    - b is a constant vector with all elements equal to 2/h.
    """
    h = 1 / n
    t = (np.arange(1, n + 1) - 0.5) * h  # Discretized points
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = np.cos(t[i] * t[j])
    np.fill_diagonal(A, A.diagonal() + 1 / h)  # Add 1/h to diagonal
    b = np.full((n,), 2 / h)  # Right-hand side vector
    return A, b



def is_diagonally_dominant(A):
    """
    Checks if the matrix A is strictly diagonally dominant.

    Parameters:
    - A (ndarray): Input matrix of size n x n.

    Returns:
    - (bool): True if A is strictly diagonally dominant, False otherwise.

    Notes:
    - A matrix is strictly diagonally dominant if |A[i, i]| > sum(|A[i, j]|) for all j ≠ i.
    - This condition ensures stability in certain iterative methods.
    """
    n = A.shape[0]
    for i in range(n):
        diag = abs(A[i, i])
        off_diag = np.sum(abs(A[i, :])) - diag
        if diag <= off_diag:
            return False
    return True


def is_symmetric_positive_definite(A):
    """
    Checks if the matrix A is symmetric positive definite.

    Parameters:
    - A (ndarray): Input matrix of size n x n.

    Returns:
    - (bool): True if A is symmetric positive definite, False otherwise.

    Notes:
    - A matrix is symmetric positive definite if:
        1. It is symmetric (A == A.T).
        2. All its eigenvalues are positive, which is verified using Cholesky decomposition.
    - If the Cholesky decomposition fails, the matrix is not positive definite.
    """
    # Check symmetry
    if not np.allclose(A, A.T):
        return False
    # Check positive definiteness
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def plot_iterates(residuals, method_name, n):
    """
    Plots the convergence of residual norms for iterative methods.

    Parameters:
    - residuals (list of lists): Residual norms for each iteration, one list per system size.
    - method_name (str): Name of the method (e.g., "Jacobi" or "Gauss-Seidel").
    - n (list): List of system sizes corresponding to the residuals.

    Notes:
    - Each curve in the plot corresponds to a specific system size (n).
    - Residual norms are plotted against iteration indices.
    """
    plt.figure()
    for n_val, res in zip(n, residuals):
        plt.plot(res, label=f"n={n_val}")
    plt.xlabel("Iterations")
    plt.ylabel("Residual Norm")
    plt.title(f"Convergence of {method_name}")
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == '__main__':
    #%% Part 1: Solve a small system to test Jacobi and Gauss-Seidel methods
    test_cases = [
        {
            "name": "Diagonally Dominant",
            "A": np.array([[4, -1, 0, 0],
                           [-1, 4, -1, 0],
                           [0, -1, 4, -1],
                           [0, 0, -1, 3]]),
            "b": np.array([15, 10, 10, 10]),
            "x0": np.ones(4)
        },
        {
            "name": "Symmetric Positive Definite",
            "A": np.array([[10, 2, 1],
                           [2, 8, 1],
                           [1, 1, 5]]),
            "b": np.array([7, 8, 6]),
            "x0": np.zeros(3)
        },
        {
            "name": "Non-Diagonally Dominant",
            "A": np.array([[2, 1, 1],
                           [1, 3, 2],
                           [1, 2, 4]]),
            "b": np.array([5, 6, 10]),
            "x0": np.zeros(3)
        },
        {
            "name": "Ill-Conditioned Matrix",
            "A": np.array([[1, 0.99, 0.98],
                           [0.99, 1, 0.99],
                           [0.98, 0.99, 1]]),
            "b": np.array([2.97, 2.97, 2.97]),
            "x0": np.zeros(3)
        }
    ]

    for case in test_cases:
        print(f"\nTest Case: {case['name']}")
        A, b, x0 = case["A"], case["b"], case["x0"]

        # Jacobi Method
        try:
            x_jacobi, jacobi_iterates, jacobi_iters, jacobi_time = jacobi_lgs(A, b, x0)
            print(f"Jacobi solution: {x_jacobi}")
            print(f"Jacobi iterations: {jacobi_iters}")
            print(f"Jacobi time: {jacobi_time:.6f} seconds")
        except ValueError as e:
            print(f"Jacobi method failed: {e}")

        # Gauss-Seidel Method
        try:
            x_gs, gs_iterates, gs_iters, gs_time = gauss_seidel_lgs(A, b, x0)
            print(f"Gauss-Seidel solution: {x_gs}")
            print(f"Gauss-Seidel iterations: {gs_iters}")
            print(f"Gauss-Seidel time: {gs_time:.6f} seconds")
        except ValueError as e:
            print(f"Gauss-Seidel method failed: {e}")

    #%% Part 2: Compare performance for increasing system sizes
    n_values = [16, 32, 64, 128, 256, 512]  # System sizes to test
    jacobi_times = []  # Store runtimes for the Jacobi method
    gs_times = []  # Store runtimes for the Gauss-Seidel method
    direct_times = []  # Store runtimes for the direct solver

    for n in n_values:
        # Construct the system (A, b) for the current size n
        A, b = construct_system(n)
        x0 = np.ones_like(b)  # Initial guess for the solution

        # Measure runtime of the Jacobi method
        start = time.time()
        jacobi_lgs(A, b, x0)
        jacobi_times.append(time.time() - start)

        # Measure runtime of the Gauss-Seidel method
        start = time.time()
        gauss_seidel_lgs(A, b, x0)
        gs_times.append(time.time() - start)

        # Measure runtime of the direct solver
        start = time.time()
        solve(A, b)
        direct_times.append(time.time() - start)

    # Plot runtime comparison
    plt.plot(n_values, jacobi_times, label="Jacobi")  # Jacobi runtime
    plt.plot(n_values, gs_times, label="Gauss-Seidel")  # Gauss-Seidel runtime
    plt.plot(n_values, direct_times, label="Direct")  # Direct solver runtime
    plt.xlabel("n (Matrix size)")
    plt.ylabel("Time (seconds)")
    plt.title("Runtime Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

    #%% Bonus Task: Analyze Residual Convergence and Matrix Properties
    n_values = [16, 32, 64, 128, 256, 512]  # Different system sizes for testing
    jacobi_residuals = []  # To store residual norms for Jacobi method
    gs_residuals = []  # To store residual norms for Gauss-Seidel method

    for n in n_values:
        # Construct the system (A, b) for the given size n
        A, b = construct_system(n)
        x0 = np.ones_like(b)  # Initial guess for the solution vector

        # Check matrix properties for the first system (n = 16) as a one-time verification
        if n == n_values[0]:
            print(f"Matrix is diagonally dominant: {is_diagonally_dominant(A)}")  # Check diagonal dominance
            print(f"Matrix is symmetric positive definite: {is_symmetric_positive_definite(A)}")  # Check SPD property

        # Solve using the Jacobi method and track residuals
        _, jacobi_iterates, _, _ = jacobi_lgs(A, b, x0)
        # Compute residual norms ||b - Ax|| for all iterations
        jacobi_residuals.append([np.linalg.norm(b - np.dot(A, x)) for x in jacobi_iterates])

        # Solve using the Gauss-Seidel method and track residuals
        _, gs_iterates, _, _ = gauss_seidel_lgs(A, b, x0)
        # Compute residual norms ||b - Ax|| for all iterations
        gs_residuals.append([np.linalg.norm(b - np.dot(A, x)) for x in gs_iterates])

    # Plot residual convergence for the Jacobi method
    plot_iterates(jacobi_residuals, "Jacobi Method", n_values)

    # Plot residual convergence for the Gauss-Seidel method
    plot_iterates(gs_residuals, "Gauss-Seidel Method", n_values)


