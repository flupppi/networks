import time
import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt


def jacobi_lgs(A, b, x0, maxiter=500, tol=1e-4):
    n = len(b)
    x = x0.copy()
    iterates = [x.copy()]

    # Use numpy.diag and numpy.diagflat
    D = np.diag(A)  # Diagonal elements of A
    R = A - np.diagflat(D)  # Off-diagonal elements of A

    r0_norm = np.linalg.norm(b - np.dot(A, x0))
    start_time = time.time()

    for k in range(maxiter):
        x_new = (b - np.dot(R, x)) / D
        iterates.append(x_new.copy())

        r_norm = np.linalg.norm(b - np.dot(A, x_new))
        if r_norm <= tol * r0_norm:
            elapsed_time = time.time() - start_time
            return x_new, iterates, k + 1, elapsed_time
        x = x_new

    elapsed_time = time.time() - start_time
    raise ValueError(f"Jacobi method did not converge within {maxiter} iterations")


def gauss_seidel_lgs(A, b, x0, maxiter=500, tol=1e-4):
    n = len(b)
    x = x0.copy()
    iterates = [x.copy()]

    # Use numpy.tril and numpy.triu
    L_plus_D = np.tril(A)  # Lower triangular part including diagonal
    U = np.triu(A, k=1)  # Strictly upper triangular part (k=1 excludes the diagonal)

    r0_norm = np.linalg.norm(b - np.dot(A, x0))
    start_time = time.time()

    for k in range(maxiter):
        # Compute the right-hand side of the equation
        rhs = b - np.dot(U, x)
        # Solve the triangular system (D + L)x_new = rhs
        x_new = np.linalg.solve(L_plus_D, rhs)

        iterates.append(x_new.copy())

        r_norm = np.linalg.norm(b - np.dot(A, x_new))
        if r_norm <= tol * r0_norm:
            elapsed_time = time.time() - start_time
            return x_new, iterates, k + 1, elapsed_time
        x = x_new

    elapsed_time = time.time() - start_time
    raise ValueError(f"Gauss-Seidel method did not converge within {maxiter} iterations")



def construct_system(n):
    h = 1 / n
    t = (np.arange(1, n + 1) - 0.5) * h  # Discretized points
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = np.cos(t[i] * t[j])
    np.fill_diagonal(A, A.diagonal() + 1 / h)  # Add 1/h to diagonal
    b = np.full((n,), 2 / h)  # Right-hand side vector
    return A, b


def direct_solver(A, b):
    return solve(A, b)



def is_diagonally_dominant(A):
    n = A.shape[0]
    for i in range(n):
        diag = abs(A[i, i])
        off_diag = np.sum(abs(A[i, :])) - diag
        if diag <= off_diag:
            return False
    return True


def is_symmetric_positive_definite(A):
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
    #%% Part 1
    A = np.array([[4, -1, 0, 0],
                  [-1, 4, -1, 0],
                  [0, -1, 4, -1],
                  [0, 0, -1, 3]])
    b = np.array([15, 10, 10, 10])
    x0 = np.ones_like(b)

    # Jacobi
    x_jacobi, jacobi_iterates, jacobi_iters, jacobi_time = jacobi_lgs(A, b, x0)
    print(f"Jacobi solution: {x_jacobi}")
    print(f"Jacobi iterations: {jacobi_iters}")
    print(f"Jacobi time: {jacobi_time:.6f} seconds")

    # Gauss-Seidel
    x_gs, gs_iterates, gs_iters, gs_time = gauss_seidel_lgs(A, b, x0)
    print(f"Gauss-Seidel solution: {x_gs}")
    print(f"Gauss-Seidel iterations: {gs_iters}")
    print(f"Gauss-Seidel time: {gs_time:.6f} seconds")

    #%% Part 2
    n_values = [16, 32, 64, 128, 256, 512]
    jacobi_times = []
    gs_times = []
    direct_times = []

    for n in n_values:
        A, b = construct_system(n)
        x0 = np.ones_like(b)

        # Jacobi
        start = time.time()
        jacobi_lgs(A, b, x0)
        jacobi_times.append(time.time() - start)

        # Gauss-Seidel
        start = time.time()
        gauss_seidel_lgs(A, b, x0)
        gs_times.append(time.time() - start)

        # Direct solver
        start = time.time()
        direct_solver(A, b)
        direct_times.append(time.time() - start)

    # Plot comparison
    plt.plot(n_values, jacobi_times, label="Jacobi")
    plt.plot(n_values, gs_times, label="Gauss-Seidel")
    plt.plot(n_values, direct_times, label="Direct")
    plt.xlabel("n (Matrix size)")
    plt.ylabel("Time (seconds)")
    plt.title("Runtime Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

    #%% Bonus Task
    n_values = [16, 32, 64, 128, 256, 512]
    jacobi_residuals = []
    gs_residuals = []

    for n in n_values:
        # Construct the system
        A, b = construct_system(n)
        x0 = np.ones_like(b)

        # Check matrix properties
        if n == n_values[0]:  # Check once for clarity
            print(f"Matrix is diagonally dominant: {is_diagonally_dominant(A)}")
            print(f"Matrix is symmetric positive definite: {is_symmetric_positive_definite(A)}")

        # Jacobi method
        _, jacobi_iterates, _, _ = jacobi_lgs(A, b, x0)
        jacobi_residuals.append([np.linalg.norm(b - np.dot(A, x)) for x in jacobi_iterates])

        # Gauss-Seidel method
        _, gs_iterates, _, _ = gauss_seidel_lgs(A, b, x0)
        gs_residuals.append([np.linalg.norm(b - np.dot(A, x)) for x in gs_iterates])

    # Plot residuals
    plot_iterates(jacobi_residuals, "Jacobi Method", n_values)
    plot_iterates(gs_residuals, "Gauss-Seidel Method", n_values)

