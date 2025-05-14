# linsys_conditioning.py
import numpy as np
import matplotlib.pyplot as plt

#%% Solving Linear Systems
def solve_linear_systems():
    """
    Solves the system Ax = b for each i in {1, ..., 10} using numpy.linalg.solve
    and stores the solutions for later comparison with the analytical solution.

    Output:
    - solutions (list): A list of solutions xi for i in {1, ..., 10}.
    """
    solutions = []
    for i in range(1, 11):
        A_i = np.array([[10**-i, 1], [0, 1]])
        b_i = np.array([10**-i + 1, 1])
        x_i = np.linalg.solve(A_i, b_i)
        solutions.append(x_i)
    return solutions

#%% Calculating Relative Errors
def calculate_relative_errors(numerical_solutions, analytical_solution):
    """
    Calculates the relative errors between the numerical solutions and the analytical solution.

    Input:
    - numerical_solutions (list of np.ndarray): Numerical solutions xi from solve_linear_systems.
    - analytical_solution (np.ndarray): The manually computed analytical solution.

    Output:
    - relative_errors (list): A list of relative errors for each solution xi.
    """
    relative_errors = []
    for xi in numerical_solutions:
        error = np.linalg.norm(xi - analytical_solution) / np.linalg.norm(analytical_solution)
        relative_errors.append(error)
    return relative_errors

#%% Plotting Conditioning vs. Error
def plot_conditioning_vs_error():
    """
    Plots the conditioning numbers of matrices Ai against the relative errors of the solutions xi.

    Output:
    - A log-log plot showing the relationship between conditioning numbers and relative errors.
    """
    analytical_solution = np.array([0, 1])  # Assuming analytical solution determined by inspection.
    numerical_solutions = solve_linear_systems()
    relative_errors = calculate_relative_errors(numerical_solutions, analytical_solution)

    condition_numbers = []
    for i in range(1, 11):
        A_i = np.array([[10**-i, 1], [0, 1]])
        cond_Ai = np.linalg.cond(A_i, p=1)
        condition_numbers.append(cond_Ai)

    # Log-log plot of conditioning numbers versus relative errors
    plt.loglog(condition_numbers, relative_errors, 'o-', label="Relative Error vs. Condition Number")
    plt.xlabel("Condition Number (1-norm)")
    plt.ylabel("Relative Error")
    plt.title("Condition Number vs. Relative Error for Ai")
    plt.legend()
    plt.show()
