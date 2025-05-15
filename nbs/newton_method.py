import numpy as np
import matplotlib.pyplot as plt


# %% Task 1
def f(x):
    return np.sin(x)


def f_prime(x):
    return np.cos(x)


def fixpoint_iteration(x0, tol=1e-6, max_iter=50):
    iterates = [x0]
    for _ in range(max_iter):
        x_new = x0 + f(x0)
        iterates.append(x_new)
        if abs(x_new - x0) < tol:
            break
        x0 = x_new
    return iterates





def aufgabe1():
    x0 = 3  # Initial guess
    fix_iter = fixpoint_iteration(x0)
    newton_iter = newton_method(x0)

    # Visualize convergence
    plt.figure()
    plt.plot(fix_iter, label="Fixpoint Iteration")
    plt.plot(newton_iter, label="Newton Method")
    plt.axhline(y=np.pi, color='r', linestyle='--', label="True Root (π)")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Convergence of Iterative Methods")
    plt.show()


# %% Task 2

# Eine funktion die Approximiert werden soll
def g(x):
    return np.sin(x / 2)

# Die erste Ableitung von g
def g_prime(x):
    return 0.5 * np.cos(x / 2)

# Die zweite Ableitung von g
def g_double_prime(x):
    return -0.25 * np.sin(x / 2)


def taylor_iteration(x0, tol=1e-6, max_iter=3):
    iterates = [x0]
    for _ in range(max_iter):
        # Compute coefficients of quadratic approximation
        f_val = g(x0)
        f_prime_val = g_prime(x0)
        f_double_prime_val = g_double_prime(x0)

        # Solve quadratic equation
        a = 0.5 * f_double_prime_val
        b = f_prime_val
        c = f_val
        roots = np.roots([a, b, c])  # Solve a*x^2 + b*x + c = 0
        x_new = x0 + roots[np.argmin(abs(roots))]  # Pick the root closer to x0

        iterates.append(x_new)
        if abs(x_new - x0) < tol:
            break
        x0 = x_new
    return iterates


def aufgabe2():
    x0 = 2  # Initial guess
    taylor_iter = taylor_iteration(x0)

    # Convergence order estimation
    x_bar = 0  # True root
    alpha_values = []
    for k in range(len(taylor_iter) - 1):
        alpha_k = np.log(abs(x_bar - taylor_iter[k + 1])) / np.log(abs(x_bar - taylor_iter[k]))
        alpha_values.append(alpha_k)

    # Visualize results
    plt.figure()
    plt.plot(taylor_iter, label="Taylor Iteration")
    plt.axhline(y=x_bar, color='r', linestyle='--', label="True Root (0)")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Convergence of Taylor Method")
    plt.show()

    print("Convergence Order Estimates:", alpha_values)


# %% Task 3

def h(x):
    # Define a function with a known root (e.g., x̄ = 1)
    return (x - 1) ** 3 - 1


def h_prime(x):
    # Derivative of h(x)
    return 3 * (x - 1) ** 2


def simplified_newton(x0, tol=1e-6, max_iter=50):
    root = 1  # True root for the example function
    h_prime_x0 = h_prime(x0)  # Fixed derivative at the initial point
    iterates = [x0]
    errors = [abs(x0 - root)]

    for _ in range(max_iter):
        x_new = x0 - h(x0) / h_prime_x0
        iterates.append(x_new)
        errors.append(abs(x_new - root))

        if abs(x_new - x0) < tol:
            break
        x0 = x_new

    return iterates, errors


def aufgabe3():
    x0 = 2  # Initial guess
    iterates, errors = simplified_newton(x0)

    # Visualize errors
    plt.figure()
    plt.plot(errors, label="Error |x_k - x̄|")
    plt.axhline(y=0, color='r', linestyle='--', label="True Root (1)")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Error (log scale)")
    plt.title("Convergence of Simplified Newton Method")
    plt.show()

    # Print errors for inspection
    print("Errors:", errors)


# Uncomment the following to run specific tasks:
#aufgabe1()
#aufgabe2()
aufgabe3()


# %% Task 4


# Implement the newton method

def newton_method(x0, tol=1e-6, max_iter=50):
    iterates = [x0]
    for _ in range(max_iter):
        x_new = x0 - f(x0) / f_prime(x0)
        iterates.append(x_new)
        if abs(x_new - x0) < tol:
            break
        x0 = x_new
    return iterates
# Implement the newton method with linesearch
def newton(f, df, x0, maxiter, tol):
    pass

def newton_linesearch(f, df, x0, q,  maxiter, tol):
    pass

def damped_newton(f, df, x0, M, maxiter, tol):
    pass

def aufgabe4():
    pass