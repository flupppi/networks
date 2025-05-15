import numpy as np
import pandas as pd
import jinja2


def f(x):
    return np.sin(x)    # f(x) = sin(x)

def f_prime(x):
    return np.cos(x)    # f'(x) = cos(x)
def newtons_method(x0, f, f_prime, z, tolerance, epsilon, max_iterations):
    """Newton's method with explicit root proximity check and iteration logging."""
    iterations = []  # To collect data for each iteration
    for iteration in range(max_iterations):
        y = f(x0)
        yprime = f_prime(x0)

        if abs(yprime) < epsilon:  # Avoid division by very small numbers
            print(f"Derivative too small after {iteration} iterations.")
            return None, iterations

        x1 = x0 - y / yprime  # Newton's update
        error = abs(x1 - z)  # Absolute error with respect to the root z

        # Log iteration data
        iterations.append({'Iteration': iteration + 1, 'x_k': x1, 'Error': error})

        # Check if the result is close enough to z
        if error < tolerance:
            return x1, iterations  # Success: x1 is close to z within tolerance

        x0 = x1  # Update for the next iteration

    print("Newton's method did not converge within the maximum iterations.")
    return None, iterations

def bisection_method(f, a, b, z, tolerance, max_iterations):
    """Bisection method with explicit root proximity check and iteration logging."""
    iterations = []  # To collect data for each iteration
    if f(a) * f(b) >= 0:
        print("The function must have opposite signs at the interval endpoints.")
        return None, iterations

    for iteration in range(max_iterations):
        midpoint = (a + b) / 2
        error = abs(midpoint - z)  # Absolute error with respect to the root z

        # Log iteration data
        iterations.append({'Iteration': iteration + 1, 'Midpoint': midpoint, 'Error': error})

        if error < tolerance:  # Check proximity to z
            return midpoint, iterations  # Success: midpoint is close to z within tolerance

        if f(a) * f(midpoint) < 0:
            b = midpoint  # The root is in the left half
        else:
            a = midpoint  # The root is in the right half

    print("Bisection method did not converge within the maximum iterations.")
    return None, iterations


if __name__ == '__main__':
    interval = [2, 4]
    z = np.pi
    x_0 = 4
    error = 0.0001
    epsilon = 1e-10       # A small value to avoid division by very small numbers
    max_iterations = 100  # Maximum number of iterations

    #%% Newton's Method
    result_newton, newton_iterations = newtons_method(x_0, f, f_prime, z, error, epsilon, max_iterations)
    if result_newton is not None:
        print(f"Newton's method converged to {result_newton:.6f}, which is close to pi.")
    else:
        print("Newton's method did not converge.")
    df_newton = pd.DataFrame(newton_iterations)
    print("\nNewton's Method Iterations:")
    print(df_newton.to_latex())

    #%% Bisection Method
    result_bisection, bisection_iterations = bisection_method(f, interval[0], interval[1], z, error, max_iterations)
    if result_bisection is not None:
        print(f"Bisection method converged to {result_bisection:.6f}, which is close to pi.")
    else:
        print("Bisection method did not converge.")
    df_bisection = pd.DataFrame(bisection_iterations)
    print("\nBisection Method Iterations:")
    print(df_bisection.to_latex())


