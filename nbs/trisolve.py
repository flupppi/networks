import numpy as np
from numpy import dtype

def trisolve(M: np.ndarray, b: np.array):
    """
    Takes a matrix M and a vector b and calculates the vector x using forward or backward substitution.
    Solves the linear system M*x = b.
    Here we use the dot product to calculate the product sum, which lets us skip the inner loop.

    :param M: input matrix (should be a lower or upper triangular matrix)
    :param b: input vector
    :return: tuple (x, info), where x is the solution vector, and info is a status flag
    """
    # %% Exercise 4.1
    print(f'Matrix M =\n{M}\nVector b = {b}')
    print(f'Solving for x in Equation Mx = b using', end=" ")
    info: int = 0  # Initialize info flag to 0 (indicates failure by default)

    try:
        n = len(M)  # Get the number of rows in M (assuming M is square)

        # Check if M is a lower triangular matrix
        if np.allclose(M, np.tril(M)):
            # np.tril(M) returns the lower triangular part of M (upper triangle set to zero)
            # np.allclose checks if M and np.tril(M) are approximately equal
            print('forward substitution')
            x = np.zeros_like(b, dtype=float)  # Initialize the solution vector x with zeros

            # Forward substitution algorithm
            for i in range(n):
                if M[i, i] == 0:
                    # Cannot divide by zero (diagonal element is zero)
                    raise ZeroDivisionError(f'M[{i},{i}] == 0')
                # Calculate the sum of M[i, :i] * x[:i]
                # This is the dot product of the i-th row of M up to column i (excluding M[i,i]) and the known x values
                sum_ = np.dot(M[i, :i], x[:i])
                # Compute x[i] using the formula for forward substitution
                x[i] = (b[i] - sum_) / M[i, i]
            info = 42  # Set info flag to 42 to indicate success

        # Check if M is an upper triangular matrix
        elif np.allclose(M, np.triu(M)):
            # np.triu(M) returns the upper triangular part of M (lower triangle set to zero)
            print('back substitution')
            x = np.zeros_like(b, dtype=float)  # Initialize the solution vector x with zeros

            # Back substitution algorithm
            for i in reversed(range(n)):
                if M[i, i] == 0:
                    # Cannot divide by zero (diagonal element is zero)
                    raise ZeroDivisionError(f'M[{i},{i}] == 0')
                # Calculate the sum of M[i, i+1:] * x[i+1:]
                # This is the dot product of the i-th row of M from column i+1 to end and the known x values
                sum_ = np.dot(M[i, i+1:], x[i+1:])
                # Compute x[i] using the formula for backward substitution
                x[i] = (b[i] - sum_) / M[i, i]
            info = 42  # Set info flag to 42 to indicate success

        else:
            # If M is neither lower nor upper triangular, we cannot solve using this method
            print("Error: Matrix M is not a triangular matrix")
            x = np.full_like(b, np.nan, dtype=float)  # Fill x with NaN values to indicate failure
            info = 0  # Set info flag to 0 to indicate failure

    except Exception as e:
        # Catch any exceptions that occur during the computation
        print(f'An unexpected error occurred: {e}')
        x = np.full_like(b, np.nan, dtype=float)  # Fill x with NaN values to indicate failure
        info = 0  # Set info flag to 0 to indicate failure

    return x, info  # Return the solution vector x and the info flag

def print_results(x, info):
    """
    Prints the results for the calculation.

    :param x: result vector
    :param info: result status flag
    :return: None
    """
    if info == 42:
        print('The algorithm terminated successfully.')
        print(f'Result for x: {x}\n')
    else:
        print(f'The algorithm had an unexpected error while solving, so it terminated with info = {info}\n')

if __name__ == '__main__':
    n = 5  # Size of the matrix and vectors

    # %% Exercise 4.2
    # Construct a lower triangular matrix A of size n x n
    # Each row i has elements 1/(i+1) in the lower triangle (including the diagonal)
    A = np.tril([np.full(n, 1/(i+1)) for i in range(n)])
    print(f'Matrix A:\n{A}')

    # Construct vectors b1 and b2
    b1 = np.full(n, 1.0)  # Vector b1 with all elements equal to 1.0
    b2 = np.array([((i + 1) + 1)/2 for i in range(n)])  # Vector b2 with elements ((i+1)+1)/2

    print(f'Vector b1: {b1}')
    print(f'Vector b2: {b2}')

    # Solve the system A x = b1
    x_, info_ = trisolve(A, b1)
    print_results(x_, info_)

    # Solve the system A x = b2
    x_, info_ = trisolve(A, b2)
    print_results(x_, info_)
