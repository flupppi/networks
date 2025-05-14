

import numpy as np


def trisolve(M, b):
    print(f'Matrix M =\n{M}\nVector b = {b}')
    print(f'Solving for x in Equation Mx = b using forward-/backward substitution')
    try:
        n = M.shape[0]
        x = np.zeros_like(b, dtype=float)

        if np.allclose(M, np.tril(M)):
            # M ist eine untere Dreiecksmatrix
            for i in range(n):
                if M[i, i] == 0:
                    raise ZeroDivisionError(f'M[{i},{i}] == 0')
                sum_ = 0.0
                for j in range(i):
                    sum_ += M[i, j] * x[j]
                x[i] = (b[i] - sum_) / M[i, i]
            info = 42

        elif np.allclose(M, np.triu(M)):
            # M ist eine obere Dreiecksmatrix
            for i in reversed(range(n)):
                if M[i, i] == 0:
                    raise ZeroDivisionError(f'M[{i},{i}] == 0')
                sum_ = 0.0
                for j in range(i + 1, n):
                    sum_ += M[i, j] * x[j]
                x[i] = (b[i] - sum_) / M[i, i]
            info = 42

        else:
            # M ist nicht dreiecksförmig
            print('Error: M is not a triangular matrix')
            x = np.full_like(b, np.nan, dtype=float)
            info = 0

    except Exception as e:
        print(f'An unexpected error occurred: {e}')
        x = np.full_like(b, np.nan, dtype=float)
        info = 0

    return x, info


def print_results(x, info):
    if info == 42:
        print('The algorithm terminated successfully.')
        print(f'Result for x: {x}\n')
    else:
        print(f'The algorithm had an unexpected error while solving, so it terminated with info = {info}\n')

def trisolve_dot(M, b):
    print(f'Matrix M =\n{M}\nVector b = {b}')
    print(f'Solving for x in Equation Mx = b using forward-/backward substitution')
    try:
        if np.allclose(M, np.tril(M)):
            # M is lower triangular
            x = np.zeros_like(b, dtype=float)
            n = M.shape[0]
            for i in range(n):
                if M[i, i] == 0:
                    raise ZeroDivisionError(f'M[{i},{i}] == 0')
                x[i] = (b[i] - np.dot(M[i, :i], x[:i])) / M[i, i]
            info = 42

        elif np.allclose(M, np.triu(M)):
            # M is upper triangular
            x = np.zeros_like(b, dtype=float)
            n = M.shape[0]
            for i in reversed(range(n)):
                if M[i, i] == 0:
                    raise ZeroDivisionError(f'M[{i},{i}] == 0')
                x[i] = (b[i] - np.dot(M[i, i+1:], x[i+1:])) / M[i, i]
            info = 42

        else:
            # M is not triangular
            print('Error: M is not a triangular matrix')
            x = np.full_like(b, np.nan, dtype=float)
            info = 0

    except Exception as e:
        print(f'An unexpected error occurred: {e}')
        x = np.full_like(b, np.nan, dtype=float)
        info = 0

    return x, info



if __name__ == '__main__':
    b1 = np.array([1, 1, 1, 1], dtype=float)
    b2 = np.array([1, 3 / 2, 2], dtype=float)

    # Beispiel für eine untere Dreiecksmatrix
    M1 = np.array([[1, 0, 0],
                   [2, 1, 0],
                   [3, 4, 1]], dtype=float)

    # Beispiel für eine obere Dreiecksmatrix
    M2 = np.array([[1, 2, 3],
                   [0, 1, 4],
                   [0, 0, 1]], dtype=float)

    (x_, info_) = trisolve(M1, b2)
    print_results(x_, info_)

    (x_, info_) = trisolve(M2, b2)
    print_results(x_, info_)


