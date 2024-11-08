import numpy as np
from numpy import dtype


def trisolve(M: np.ndarray, b: np.array):
    # %% Exercise 4.1

    print(f'Matrix M = {M}\nVector b = {b}')
    print(f'Solving for x in Equation Mx = b using', end=" ")
    info: int = 0
    try:
        n = len(M)
        if np.allclose(M, np.tril(M)):# checks if each element of a matrix is close (equal) to another matrix it is compared to
            print('foreward substitution')

            x = np.zeros_like(b, dtype=float)
            # np.tril: returns a copy of the matrix with all values in the upper triangle zeroed
            M_l = M.copy()
            for i in range(n):
                if M_l[i, i] == 0:
                    raise ZeroDivisionError(f'M[{i},{i}] == 0')
                # sum_ = 0.0
                # for k in range(i): # slower version using two loops
                #     sum_ += M_l[i, k] * x[k]
                #x[i] = (b[i] - sum_) / M_l[i, i]
                x[i] = (b[i] - np.dot(M[i, :i], x[:i])) / M[i, i]
            info = 42
        elif np.allclose(M, np.triu(M)):
            print('back substitution')

            x = np.zeros_like(b, dtype=float)
            # np.tril: returns a copy of the matrix with all values in the lower triangle zeroed
            M_u = M.copy
            for i in reversed(range(n)):
                if M_u[i, i] == 0:
                    raise ZeroDivisionError(f'M[{i},{i}] == 0')
                #sum_ = 0.0
                #for k in range(i+1, n): # slower version using two loops
                #    sum_ += M_u[i, k] * b[k]
                #x[i] = (b[i] - sum_) / M_u[i, i]
                x[i] = (b[i] - np.dot(M[i, i+1:], x[i+1:])) / M[i, i]
                print(M[i, i+1:])
                print(b[i+1:])

            info = 42

        else:
            print("Error: Matrix M is not a triangular matrix")
            x = np.full_like(M, np.nan, dtype=float)
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
        print(f'The algorithm had an unexpected error while solving, so it terminated with {info}\n')

if __name__ == '__main__':
    n=40
    # %% Exercise 4.2
    A = np.tril([np.full(n, 1/(i+1)) for i in range(n)])
    print(A)
    b1 = np.full (n, 1.0)
    b2 = np.array([((i + 1) + 1)/2 for i in range(n)])
    print(b1)
    print(b2)

    (x_, info_) = trisolve(A, b1)
    print_results(x_, info_)

    (x_, info_) = trisolve(A, b2)
    print_results(x_, info_)