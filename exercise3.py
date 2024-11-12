import numpy as np


class SingularMatrixError(Exception):
    """Exception raised when the matrix is singular and cannot be solved."""
    pass


def solve(M, b) -> np.ndarray:
    print(f'Trying to solve Linear System Mx = b with M = \n{M}\n and b = {b}.')
    if np.linalg.det(M) == 0:
        raise SingularMatrixError("The matrix is singular (determinant is zero) and the equation cannot be solved.")
    else:
        result = np.linalg.solve(M, b)
        print(f'Result = {result}\n')
        return result


if __name__ == "__main__":
    # %% A4.1 a)
    A1 = np.array([[2, -1, -3, 3],
                   [4, 0, -3, 1],
                   [6, 1, -1, 6],
                   [-2, -5, 4, 1]])
    b1 = np.array([1, -8, -16, -12])

    try:
        solve(A1, b1)
    except SingularMatrixError as e:
        print(e)

    # %% A4.1 b)
    A2 = np.array([[1, 0, 6, 2],
                   [8, 0, -2, -2],
                   [2, 9, 1, 3],
                   [2, 1, -3, 10]])
    b2 = np.array([6, -2, -8, -4])

    try:
        solve(A2, b2)
    except SingularMatrixError as e:
        print(e)


    # %% A4.1 Test case with determinant equal to zero

    M_singular = np.array([
        [1, 2, 3],
        [2, 4, 6],
        [3, 6, 9]
    ])
    b_singular = np.array([1, 2, 3])


    try:
        solve(M_singular, b_singular)
    except SingularMatrixError as e:
        print(e)
