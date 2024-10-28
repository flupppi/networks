# mainfile.py
# Authors: Felix Kalchschmid
# Contact: felix@example.com

# Main file to execute and test functions for Numerische Mathematik exercises.

from myname import print_name
from plotsinus import plot_sine
from matrix_ops import genmatrix, fastmatrix, compare_matrices
from linsys_conditioning import solve_linear_systems, plot_conditioning_vs_error

#%% Main Execution Block
if __name__ == '__main__':
    # Task 1: Print name
    print_name("Felix Kalchschmid")

    # Task 2: Plot sinus function
    plot_sine(n=100)

    # Task 3: Testing matrix generation and comparison
    n, d, x = 5, 1, 2  # Example values for matrix generation
    matrix_slow = genmatrix(n, d, x)  # Matrix generated with loops
    matrix_fast = fastmatrix(n, d, x)  # Matrix generated without loops
    compare_matrices(n_values=[500, 1000, 1500, 2000, 2500, 3000], d=1, x=2)  # Runtime comparison

    # Task 4: Conditioning and relative error analysis
    plot_conditioning_vs_error()