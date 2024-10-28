# plotsinus.py
# Script to generate and plot sine function values based on a specified parameter.

import numpy as np
import matplotlib.pyplot as plt


# %% Plotting Sine Function
def plot_sine(n):
    """
    Plots the sine function based on a specified array of values.

    Input:
    - n (int): Number of values to generate in the range (0, 1) for x-axis.

    Output:
    - A plot of sin(2πx) over the range (0, 1).
    """
    # Generate an array a with values in (0, 1)
    a = np.linspace(1, n, n) / n
    b = np.sin(2 * np.pi * a)  # Array of sine values based on a

    print(f"Sum of array a: {np.sum(a)}")  # Display sum of array components

    # Plot the sine function with labels and title
    plt.plot(a, b, label="sin(x)")
    plt.xlabel("x values")
    plt.ylabel("sin(x)")
    plt.title("Sinus Plot")
    plt.legend()
    plt.show()
