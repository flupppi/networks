import numpy as np

# Gegebene Daten
r = np.array([10, 5, 2.5, 1.3, 1])  # in LE
cos_phi = np.array([0.63, 0.39, 0.12, -0.31, -0.59])

# Matrizendarstellung A und b
A = np.column_stack((np.ones(len(cos_phi)), cos_phi))
b = 1 / r

# Lösung des Minimierungsproblems mit Normalengleichung
x = np.linalg.inv(A.T @ A) @ A.T @ b

# Ergebnisse
x1, x2 = x
p = 1 / x1
epsilon = x2 / x1

print(f"Halbparameter p: {p:.4f}")
print(f"Numerische Exzentrizität ε: {epsilon:.4f}")

# Typ der Kometenbahn bestimmen
if epsilon < 1:
    typ = "Ellipse"
elif epsilon == 1:
    typ = "Parabel"
else:
    typ = "Hyperbel"

print(f"Typ der Kometenbahn: {typ}")




