import numpy as np
import cv2
from matplotlib import pyplot as plt


def aufgabe_1():
    # Define Variables
    A = np.array([[1, 0],[0, 1],[0,0]])
    b = np.array([0.01, 0, 1])
    delta_b = np.array([0.0001, 0, 0])
    b_permuted = b + delta_b

    #%% 1.1 Calculate Solution for Ausgleichsproblem using Normal Equation

    # Is the Solution Unique
    # Why?

    #%% 1.2 Calculate the sensitivity of the calculation


def aufgabe_2():
    # %% 2.1
    # Define Variables
    A = 1/27 * np.array([[ 16,  52,  80],
                         [ 44,  80, -32],
                         [ -9, -36, -72],
                         [-16, -16,  64]])
    print(f'A: \n{A}')
    b = np.array([4, 1, 3, 0])
    print(f'b:\n{b}')
    U_ref = 1/45 * np.array([[ 32,   5, 24, -20],
                             [  4,  40,  3,  20],
                             [-27,   0, 36,   0],
                             [ 16, -20, 12,  35]])
    print(f'U_ref:\n{U_ref}')
    Sigma_ref = np.array([[5, 0, 0], [0, 4, 0], [0, 0, 0], [0, 0, 0]])
    print(f'Sigma_ref:\n{Sigma_ref}')
    V_T_ref = 1/9 * np.array([[ 1, 4,  8],
                              [ 4, 7, -4],
                              [-8, 4, -1]])
    print(f'V_T_ref:\n{V_T_ref}')
    U,S,V_T = np.linalg.svd(A)
    Sigma = np.array([[S[0], 0, 0], [0, S[1], 0], [0, 0, S[2]], [0, 0, 0]])
    print(f'U:\n{U}')
    print(f'Sigma:\n{Sigma}')
    print(f'V_T:\n{V_T}')
    print(np.allclose(U_ref, U))
    print(np.allclose(Sigma_ref, Sigma))
    print(np.allclose(V_T_ref, V_T))

    A_ref = U_ref @ Sigma_ref @ V_T_ref
    print(f'A_ref:\n{A_ref}')
    print(np.allclose(A_ref, A))

    A_svd = U @ Sigma @ V_T
    print(f'A_svd:\n{A_svd}')
    print(np.allclose(A_svd, A))

    # Berechnung der Pseudoinversen von A
    Sigma_plus = np.zeros((3, 4))
    for i in range(len(S)):
        if S[i] > 1e-10: # Nur nicht-null Singulärwerte invertieren
            Sigma_plus[i, i] = 1/S[i]

    print(f'Sigma_plus:\n{Sigma_plus}')
    A_plus = V_T_ref.T @ Sigma_plus @ U_ref.T
    print(f'Pseudoinverse von A (A^+):\n{A_plus}')

    #%% 2.1 Lösung für Ausgleichsproblem mit minimaler euklidischer norm bestimmen
    x_star = A_plus @ b
    print(f'Lösung mit minimaler euklidischer Norm (x^*):\n{x_star}')

    # %% 2.2 Lösungsmenge L(b) des Ausgleichsproblems bestimmen.
    # Basis des Nullraums von A
    nullraum_basis = V_T.T[:, 2:] # Nullraum entspricht den letzten Spalten von V_T
    print(f"Basis des Nullraums von A:\n{nullraum_basis}")

    # Lösungsmenge L(b)
    print(f"L(b) = {{x^* + z | z in Nullraum von A}}")

    print(A @ nullraum_basis)
    return x_star, nullraum_basis




def compress_image(image, k):
    """
    Komprimiert ein Bild mit der Singulärwertzerlegung (SVD).

    :param image: Pfad zum Bild.
    :param k: Anzahl der verwendeten Singulärwerte.
    """
    # 1. Laden des Bildes und Anzeige mit Matplotlib
    img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
    if img is None:
        print("Error: Image not found. Please check the file path.")
        return
    cv2.imshow("Input Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 2. Farbraumaufteilung
    r, g, b = cv2.split(img)

    def compress_channel(m, name):
        """
        Komprimiert einen Farbkanal mit der k-trunkierten Singulärwertzerlegung.

        :param m: Eingabematrix für den Farbkanal.
        :param name: Name des Kanals (für Debugging).
        :return: Komprimierte Matrix.
        """
        # 3. Skalieren und Zentrieren
        m_scaled = m / 255.0
        mean_m = np.mean(m_scaled, axis=0)
        m_centered = m_scaled - mean_m

        # 4. Singulärwertzerlegung
        U, Sigma, V_T = np.linalg.svd(m_centered, full_matrices=False)

        # 5. Trunkierung auf k Singulärwerte
        U_k = U[:, :k]
        Sigma_k = np.diag(Sigma[:k])
        V_T_k = V_T[:k, :]

        # Rekonstruktion der k-trunkierten Matrix
        A_k = U_k @ Sigma_k @ V_T_k

        # 7. Rückgängigmachen der Zentrierung
        m_result = A_k + mean_m
        return m_result

    # Komprimierung der Farbkanäle
    reduced_r = compress_channel(r, "Red")
    reduced_g = compress_channel(g, "Green")
    reduced_b = compress_channel(b, "Blue")

    # 8. Zusammenfügen der komprimierten Farbkanäle
    reduced_img = cv2.merge((reduced_r, reduced_g, reduced_b))

    # 9. Anzeige des Original- und komprimierten Bildes
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Compressed Image (k={k})")
    plt.imshow(reduced_img)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("comparison.png")
    plt.show()
    return img, reduced_img


if __name__ == '__main__':
    # Blatt 5 - Aufgabe 4
    original, compressed = compress_image("unidach.jpg", 20)
    x_star, nullraum_basis = aufgabe_2()
