import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft, fftshift, ifftshift

def pga_autofocus(sar_image, iterations=5, window_width=None):
    """
    Implémentation de l'algorithme Phase Gradient Autofocus (PGA).
    
    Args:
        sar_image (ndarray): Image SAR complexe (Range x Azimut).
        iterations (int): Nombre d'itérations pour affiner la phase.
        window_width (int): Largeur initiale de la fenêtre de fenêtrage.
    
    Returns:
        corrected_image (ndarray): Image SAR corrigée.
        total_phase_error (ndarray): L'erreur de phase estimée cumulée.
    """
    num_range, num_azimuth = sar_image.shape
    corrected_image = sar_image.copy().astype(np.complex128)
    total_phase_error = np.zeros(num_azimuth)

    if window_width is None:
        window_width = num_azimuth // 2

    for i in range(iterations):
        # 1. Sélection du point le plus brillant (Circular Shifting)
        # On aligne le pixel le plus fort de chaque ligne sur le centre en azimut
        max_indices = np.argmax(np.abs(corrected_image), axis=1)
        shifted_image = np.zeros_like(corrected_image)
        shift_amounts = num_azimuth // 2 - max_indices
        
        for r in range(num_range):
            shifted_image[r, :] = np.roll(corrected_image[r, :], shift_amounts[r])

        # 2. Fenêtrage (Windowing)
        # Réduit l'influence du bruit et des cibles secondaires
        current_window = max(window_width // (2**i), 10)  # Réduction progressive
        win = np.zeros(num_azimuth)
        start, end = (num_azimuth//2 - current_window//2), (num_azimuth//2 + current_window//2)
        win[start:end] = 1
        windowed_image = shifted_image * win

        # 3. Passage dans le domaine des fréquences spatiales (Phase Gradient Estimation)
        # L'erreur de phase se trouve dans le domaine de la compression azimut
        g_n = ifft(ifftshift(windowed_image, axes=1), axis=1)
        
        # Calcul du gradient de phase (Estimateur du Maximum de Vraisemblance)
        # On utilise la dérivée de la phase : Delta_phi = angle(g_n[n] * conj(g_n[n-1]))
        g_n_diff = np.diff(g_n, axis=1, append=g_n[:, :1])
        phase_gradient = np.imag(np.sum(np.conj(g_n) * g_n_diff, axis=0) / 
                                 np.sum(np.abs(g_n)**2, axis=0))

        # 4. Intégration et suppression des termes linéaires (Removes bias)
        phase_error = np.cumsum(phase_gradient)
        # On retire la rampe linéaire (shift de l'image) et l'offset moyen
        poly = np.polyfit(np.arange(num_azimuth), phase_error, 1)
        phase_error -= np.polyval(poly, np.arange(num_azimuth))
        
        total_phase_error += phase_error

        # 5. Correction de l'image
        # On applique la correction dans le domaine fréquentiel (Azimuth Doppler)
        image_freq = ifft(corrected_image, axis=1)
        correction_term = np.exp(-1j * phase_error)
        corrected_image = fft(image_freq * correction_term, axis=1)

    return corrected_image, total_phase_error
# if __name__ == "__main__":
#     # Création d'une image synthétique floue
#     N = 256
#     t = np.linspace(-10, 10, N)
    
#     # Un point brillant au milieu
#     true_image = np.zeros((100, N), dtype=complex)
    
#     # Positions des points brillants (range, azimut)
#     target_positions = [(50, 100), (20, 150)]  # Stocker les positions
#     true_image[50, 100] = 10 + 0j # Cible forte
#     true_image[20, 150] = 5 + 0j  # Cible moyenne
    
#     # Ajout d'une erreur de phase (Quadratique + Sinusoïdale)
#     phase_err = 5 * np.sin(0.1 * np.arange(N)) + 0.005 * (np.arange(N) - N/2)**2
    
#     # Application du flou (Produit de convolution <=> Produit en fréq)
#     IMG = np.fft.fft(true_image, axis=1)
#     IMG_blurred = IMG * np.exp(1j * phase_err)
#     blurred_image = np.fft.ifft(IMG_blurred, axis=1)
    
#     # Lancement du PGA
#     corrected_image, est_err = pga_autofocus(blurred_image, iterations=5)
    
#     # Affichage
#     plt.figure(figsize=(10, 8))
    
#     plt.subplot(2, 2, 1)
#     plt.imshow(np.abs(blurred_image), aspect='auto', cmap='gray')
#     plt.title("Image Floue (Entrée)")
    
#     plt.subplot(2, 2, 2)
#     plt.imshow(np.abs(corrected_image), aspect='auto', cmap='gray')
#     # Ajouter les croix sur les positions réelles des points brillants
#     for pos in target_positions:
#         plt.plot(pos[1], pos[0], 'r+', markersize=15, markeredgewidth=3, label='Position réelle' if pos == target_positions[0] else "")
#     plt.title("Image Corrigée (Sortie PGA)")
#     plt.legend()
    
#     plt.subplot(2, 1, 2)
#     plt.plot(phase_err, label="Erreur réelle")
#     plt.plot(est_err, '--', label="Erreur estimée par PGA")
#     plt.title("Comparaison des erreurs de phase")
#     plt.legend()
#     plt.grid()
    
#     plt.tight_layout()
#     plt.show()


import matplotlib.pyplot as plt

def main():
    # 1. Paramètres de simulation
    n_range = 128
    n_azimuth = 512
    image_pure = np.zeros((n_range, n_azimuth), dtype=np.complex128)
    
    # Ajouter des cibles ponctuelles (points brillants)
    targets = [(32, 256), (64, 128), (64, 384), (96, 256)]
    for r, a in targets:
        image_pure[r, a] = 100.0  # Impulsion "parfaite"

    # 2. Simuler l'étalement de la réponse impulsionnelle (PSF) en Azimut
    # Dans un vrai radar, cela vient de la compression de l'impulsion
    sar_blurred = fft(image_pure, axis=1) # Passage dans le domaine Doppler
    
    # 3. Générer une erreur de phase complexe (Mouvements de la plateforme)
    # On simule une combinaison de sinus et de polynômes pour mimer le roulis/tangage
    t = np.linspace(-1, 1, n_azimuth)
    phase_error = 5.0 * t**2 #+ 2.0 * t**3 + 1.5 * np.sin(2 * np.pi * 3 * t)
    error_signal = np.exp(1j * phase_error)
    
    # Appliquer l'erreur dans le domaine Doppler
    sar_blurred = sar_blurred * error_signal
    
    # Retour dans le domaine spatial (Image floue)
    sar_blurred_spatial = ifft(sar_blurred, axis=1)

    # 4. Exécuter le PGA
    corrected_image, estimated_phase = pga_autofocus(sar_blurred_spatial, iterations=10)

    # 5. Visualisation
    plt.figure(figsize=(12, 12))

    plt.subplot(3, 2, 1)
    plt.title("Image Originale (Théorique)")
    plt.imshow(np.abs(image_pure), cmap='gray', aspect='auto')

    plt.subplot(3, 2, 2)
    plt.title("Comparaison des Phases")
    plt.plot(phase_error - np.mean(phase_error), label="Erreur Réelle", linestyle='--')
    plt.plot(estimated_phase, label="Estimation PGA")
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.title("Image Dégradée (Erreurs de mouvement)")
    plt.imshow(np.abs(sar_blurred_spatial), cmap='gray', aspect='auto')

    plt.subplot(3, 2, 4)
    plt.title("Phase Image Dégradée")
    plt.imshow(np.angle(sar_blurred_spatial), cmap='jet', aspect='auto')
    plt.colorbar()

    plt.subplot(3, 2, 5)
    plt.title("Image Corrigée (PGA)")
    plt.imshow(np.abs(corrected_image), cmap='gray', aspect='auto')

    plt.subplot(3, 2, 6)
    plt.title("Phase Image Corrigée")
    plt.imshow(np.angle(corrected_image), cmap='jet', aspect='auto')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    main()