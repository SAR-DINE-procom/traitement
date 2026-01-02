import numpy as np
import matplotlib.pyplot as plt

def calculate_entropy(image):
    """
    Calcule l'entropie de l'image basée sur l'histogramme des amplitudes.
    Une entropie plus faible indique généralement une image plus nette.
    """
    # Prendre l'amplitude de l'image complexe
    amplitude = np.abs(image)
    
    # Normaliser et quantifier pour créer un histogramme
    amplitude_norm = amplitude / np.max(amplitude)
    # Quantification sur 256 niveaux
    quantized = (amplitude_norm * 255).astype(int)
    
    # Calculer l'histogramme
    hist, _ = np.histogram(quantized, bins=256, range=(0, 255), density=True)
    
    # Supprimer les bins vides pour éviter log(0)
    hist = hist[hist > 0]
    
    # Calculer l'entropie: H = -sum(p * log2(p))
    entropy = -np.sum(hist * np.log2(hist))
    
    return entropy

def pga_autofocus(sar_image, iterations=5, window_width=None):
    """
    Implémentation de l'algorithme Phase Gradient Autofocus (PGA).
    
    Args:
        sar_image (ndarray): Image SAR complexe focalisée (mais floue).
                             Format : [Range, Azimut] (Lignes, Colonnes).
        iterations (int): Nombre d'itérations.
        window_width (int): Largeur de la fenêtre initiale. Si None, prend N/2.
        
    Returns:
        corrected_image (ndarray): Image nette.
        phase_error (ndarray): L'erreur de phase estimée finale.
    """
    img = sar_image.copy()
    N_range, N_az = img.shape
    
    # Phase d'erreur cumulée (pour le diagnostic)
    total_phase_error = np.zeros(N_az)
    
    if window_width is None:
        window_width = N_az // 2

    # Calculer l'entropie initiale
    initial_entropy = calculate_entropy(img)
    print(f"Entropie initiale: {initial_entropy:.4f}")

    for i in range(iterations):
        # 1. CENTER SHIFTING (Alignement des points brillants)
        # On trouve le max sur chaque ligne (Range)
        max_indices = np.argmax(np.abs(img), axis=1)
        
        # On décale circulairement chaque ligne pour mettre le max au centre
        shifted_img = np.zeros_like(img)
        for r in range(N_range):
            shift = (N_az // 2) - max_indices[r]
            shifted_img[r, :] = np.roll(img[r, :], shift)

        # 2. WINDOWING (Fenêtrage)
        # On réduit la fenêtre à chaque itération pour exclure le bruit
        # (Stratégie adaptative simple)
        current_width = int(window_width / (i + 1)) 
        center = N_az // 2
        start = max(0, center - current_width // 2)
        end = min(N_az, center + current_width // 2)
        
        window = np.zeros(N_az)
        window[start:end] = 1.0
        
        # Application de la fenêtre sur toutes les lignes
        windowed_img = shifted_img * window

        # 3. PASSAGE DOMAINE DONNÉES (FFT)
        # On passe dans le domaine fréquentiel (Doppler/Temps lent)
        G = np.fft.fft(windowed_img, axis=1)

        # 4. ESTIMATION DU GRADIENT (Produit conjugué + Somme)
        # Différence de phase entre k et k-1
        # G[:, 1:] * np.conj(G[:, :-1])
        # On somme sur l'axe Range (axis=0) pour moyenner le bruit
        numerator = np.sum(G[:, 1:] * np.conj(G[:, :-1]), axis=0)
        
        # On extrait la phase (le gradient)
        dphi = np.angle(numerator)
        
        # On remet le premier échantillon à 0 (pas de gradient au début)
        dphi = np.insert(dphi, 0, 0)

        # 5. INTÉGRATION (Retrouver l'erreur de phase)
        estimated_error = np.cumsum(dphi)
        
        # On enlève la tendance linéaire (qui correspondrait juste à un décalage global de l'image)
        # (Optionnel mais recommandé pour éviter que l'image ne sorte du cadre)
        trend = np.polyfit(np.arange(N_az), estimated_error, 1)
        estimated_error -= np.polyval(trend, np.arange(N_az))

        # Mise à jour de l'erreur totale
        total_phase_error += estimated_error

        # 6. CORRECTION DE L'IMAGE
        # On applique la correction dans le domaine fréquentiel de l'image ORIGINALE (non shiftée)
        # Attention : Il faut passer l'image originale en FFT d'abord
        IMG_original_freq = np.fft.fft(img, axis=1)
        
        # Correction : on multiplie par e^(-j * erreur)
        correction_phasor = np.exp(-1j * estimated_error)
        IMG_corrected = IMG_original_freq * correction_phasor
        
        # Retour domaine image
        img = np.fft.ifft(IMG_corrected, axis=1)

        # Calculer et afficher l'entropie après correction
        current_entropy = calculate_entropy(img)
        print(f"Itération {i+1}/{iterations}: entropie: {current_entropy:.4f}")

    return img, total_phase_error

# --- EXEMPLE D'UTILISATION (Simulation) ---
if __name__ == "__main__":
    # Création d'une image synthétique floue
    N = 256
    t = np.linspace(-10, 10, N)
    
    # Un point brillant au milieu
    true_image = np.zeros((100, N), dtype=complex)
    
    # Positions des points brillants (range, azimut)
    target_positions = [(50, 100), (20, 150)]  # Stocker les positions
    true_image[50, 100] = 10 + 0j # Cible forte
    true_image[20, 150] = 5 + 0j  # Cible moyenne
    
    # Ajout d'une erreur de phase (Quadratique + Sinusoïdale)
    phase_err = 5 * np.sin(0.1 * np.arange(N)) + 0.005 * (np.arange(N) - N/2)**2
    
    # Application du flou (Produit de convolution <=> Produit en fréq)
    IMG = np.fft.fft(true_image, axis=1)
    IMG_blurred = IMG * np.exp(1j * phase_err)
    blurred_image = np.fft.ifft(IMG_blurred, axis=1)
    
    # Lancement du PGA
    corrected_image, est_err = pga_autofocus(blurred_image, iterations=11)
    
    # Affichage
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(np.abs(blurred_image), aspect='auto', cmap='gray')
    plt.title("Image Floue (Entrée)")
    
    plt.subplot(2, 2, 2)
    plt.imshow(np.abs(corrected_image), aspect='auto', cmap='gray')
    # Ajouter les croix sur les positions réelles des points brillants
    for pos in target_positions:
        plt.plot(pos[1], pos[0], 'r+', markersize=15, markeredgewidth=3, label='Position réelle' if pos == target_positions[0] else "")
    plt.title("Image Corrigée (Sortie PGA)")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(phase_err, label="Erreur réelle")
    plt.plot(est_err, '--', label="Erreur estimée par PGA")
    plt.title("Comparaison des erreurs de phase")
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()