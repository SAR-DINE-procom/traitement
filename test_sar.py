import numpy as np
import matplotlib.pyplot as plt
from processing.backprojection import Backprojection

def test_image_formation():
    print("--- Démarrage du test de formation d'image SAR ---")

    # 1. Paramètres de Simulation
    f_c = 24e9          # Fréquence porteuse (24 GHz)
    c = 3e8             # Vitesse lumière
    B = 200e6           # Bande passante (200 MHz) -> Résolution ~ 75cm
    T_chirp = 1e-6      # Durée du chirp (1 µs)
    fs = 500e6          # Fréquence d'échantillonnage (500 MHz)
    
    # Géométrie d'acquisition (Vol rectiligne)
    v_plat = 50.0       # Vitesse plateforme (m/s)
    H_plat = 100.0      # Altitude (m)
    L_synth = 100.0     # Longueur de l'ouverture synthétique (m)
    
    # Position de la cible (au sol)
    target_pos = np.array([0, 50, 0]) # X=0 (centre azimut), Y=50 (Range au sol), Z=0

    # 2. Instanciation
    bp = Backprojection(fs, T_chirp, B)
    bp.build_chirp()
    bp.build_window()
    
    # 3. Génération de la trajectoire (Slow Time)
    # Le radar se déplace le long de l'axe X à l'altitude H
    slow_time = np.linspace(-L_synth/(2*v_plat), L_synth/(2*v_plat), 101) # 101 pulses
    N_pulses = len(slow_time)
    
    M_pos = np.zeros((3, N_pulses))
    M_pos[0, :] = v_plat * slow_time  # X varie
    M_pos[1, :] = 0                   # Y = 0 (trajectoire au-dessus de l'axe X)
    M_pos[2, :] = H_plat              # Z = H
    
    print(f"Simulation de {N_pulses} impulsions...")

    # 4. Simulation des données brutes (Raw Data)
    # On calcule le retard pour chaque pulse et on place le chirp
    N_fast_time = int(2 * (H_plat + 100) / c * fs) # Assez long pour capter l'écho
    raw_data = np.zeros((N_fast_time, N_pulses), dtype=complex)
    
    for i in range(N_pulses):
        radar_pos = M_pos[:, i]
        dist = np.linalg.norm(radar_pos - target_pos)
        tau = 2 * dist / c # Retard aller-retour
        
        # Calcul du déphasage dû à la propagation (Terme de phase constant par pulse)
        phase_shift = np.exp(-1j * 4 * np.pi * f_c * dist / c)
        
        # Insertion du chirp retardé dans le vecteur Fast Time
        start_idx = int(tau * fs)
        end_idx = start_idx + len(bp.symbol)
        
        if end_idx < N_fast_time:
            # On ajoute le chirp modulé par la phase de propagation
            raw_data[start_idx:end_idx, i] += bp.symbol * phase_shift

    # 5. Compression en Distance (Range Compression)
    print("Compression en distance...")
    # Note: Votre fonction range_compression attend une matrice et fait une FFT sur l'axe 0
    # Il faut s'assurer que raw_data est bien dimensionné.
    # Votre matched_filtering actuel fait ifft(fft(sig) * conj(fft(ref))).
    # Pour que ça marche bien, il faut souvent du zero-padding, mais testons tel quel.
    # Ici on va tricher un peu pour adapter la taille si besoin, ou utiliser une convolution directe
    # pour être sûr du résultat physique dans ce test simplifié.
    
    M_rc = np.zeros_like(raw_data)
    ref_chirp = bp.symbol * bp.window
    
    # On fait une convolution "valid" ou "same" colonne par colonne
    for i in range(N_pulses):
        # Corrélation (Filtrage adapté)
        col_compressed = np.correlate(raw_data[:, i], ref_chirp, mode='same')
        M_rc[:, i] = col_compressed

    # Visualisation des données compressées (Range vs Azimuth)
    plt.figure(figsize=(10, 4))
    plt.imshow(np.abs(M_rc), aspect='auto', cmap='jet', extent=[M_pos[0,0], M_pos[0,-1], N_fast_time/fs*c/2, 0])
    plt.title("Données Compressées en Distance (Range Migration visible)")
    plt.xlabel("Position Azimut (m)")
    plt.ylabel("Distance (m)")
    plt.colorbar()
    plt.show()

    # 6. Formation d'Image (Backprojection)
    print("Formation de l'image (Backprojection)...")
    
    # Définition de la grille de reconstruction autour de la cible
    # Cible en (0, 50). On fait une grille de +/- 10m autour.
    axe_X = np.linspace(-10, 10, 100) # Cross-range
    axe_Y = np.linspace(40, 60, 100)  # Range (Ground range)
    
    # Création de la grille interne de l'objet (nécessaire pour votre méthode image_formation)
    bp.create_grid(xlim=[-10, 10], ylim=[40, 60], steps=[axe_X[1]-axe_X[0], axe_Y[1]-axe_Y[0]])
    
    # Appel de la fonction
    image = bp.image_formation(M_rc, M_pos, axe_X, axe_Y, f_c, fs)
    
    # 7. Visualisation Finale
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(axe_X, axe_Y, np.abs(image), shading='auto', cmap='inferno')
    plt.plot(target_pos[0], target_pos[1], 'rx', markersize=10, label='Vraie Cible')
    plt.title("Image SAR Reconstruite")
    plt.xlabel("Cross-Range X (m)")
    plt.ylabel("Ground Range Y (m)")
    plt.legend()
    plt.colorbar(label='Amplitude')
    plt.axis('equal')
    plt.show()


if __name__ == '__main__': 
    test_image_formation()
