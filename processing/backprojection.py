import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import chirp
from scipy.signal.windows import hann
from scipy import constants
from scipy.ndimage import map_coordinates

class Backprojection: 
    def __init__(self, fs_hz, T_chirp_s, B_hz):
        self.fs_hz = fs_hz 
        self.T_chirp_s = T_chirp_s 
        self.B_hz = B_hz 
        self.c_m_s =  constants.speed_of_light

        pass 

    def build_chirp(self): 
        self.N = int(self.fs_hz * self.T_chirp_s )
        self.t_s = np.linspace(0, self.T_chirp_s / 2, self.N // 2, endpoint=False)
        ramp_up = chirp(
            self.t_s,
            f0=0,
            t1=self.T_chirp_s / 2,
            f1=self.B_hz,
            method='linear'
        )
        ramp_down = chirp(
            self.t_s,
            f0=self.B_hz,
            t1=self.T_chirp_s / 2,
            f1=0,
            method='linear'
        )
        self.symbol = np.concatenate((ramp_up, ramp_down))
    def build_window(self): 
        self.window = hann(self.N)

    
    def matched_filtering(self, sig): 
        reference = self.window * self.symbol 
        reference_ft = np.fft.fft(reference)
        sig_ft = np.fft.fft(sig)
        corr_ft = sig_ft * np.conjugate(reference_ft)
        return np.fft.ifft(corr_ft)

    def range_compression(self, mat): 
        mat_ft = np.fft.fft(mat, axis=0)
        mat_compressed = np.apply_along_axis(
            self.matched_filtering, 
            axis=0, 
            arr=mat_ft, 
            reference_ft=self.symbol
        )
        return mat_compressed
    
    def create_grid(self, xlim, ylim, steps): 
        eps = steps[0] / 10
        x = np.arange(xlim[0], xlim[1] + eps, steps[0])
        y = np.arange(ylim[0], ylim[1] + eps, steps[1])
        self.grid_pos_x, self.grid_pos_y = np.meshgrid(x, y) 
        self.grid_xy = np.stack([self.grid_pos_x, self.grid_pos_y], axis=-1)
    
    def image_formation(self, M_rc, M_pos, axe_X, axe_Y, f_p, f_s): 
        Ny = len(axe_Y)          # Nombre de pixels en Range (Y)
        Nx = len(axe_X)          # Nombre de pixels en Cross-range (X)
        
        N_samples, N_pulses = M_rc.shape 

        k = (4 * np.pi * f_p) / self.c_m_s
        

        M_pos_exp = M_pos[np.newaxis, np.newaxis, :, :]
        

        if self.grid_xy.shape[-1] == 2:
            zeros_z = np.zeros((Ny, Nx, 1))
            grid_xyz = np.concatenate([self.grid_xy, zeros_z], axis=-1)
        else:
            grid_xyz = self.grid_xy

        Grille_xyz_exp = grid_xyz[:, :, :, np.newaxis]
        
        Diff_coords_all = Grille_xyz_exp - M_pos_exp
        M_dist = np.linalg.norm(Diff_coords_all, axis=2) # Résultat : (Ny, Nx, N_pulses)

        M_delay = 2 * M_dist / self.c_m_s 

        M_indices = M_delay * f_s

        M_phase = np.exp(1j * k * M_dist)

        # interpolation
        idx_samples = M_indices.ravel()

        k_indices_grid = np.arange(N_pulses)
        k_indices_grid = np.broadcast_to(k_indices_grid[np.newaxis, np.newaxis, :], (Ny, Nx, N_pulses))
        idx_pulses = k_indices_grid.ravel()

        coords = np.stack([idx_samples, idx_pulses], axis=0)

        
        M_inter_flat = map_coordinates(
            input=M_rc, # Pas besoin de transpose si M_rc est (Samples, Pulses) et coords est [Samples, Pulses]
            coordinates=coords,
            order=3,
            mode='constant',
            cval=0.0
        )

        M_inter = M_inter_flat.reshape(Ny, Nx, N_pulses)
        
        M_corrected = M_inter * M_phase
        
        image = np.sum(M_corrected, axis=2)
        
        return image
    
    def plot_window(self): 
        plt.plot(self.window)
        plt.show()
    def plot_chirp(self): 
        if self.symbol is None:
            print("Erreur: Le chirp n'a pas encore été généré.")
            return

        # Reconstruction du vecteur temps complet pour l'affichage
        N = len(self.symbol)
        t_full = np.linspace(0, self.T_chirp_s, N, endpoint=False)

        plt.figure(figsize=(12, 10))

        # 1. Amplitude (Time Domain)
        plt.subplot(3, 1, 1)
        plt.plot(t_full * 1e6, self.symbol)
        plt.title(f"Chirp Triangulaire (Time Domain) - B={self.B_hz/1e6} MHz")
        plt.xlabel("Temps (µs)")
        plt.ylabel("Amplitude")
        plt.grid(True)

        # 2. Phase (Time Domain)
        # On utilise unwrap pour éviter les sauts de 2pi et voir la continuité
        plt.subplot(3, 1, 2)
        # Note: scipy.signal.chirp génère un signal réel. 
        # Pour voir une phase significative, on prend souvent la phase analytique (Hilbert)
        # ou simplement l'acos du signal normalisé, mais ici affichons le signal brut zoomé
        # ou mieux : le spectrogramme est plus parlant pour la fréquence.
        # Si le signal était complexe (exp), on ferait np.angle.
        # Ici, comme c'est réel, on va afficher un zoom sur la transition centrale.
        
        mid_point = int(N/2)
        zoom_range = 100 # points
        plt.plot(t_full[mid_point-zoom_range:mid_point+zoom_range]*1e6, 
                 self.symbol[mid_point-zoom_range:mid_point+zoom_range])
        plt.title("Zoom sur la transition centrale (Haut du triangle)")
        plt.xlabel("Temps (µs)")
        plt.grid(True)

        # 3. Spectrogramme (Time-Frequency)
        plt.subplot(3, 1, 3)
        plt.specgram(self.symbol, NFFT=256, Fs=self.fs_hz, noverlap=128, cmap='inferno')
        plt.title("Spectrogramme (Fréquence vs Temps)")
        plt.xlabel("Temps (s)")
        plt.ylabel("Fréquence (Hz)")
        plt.colorbar(label='Densité de Puissance Spectrale (dB/Hz)')
        plt.tight_layout()
        plt.show()
    


"""
# --- TEST DU FILTRAGE ADAPTÉ ---
if __name__ == "__main__":
    # 1. Configuration
    fs = 500e6
    T_chirp = 1e-6
    B = 200e6
    
    bp = Backprojection(fs, T_chirp, B)
    bp.build_chirp()
    bp.build_window()
    
    # 2. Création d'un signal reçu simulé (plus long que le chirp)
    # On imagine que le radar écoute pendant 3x la durée du chirp
    len_signal = int(3 * bp.N)
    rx_signal = np.zeros(len_signal)
    
    # On place le chirp (l'écho) à un endroit précis (ex: index 500)
    delay_idx = 500
    rx_signal[delay_idx : delay_idx + bp.N] = bp.symbol
    
    # 3. Ajout de bruit (SNR faible pour tester la robustesse)
    noise_power = 0.5 
    noise = np.random.normal(0, np.sqrt(noise_power), len_signal)
    rx_signal_noisy = rx_signal + noise
    
    # 4. Filtrage Adapté
    # Note: Votre fonction matched_filtering fait une FFT. 
    # Pour que la multiplication spectrale fonctionne, il faut que 'sig' et 'reference' aient la même taille.
    # Votre implémentation actuelle de matched_filtering suppose que 'sig' a la même taille que 'window'.
    # Nous devons donc adapter l'appel ou modifier la méthode pour gérer le Zero-Padding.
    
    # --- Adaptation temporaire pour votre méthode actuelle ---
    # Votre méthode matched_filtering multiplie terme à terme : reference * sig.
    # Cela implique que le signal reçu doit être découpé à la taille du chirp, 
    # OU que la référence doit être "padée" à la taille du signal reçu.
    # La méthode standard est de padder la référence.
    
    # On va modifier légèrement la logique ici pour utiliser numpy.correlate ou adapter votre classe.
    # Pour rester simple et utiliser VOTRE méthode, on va tricher : 
    # On va faire la convolution manuellement ici pour la démo, car votre méthode matched_filtering
    # semble conçue pour traiter une fenêtre de la même taille que le chirp (pulse compression pulse par pulse).
    
    # Utilisons une convolution standard pour voir le résultat sur tout le signal
    matched_output = np.correlate(rx_signal_noisy, bp.symbol * bp.window, mode='full')
    
    # 5. Visualisation
    plt.figure(figsize=(12, 8))
    
    # Signal Bruité
    plt.subplot(3, 1, 1)
    plt.plot(rx_signal_noisy)
    plt.title("Signal Reçu Bruité (Echo caché vers l'index 500)")
    plt.grid(True)
    
    # Chirp de Référence
    plt.subplot(3, 1, 2)
    plt.plot(bp.symbol * bp.window)
    plt.title("Référence (Chirp fenêtré)")
    plt.grid(True)
    
    # Résultat du Filtrage
    plt.subplot(3, 1, 3)
    lags = np.arange(len(matched_output)) - (len(bp.symbol) - 1)
    plt.plot(lags, np.abs(matched_output))
    plt.title("Sortie du Filtrage Adapté (Pic de corrélation)")
    plt.xlabel("Décalage (échantillons)")
    plt.grid(True)
    
    # On marque la position théorique
    plt.axvline(x=delay_idx, color='r', linestyle='--', label='Position Réelle')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    """