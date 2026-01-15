import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
from scipy import constants

class Backprojection:
    def __init__(self, fs_hz, T_chirp_s, B_hz, fc_hz):
        self.fs_hz = fs_hz
        self.T_chirp_s = T_chirp_s
        self.B_hz = B_hz
        self.fc_hz = fc_hz
        self.c_m_s = constants.speed_of_light
        self.symbol = None
        self.window = None

    def build_chirp(self):
        # 1. Chirp défini sur [-T/2, T/2] (Symetric)
        self.N_chirp = int(self.fs_hz * self.T_chirp_s)
        
        # Correction 1: Axe temporel centré (Symétrique)
        self.t_chirp = (np.arange(self.N_chirp) - self.N_chirp / 2) / self.fs_hz
        self.t_chirp = (np.arange(self.N_chirp)) / self.fs_hz 

        self.symbol = np.exp(1j * np.pi * self.B_hz * (self.t_chirp**2 / self.T_chirp_s - self.t_chirp))
        
        # Pente k = B / T
        k = self.B_hz / self.T_chirp_s
        
        # Chirp symétrique (balaye de -B/2 à B/2)
        # f(t) = k * t
        # Phi(t) = pi * k * t^2
        #self.symbol = np.exp(1j * np.pi * k * self.t_chirp**2)
        plt.plot(np.real(self.symbol))
        plt.show()

    def build_window(self):
        self.window = hann(self.N_chirp)
        #self.window = np.ones(self.N_chirp)

    def range_compression(self, mat_raw):
        """
        Correction MAJEURE ici : Le filtre adapté en fréquence est S * conj(FFT(ref))
        """
        N_samples, N_pulses = mat_raw.shape
        #mat_raw = mat_raw[1:47, :]
        # plt.plot(np.abs(mat_raw[:, 0]))
        # plt.grid()
        # plt.show()
        
        # Taille FFT pour convolution linéaire rapide
        n_fft = N_samples + self.N_chirp
        n_fft = int(2**np.ceil(np.log2(n_fft))) # Puissance de 2

        # 1. FFT du Signal
        S_fft = np.fft.fft(mat_raw, n=n_fft, axis=0)
        # plt.plot(S_fft[:, 0])
        # plt.show()
        
        # 2. FFT de la Référence (Le Chirp)
        # Attention : Pour ne pas décaler le pic temporel, on prend le chirp
        # tel quel et on fera le conjugué dans le domaine fréquentiel.
        
        # --- Fenêtrage Distance ---
        # NOTE: Pour voir le Sinus Cardinal (Sinc) classique, utiliser np.ones()
        # La fenêtre de Hann supprime les lobes secondaires.
        range_window = hann(self.N_chirp)
        #range_window = np.ones(self.N_chirp)
        ref = self.symbol * range_window
        print(f"{np.shape(self.symbol)}, ")
         
        Ref_fft = np.fft.fft(ref, n=n_fft)
        #Ref_fft = np.fft.fft(np.conj(ref), n=n_fft)
        #plt.plot(np.abs(tmp))
        #plt.show()
         
        
        # 3. Filtrage Adapté (Corrélation)
        # H(f) = S(f) * conj(Ref(f))
        OUT_fft = S_fft * np.conj(Ref_fft)[:, np.newaxis]
        #OUT_fft = S_fft * Ref_fft[:, np.newaxis]
        
        # 4. Retour Temporel
        compressed = np.fft.ifft(OUT_fft, axis=0)
        #plt.plot(a)
        # plt.grid()
        # plt.title(f"Range-compressed data")
        # plt.show()
    

        return compressed

    def create_grid(self, xlim, ylim, step):
        self.grid_x = np.arange(xlim[0], xlim[1], step)
        self.grid_y = np.arange(ylim[0], ylim[1], step)
        self.gx, self.gy = np.meshgrid(self.grid_x, self.grid_y)

    def image_formation(self, M_rc, M_pos):
        Ny, Nx = self.gx.shape
        N_fft_len, N_pulses = M_rc.shape
        
        image = np.zeros((Ny, Nx), dtype=complex)
        k_rad = 4 * np.pi * self.fc_hz / self.c_m_s

        # --- Fenêtrage Azimut ---
        az_window = hann(N_pulses)

        print(f"Backprojection de {N_pulses} pulses...")

        for i in range(N_pulses):
            pos_ant = M_pos[i] # [x, y, z]
            
            # 1. Distance Exacte
            dist = np.sqrt((self.gx - pos_ant[0])**2 + 
                           (self.gy - pos_ant[1])**2 + 
                           (0       - pos_ant[2])**2)
            
            # 2. Index Echantillon
            # Le délai aller-retour
            delay = 2 * dist / self.c_m_s
            
            # Note importante sur le délai de compression :
            # La convolution via FFT sans shift décale le résultat.
            # Avec np.fft.fft(chirp) où le chirp est défini sur [-T/2, T/2], 
            # numpy considère que le temps commence à 0.
            # L'index brut correspond donc directement au retard.
            idx = delay * self.fs_hz
            
            # 3. Interpolation (Gère le complexe)
            col = M_rc[:, i]
            
            # np.interp ne gère pas le complexe, on sépare
            val_real = np.interp(idx.ravel(), np.arange(N_fft_len), np.real(col), left=0, right=0)
            val_imag = np.interp(idx.ravel(), np.arange(N_fft_len), np.imag(col), left=0, right=0)
            
            val_interp = (val_real + 1j * val_imag).reshape(Ny, Nx)
            
            # 4. Correction de Phase
            # Le debug a montré que la phase mesurée suit bien -4*pi*R/lambda.
            # Il faut donc compenser par +4*pi*R/lambda.
            phase_corr = np.exp(1j * k_rad * dist)
            
            # Accumulation
            image += val_interp * phase_corr * az_window[i]
            
        return image
if __name__ == "__main__":
    # --- 1. Tes Paramètres (24 GHz) ---
    fc = 24e9       # 24 GHz
    B = 200e6       # 200 MHz -> Résolution Range = 75 cm
    
    # Échantillonnage : Il faut être au dessus de B (complexe)
    # Prenons de la marge pour une belle interpolation
    fs = 500e6      # 500 MHz
    
    T_chirp = 10e-6 # 10 µs (Chirp assez long pour donner de l'énergie)
    
    bp = Backprojection(fs, T_chirp, B, fc)
    bp.build_chirp()
    bp.build_window()
    
    # --- 2. Géométrie pour atteindre la limite Azimut ---
    # Lambda = c / 24GHz = 1.25 cm
    # Antenne estimée = 10 cm.
    # Beamwidth approx = Lambda / L_ant = 0.125 rad (~7 degrés)
    # À 300m de distance, le footprint est de ~37 mètres.
    # Il faut que L_synth > Footprint pour avoir la résolution max (L_ant/2).
    
    dist_target = 300.0 # On se met à 300m
    
    N_pulses = 1500    # Il faut "sur-échantillonner" spatialement (Lambda est petit !)
    L_synth = 50.0     # Trajectoire de 50m (suffisant pour couvrir le faisceau à 300m)
    
    y_traj = np.linspace(-L_synth/2, L_synth/2, N_pulses) 
    
    M_pos = np.zeros((N_pulses, 3))
    M_pos[:, 0] = y_traj 
    M_pos[:, 1] = -dist_target
    M_pos[:, 2] = 50.0 # Vol à 50m de haut
    
    target_pos = np.array([0, 0, 0])
    
    # --- 3. Buffer ---
    dist_max = np.linalg.norm(M_pos[0] - target_pos) + 20 
    tau_max = 2 * dist_max / constants.speed_of_light
    N_samples = int(tau_max * fs) + bp.N_chirp + 500
    
    print(f"--- SIMULATION 24 GHz ---")
    print(f"Résolution Range (Y) attendue : {constants.speed_of_light/(2*B):.3f} m")
    print(f"Résolution Azimut (X) théorique (L_ant/2) : ~0.05 m")
    
    raw_data = np.zeros((N_samples, N_pulses), dtype=complex)
    
    # --- 4. Génération ---
    print("Génération des données...")
    for i in range(N_pulses):
        dist = np.linalg.norm(M_pos[i] - target_pos)
        tau = 2 * dist / constants.speed_of_light
        idx_start = int(tau * fs)
        
        # Phase très sensible à 24 GHz !
        phase_carrier = np.exp(-1j * 4 * np.pi * fc * dist / constants.speed_of_light)
        
        if idx_start + bp.N_chirp < N_samples:
            raw_data[idx_start:idx_start+bp.N_chirp, i] = bp.symbol * phase_carrier

    plt.figure()
    plt.imshow(np.real(raw_data), aspect='auto', interpolation='nearest', cmap='inferno')
    plt.colorbar(label='Amplitude')
    plt.title("Raw Data Magnitude")
    plt.xlabel("Pulse Index")
    plt.ylabel("Sample Index")
    plt.show()

    # --- 5. Traitement ---
    print("Compression distance...")
    rc = bp.range_compression(raw_data)
    
    print("Backprojection...")
    # Grille avec pixels rectangulaires ou carrés ?
    # Prenons des pixels carrés très fins (2cm) pour voir la finesse en Azimut
    bp.create_grid([-1, 1], [-2, 2], 0.02) 
    
    img = bp.image_formation(rc, M_pos)
    
    # --- 6. Affichage ---
    plt.figure(figsize=(8, 10)) # Format portrait car la tache sera allongée en Y
    img_db = 20 * np.log10(np.abs(img) + 1e-9)
    vmax = np.max(img_db)
    vmin = vmax - 30 
    
    plt.imshow(img_db, extent=[-1, 1, -2, 2], origin='lower', cmap='inferno', vmin=vmin, vmax=vmax, interpolation='nearest')
    plt.colorbar(label='Amplitude (dB)')
    plt.title(f"24 GHz Backprojection\nResolution: Azimut < 10cm | Range ~75cm")
    plt.xlabel("Azimut X (m)")
    plt.ylabel("Range Y (m)")
    plt.grid(True, color='white', alpha=0.3, linestyle='--')
    
    # Cercle théorique de résolution pour comparaison
    # Ellipse de 5cm de large et 75cm de haut
    from matplotlib.patches import Ellipse
    ax = plt.gca()
    ell = Ellipse((0, 0), width=0.05, height=0.75, edgecolor='cyan', facecolor='none', linestyle='--', linewidth=2, label='Résolution Théorique')
    ax.add_patch(ell)
    plt.legend()
    
    plt.show()