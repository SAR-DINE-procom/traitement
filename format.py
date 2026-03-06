import numpy as np
import pandas as pd
from scipy.io import savemat
import os

import matplotlib.pyplot as plt

def prepare_data_for_bp(npz_file, csv_file, output_mat_path="output/KMC4_RawData.mat"):
    print(f"--- Lecture des données ---")
    print(f"Radar : {npz_file}")
    print(f"Robot : {csv_file}")
    
    # 1. Chargement des données Radar (.npz)
    data = np.load(npz_file)
    mat_I = data['mat_I']
    mat_Q = data['mat_Q']
    fs = float(data['fs'])
    Tchirp = float(data['Tchirp'])
    B_Hz = float(data['B_Hz'])
    
    N_pulses, N_samples = mat_I.shape
    print(f"Dimensions radar : {N_pulses} impulsions, {N_samples} échantillons par impulsion.")

    # 2. Formatage de la matrice complexe pour le Backprojection
    # bp.m attend RawData_UP de taille [N_samples, N_pulses, N_channels]
    raw_complex = mat_I + 1j * mat_Q
    RawData_UP = np.transpose(raw_complex) # Devient [N_samples, N_pulses]
    RawData_UP = np.expand_dims(RawData_UP, axis=2) # Devient [N_samples, N_pulses, 1]

    # 3. Chargement et Interpolation des données Robot (.csv)
    # Les fréquences d'acquisition du robot et du radar diffèrent.
    # On interpole la position pour avoir exactement 1 position par impulsion radar.
    df = pd.read_csv(csv_file)
    
    # Adaptation dynamique aux noms de colonnes (à adapter si besoin)
    # Le CSV contient les colonnes pose_x, pose_y, pose_z
    if all(col in df.columns for col in ['pose_x', 'pose_y', 'pose_z']):
        pos_raw = df[['pose_x', 'pose_y', 'pose_z']].values
    elif all(col in df.columns for col in ['x', 'y', 'z']):
        pos_raw = df[['x', 'y', 'z']].values
    else:
        # Fallback : on prend les 3 premières colonnes numériques
        pos_raw = df.iloc[:, 0:3].values

    # --- VISUALISATION POS_RAW ---
    plt.figure()
    plt.plot(pos_raw[:, 0], label='x')
    plt.plot(pos_raw[:, 1], label='y')
    plt.plot(pos_raw[:, 2], label='z')
    plt.legend()
    plt.title("Position brute (pos_raw) avant interpolation")
    plt.show()
    # -----------------------------

    M = len(pos_raw)
    old_time = np.linspace(0, 1, M)
    new_time = np.linspace(0, 1, N_pulses)
    
    Pos_Radar = np.zeros((N_pulses, 3))
    for i in range(3): # Interpolation linéaire pour X, Y et Z
        Pos_Radar[:, i] = np.interp(new_time, old_time, pos_raw[:, i])

    # Calcul de la longueur de la course (Track Length)
    track_length = float(np.max(Pos_Radar[:, 0]) - np.min(Pos_Radar[:, 0]))
    print(f"track_length: {track_length}")

    # 4. Construction de la structure 'cfg' attendue par bp.m
    # /!\ ATTENTION : Remplace 122e9 par 24e9 si ton ST200 est un radar 24 GHz
    fc = 122e9 
    
    # Création d'une cible fictive pour éviter que le plot ground truth de bp.m ne crashe
    dummy_target = np.zeros(1, dtype=[('id', 'O'), ('pos', 'O')])
    dummy_target[0]['id'] = 1
    dummy_target[0]['pos'] = np.array([track_length/2, 2.5, 0.0])

    cfg = {
        'radar': {
            'fc': fc,
            'bandwidth': B_Hz,
            'adc_sample_rate': fs
        },
        'modulation': {
            'sweep_time': Tchirp
        },
        'platform': {
            'track_length_m': track_length
        },
        'antenna': {
            'rx_spacing_mm': 0.0 # 0 car 1 seul canal (ST200 de base)
        },
        'scene': {
            'targets': dummy_target
        }
    }

    # 5. Sauvegarde au format .mat
    os.makedirs(os.path.dirname(output_mat_path), exist_ok=True)
    
    mat_dict = {
        'RawData_UP': RawData_UP,
        'RawData_DOWN': RawData_UP, # Optionnel, mis en miroir au cas où
        'Pos_Radar': Pos_Radar,
        'cfg': cfg
    }
    
    savemat(output_mat_path, mat_dict)
    print(f"--- Succès ! Fichier généré : {output_mat_path} ---")

# ==========================================
# EXÉCUTION
# ==========================================
if __name__ == "__main__":
    # Remplace par les vrais noms de tes fichiers générés par l'IHM
    FICHIER_NPZ = "tools/test_grand_triedre_loin_moyen_plus_proche_20260305_104722_004.npz" 
    FICHIER_CSV = "tools/test_grand_triedre_loin_moyen_plus_proche_20260305_104722_004.csv"
    
    prepare_data_for_bp(FICHIER_NPZ, FICHIER_CSV)