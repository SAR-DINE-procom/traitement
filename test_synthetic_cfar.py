import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from processing.detection.k_cfar import k_cfar_detector

#Paramètres de l'image
Nx, Ny = 401, 401           # taille image
shape_clutter = 5.0          # nu (texture K/Gamma)
mean_intensity = 1.0
scale_clutter = mean_intensity / shape_clutter

# Clutter en intensité
I_clutter = gamma.rvs(
    a=shape_clutter,
    scale=scale_clutter,
    size=(Nx, Ny)
)

# Ajout cible (un point brillant)
I = I_clutter.copy()
target_pos = (80, 230)     #(row, col)
target_power = 50.0        #clutter moyen

# Pic central
I[target_pos] += target_power

# Petit lobe autour (PSF gaussienne 3x3)
for dx in [-1,0,1]:
    for dy in [-1,0,1]:
        if dx == 0 and dy == 0:
            continue
        I[target_pos[0]+dx, target_pos[1]+dy] += target_power / 4

#Conversion en amplitude 
image_amp = np.sqrt(I)

# Visualtion de l'image 
plt.figure(figsize=(6,6))
mappable = plt.imshow(20*np.log10(image_amp + 1e-6), cmap="gray")
plt.title("Synthetic SAR Image (log scale)")
plt.colorbar(mappable, label="dB")
plt.show()

# Application du K-CFAR
guard_size = 2
train_size = 12
pfa = 1e-7

detection_map, threshold_map = k_cfar_detector(
    image_amp,
    guard_size=guard_size,
    train_size=train_size,
    pfa=pfa
)

# Visualisation résultats
plt.figure(figsize=(6,6))
mappable = plt.imshow(20*np.log10(image_amp + 1e-6), cmap="gray")
plt.contour(detection_map, levels=[0.5], colors='r')
plt.title("K-CFAR Detections on Synthetic SAR Image")
plt.colorbar(mappable, label="dB")
plt.show()


