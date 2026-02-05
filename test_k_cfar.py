import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from processing.detection.k_cfar import k_cfar_detector

# Load SAR image
mat = loadmat("output/out.mat")
image_amp = np.abs(mat["bpa_autofocused"])
print("Using autofocus SAR image")
print("Shape:", image_amp.shape)


# Run K-CFAR
detection_map, threshold_map = k_cfar_detector(
    image_amp,
    guard_size=2,
    train_size=12,
    pfa=1e-5
)


plt.figure()
plt.imshow(image_amp, cmap="gray")
plt.title("SAR image AFTER backprojection + autofocus")
plt.colorbar()
plt.show()


# Visualisation
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(image_amp, cmap="gray")
plt.title("SAR Amplitude Image")
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(detection_map, cmap="hot")
plt.title("K-CFAR Detection Map")
plt.colorbar()

plt.tight_layout()
plt.show()
