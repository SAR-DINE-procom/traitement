import numpy as np
from scipy.special import kv, gamma
from scipy.optimize import brentq
from scipy.ndimage import maximum_filter
from scipy.stats import gamma as gamma_dist
from scipy.ndimage import uniform_filter, maximum_filter
from tqdm import tqdm


# K-CFAR implementation for SAR image detection
def estimate_k_params_mom(intensity):
    """
    Method of Moments estimation of K-distribution parameters
    intensity : 1D array of training cell intensities
    Returns : nu (shape), mu (mean intensity)
    """
    mu = np.median(intensity)
    var = np.var(intensity)

    # Avoid degenerate cases
    if var <= mu**2:
        # Approximates Gamma clutter (high nu)
        nu = 50.0
    else:
        nu = mu**2 / (var - mu**2)

    return nu, mu


# CDF of K-distribution for intensity
def k_cdf_intensity(x, nu, mu):
    """
    CDF of K-distribution for intensity
    """
    arg = 2.0 * np.sqrt(nu * np.maximum(x, 1e-12) / mu)
    coef = 2.0 / gamma(nu)
    return 1.0 - coef * (arg / 2.0)**nu * kv(nu, arg)

def k_threshold(nu, mu, pfa):
    """
    Robust threshold computation with Gamma fallback
    """

    # Gamma approximation if nu is large
    if nu > 30:
        shape = nu
        scale = mu / nu
        return gamma_dist.ppf(1 - pfa, a=shape, scale=scale)

    def func(T):
        return 1.0 - k_cdf_intensity(T, nu, mu) - pfa

    T_min = 1e-12
    T_max = 1e3 * mu

    try:
        return brentq(func, T_min, T_max, maxiter=100)
    except ValueError:
        # Ultimate fallback: empirical scaling
        return mu * (-np.log(pfa))



def extract_training_cells(I, i, j, g, t):
    """
    Extract training cells excluding guard and CUT
    """
    i_min = i - (g + t)
    i_max = i + (g + t) + 1
    j_min = j - (g + t)
    j_max = j + (g + t) + 1

    window = I[i_min:i_max, j_min:j_max]

    # Mask guard + CUT
    center = g + t
    mask = np.ones_like(window, dtype=bool)
    mask[center-g:center+g+1, center-g:center+g+1] = False

    return window[mask]

def remove_isolated_pixels(detection_map, window_size=3):
    """
    Ne garde que les pixels qui sont des maxima locaux dans leur voisinage.
    Pixels isolés disparaissent.
    """
    # Calcul du maximum local
    local_max = maximum_filter(detection_map, size=window_size)
    
    # Garder seulement les pixels qui sont le maximum local
    detection_map_filtered = detection_map * (detection_map == local_max)
    
    return detection_map_filtered

def estimate_nu_global(I, max_samples=200000):
    """
    Estimation globale de nu (texture K) sur toute l'image
    pour stabiliser le CFAR
    """
    flat = I.ravel()

    if flat.size > max_samples:
        flat = np.random.choice(flat, max_samples, replace=False)

    mu = np.median(flat)
    var = np.var(flat)

    if var <= mu**2:
        return 50.0  # Gamma-like clutter
    else:
        return mu**2 / (var - mu**2)

def k_cfar_detector(
    image_amp,
    guard_size=4,
    train_size=20,
    pfa=1e-7
):
    """
    Robust K-CFAR detector for SAR images
    """

    # Convert amplitude to intensity and normalize
    I = image_amp**2
    I = I / np.mean(I)

    Nx, Ny = I.shape
    detection_map = np.zeros_like(I, dtype=np.uint8)
    threshold_map = np.zeros_like(I)

    # CUT lissé
    I_cut = uniform_filter(I, size=3)

    # nu global 
    nu_global = estimate_nu_global(I)

    margin = guard_size + train_size

    for i in tqdm(range(margin, Nx - margin), desc="K-CFAR"):
        for j in range(margin, Ny - margin):

            training_cells = extract_training_cells(
                I, i, j, guard_size, train_size
            )

            if training_cells.size < 10:
                continue

            # mu local robuste
            mu = np.median(training_cells)
            nu = nu_global

            T = k_threshold(nu, mu, pfa)
            threshold_map[i, j] = T

            # test sur CUT lissé (pas pixel brut)
            if I_cut[i, j] > T:
                detection_map[i, j] = 1

    # --- Suppression des pixels isolés ---
    detection_map = remove_isolated_pixels(detection_map, window_size=3)

    return detection_map, threshold_map
