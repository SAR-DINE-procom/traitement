import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to allow imports from tools and processing
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from tools.openTdms import get_sar_data
from backprojection import Backprojection
from autofocus import pga_autofocus

def main():
    # Paths
    tdms_path = os.path.join(parent_dir, 'output', 'simulation_raw.tdms')
    # Utilisation de config2.json comme demandé
    config_path = os.path.join(parent_dir, 'conf', 'config2.json')
    
    # 1. Load Data
    print("Loading SAR data...")
    try:
        raw_signal, platform_pos = get_sar_data(tdms_path, config_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Transpose position to match Backprojection expectation (N_pulses, 3)
    platform_pos = platform_pos.T
    
    # 2. Load Config for Radar Parameters
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    fs = float(config['fs'])
    fc = float(config['fc'])
    bw = float(config['bw'])
    tpd = float(config['tpd'])
    
    print(f"Parameters: fs={fs/1e6}MHz, fc={fc/1e9}GHz, BW={bw/1e6}MHz, Tpd={tpd*1e6}us")

    # 3. Initialize Backprojection
    bp = Backprojection(fs, tpd, bw, fc)
    bp.build_chirp()
    bp.build_window()
    
    # 4. Range Compression
    print("Performing Range Compression...")
    rc_data = bp.range_compression(raw_signal)
    
    # Debug: Check peak location for first pulse
    peak_idx = np.argmax(np.abs(rc_data[:, 0]))
    peak_dist = (peak_idx / fs) * 3e8 / 2
    print(f"Debug: Peak detected at index {peak_idx}, approx distance {peak_dist:.2f}m")
    
    # 5. Image Formation
    print("Performing Backprojection...")
    
    # Define Grid
    # Wide view
    range_lims = [0, 6]      # X axis (Range)
    cross_range_lims = [0, 5]  # Y axis (Cross-Range)
    step = 0.05 # 5cm grid
    
    bp.create_grid(range_lims, cross_range_lims, step)
    
    # Note: Backprojection.image_formation expects M_pos as (N_pulses, 3)
    image = bp.image_formation(rc_data, platform_pos)
    
    # 6. Autofocus
    print("Performing Autofocus (PGA)...")
    # pga_autofocus returns (corrected_image, phase_error)
    img_autofocus, phase_error = pga_autofocus(image, iterations=9)
    
    # 7. Visualization
    print("Displaying Images...")
    plt.figure(figsize=(12, 6))
    
    # --- Image Originale ---
    plt.subplot(1, 2, 1)
    img_db = 20 * np.log10(np.abs(image) + 1e-9)
    vmax = np.max(img_db)
    vmin = vmax - 40
    
    plt.imshow(img_db.T, 
               extent=[cross_range_lims[0], cross_range_lims[1], range_lims[0], range_lims[1]], 
               origin='lower', 
               cmap='jet', 
               vmin=vmin, 
               vmax=vmax,
               aspect='equal')
    plt.colorbar(label='Amplitude (dB)')
    plt.title('SAR Image (Original)')
    plt.xlabel('Cross-Range (m)')
    plt.ylabel('Range (m)')
    plt.grid(True, alpha=0.3)

    # --- Image Autofocus ---
    plt.subplot(1, 2, 2)
    img_autofocus_db = 20 * np.log10(np.abs(img_autofocus) + 1e-9)
    
    plt.imshow(img_autofocus_db.T, 
               extent=[cross_range_lims[0], cross_range_lims[1], range_lims[0], range_lims[1]], 
               origin='lower', 
               cmap='jet', 
               vmin=vmin, 
               vmax=vmax,
               aspect='equal')
    plt.colorbar(label='Amplitude (dB)')
    plt.title('SAR Image (PGA Autofocus)')
    plt.xlabel('Cross-Range (m)')
    plt.ylabel('Range (m)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.title('SAR Image (Backprojection)')
    plt.xlabel('Cross-Range (m)')
    plt.ylabel('Range (m)')
    plt.grid(True, alpha=0.3)
    
    output_fig = os.path.join(parent_dir, 'output', 'sar_image_python.png')
    plt.savefig(output_fig)
    print(f"Image saved to {output_fig}")
    plt.show()

if __name__ == "__main__":
    main()
