import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from tools.openTdms import get_sar_data
from processing.backprojection import Backprojection

def debug_phase():
    # Paths
    tdms_path = os.path.join(parent_dir, 'output', 'simulation_raw.tdms')
    config_path = os.path.join(parent_dir, 'conf', 'config.json')
    
    # 1. Load Data & Config
    raw_signal, platform_pos = get_sar_data(tdms_path, config_path)
    platform_pos = platform_pos.T # (N_pulses, 3)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    fs = float(config['fs'])
    fc = float(config['fc'])
    bw = float(config['bw'])
    tpd = float(config['tpd'])
    c = 299792458.0
    
    # 2. Range Compression
    bp = Backprojection(fs, tpd, bw, fc)
    bp.build_chirp()
    bp.build_window()
    rc_data = bp.range_compression(raw_signal)
    
    # 3. Target Geometry
    target_pos = np.array([2.0, 2.0, 0.0]) # Known target position
    
    # 4. Analyze Phase History
    num_pulses = rc_data.shape[1]
    phases_measured = []
    phases_theoretical = []
    distances = []
    
    print("Analyzing phase history...")
    
    for i in range(num_pulses):
        radar_pos = platform_pos[i]
        dist = np.linalg.norm(radar_pos - target_pos)
        distances.append(dist)
        
        # Theoretical Phase (Two-way)
        # Standard SAR: exp(-j * 4 * pi * fc * R / c)
        phi_theo = -4 * np.pi * fc * dist / c
        phases_theoretical.append(phi_theo)
        
        # Measured Phase
        # Find index corresponding to distance
        delay = 2 * dist / c
        idx = delay * fs
        
        # Extract value at index (nearest neighbor for simplicity, or interp)
        idx_int = int(round(idx))
        if 0 <= idx_int < rc_data.shape[0]:
            val = rc_data[idx_int, i]
            phases_measured.append(np.angle(val))
        else:
            phases_measured.append(0)

    phases_measured = np.unwrap(phases_measured)
    phases_theoretical = np.unwrap(phases_theoretical)
    
    # 5. Plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(phases_measured, label='Measured Phase (Unwrapped)')
    plt.plot(phases_theoretical, label='Theoretical Phase (-4pi R/lambda)')
    plt.legend()
    plt.title('Phase History')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    # Remove constant offset for comparison
    diff = phases_measured - phases_theoretical
    diff = np.unwrap(diff) # Unwrap again just in case
    diff = diff - diff[0] # Zero start
    
    plt.plot(diff, label='Residual (Measured - Theoretical)')
    plt.plot(phases_measured + phases_theoretical, label='Sum (Measured + Theoretical)')
    plt.legend()
    plt.title('Phase Residual')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(distances)
    plt.title('Distance to Target')
    plt.ylabel('Meters')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(parent_dir, 'output', 'debug_phase.png'))
    print("Debug plot saved to output/debug_phase.png")
    
    # Analysis
    slope_diff = (diff[-1] - diff[0]) / num_pulses
    print(f"Residual Drift: {diff[-1] - diff[0]:.2f} rad over trajectory")
    
    # Check if Sum is constant (implies sign inversion)
    sum_sig = phases_measured + phases_theoretical
    sum_sig = np.unwrap(sum_sig)
    drift_sum = sum_sig[-1] - sum_sig[0]
    print(f"Sum Drift: {drift_sum:.2f} rad (If small, implies Sign Inversion)")

if __name__ == "__main__":
    debug_phase()
