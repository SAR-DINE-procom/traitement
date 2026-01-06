import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from openTdms import get_sar_data

def analyze_chirp():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tdms_path = os.path.join(base_dir, 'output', 'simulation_raw.tdms')
    config_path = os.path.join(base_dir, 'conf', 'config.json')
    
    # Load data
    raw_signal, _ = get_sar_data(tdms_path, config_path)
    
    # Take the first pulse
    pulse = raw_signal[:, 0]
    
    # Find the active part of the pulse (it might be padded or delayed)
    # The target is at ~3.46m (round trip ~23ns). 
    # Tpd is 100ns.
    # So the pulse should be present.
    
    amplitude = np.abs(pulse)
    threshold = 0.5 * np.max(amplitude)
    active_indices = np.where(amplitude > threshold)[0]
    
    if len(active_indices) == 0:
        print("No signal detected.")
        return

    start_idx = active_indices[0]
    end_idx = active_indices[-1]
    
    print(f"Pulse detected from index {start_idx} to {end_idx} (Length: {end_idx - start_idx})")
    
    signal_segment = pulse[start_idx:end_idx]
    
    # Phase analysis
    phase = np.unwrap(np.angle(signal_segment))
    
    # Time vector for the segment
    fs = 400e6 # From config
    t = np.arange(len(signal_segment)) / fs
    
    # Fit polynomial: Phase = 2*pi * (f0*t + 0.5*k*t^2) + phi0
    # Polyfit returns [p2, p1, p0] for p2*x^2 + p1*x + p0
    coeffs = np.polyfit(t, phase, 2)
    
    # Identify parameters
    # p2 = pi * k  => k = p2 / pi
    # p1 = 2 * pi * f0 => f0 = p1 / (2 * pi)
    
    k_est = coeffs[0] / np.pi
    f0_est = coeffs[1] / (2 * np.pi)
    
    print(f"Estimated Slope k: {k_est:.2e} Hz/s")
    print(f"Estimated Start Freq f0: {f0_est:.2e} Hz")
    
    # Expected
    bw = 200e6
    tpd = 1e-7
    k_expected = bw / tpd
    print(f"Expected Slope (BW/T): {k_expected:.2e} Hz/s")
    
    # Check if f0 is close to -BW/2 or 0
    print(f"f0 / BW = {f0_est/bw:.2f}")

if __name__ == "__main__":
    analyze_chirp()
