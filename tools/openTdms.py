import numpy as np
from nptdms import TdmsFile
import json
import os
import sys

def get_sar_data(tdms_path, config_path):
    """
    Reads SAR data from TDMS and reconstructs context from config.
    
    Args:
        tdms_path (str): Path to the .tdms file.
        config_path (str): Path to the config.json file.
        
    Returns:
        raw_signal (np.array): Complex 2D array (FastTime x SlowTime).
        platform_pos (np.array): 3D coordinates (3 x SlowTime).
    """
    # 1. Load Configuration
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # Extract parameters
    c = 299792458.0
    fs = float(config['fs'])
    max_range = float(config['maxRange'])
    flight_duration = float(config['flightDuration'])
    prf = float(config['prf'])
    speed = float(config['speed'])
    
    # Calculate dimensions
    # Matches MATLAB: numpulses = flightDuration/slowTime + 1
    # slowTime = 1/prf
    num_pulses = int(flight_duration * prf) + 1
    
    # Matches MATLAB: truncrangesamples = ceil((2*maxRange/c)*fs)
    num_range_samples = int(np.ceil((2 * max_range / c) * fs))
    
    print(f"Expected Dimensions: Range Samples={num_range_samples}, Pulses={num_pulses}")
    
    # 2. Load TDMS Data
    if not os.path.exists(tdms_path):
        raise FileNotFoundError(f"TDMS file not found: {tdms_path}")
        
    tdms_file = TdmsFile.read(tdms_path)
    
    # Find I and Q channels
    # MATLAB tdmswrite with timetable often puts data in a group. 
    # We search for channels ending in _I and _Q.
    channel_i = None
    channel_q = None
    
    for group in tdms_file.groups():
        for channel in group.channels():
            if channel.name.endswith('_I'):
                channel_i = channel
            elif channel.name.endswith('_Q'):
                channel_q = channel
                
    if channel_i is None or channel_q is None:
        # Fallback: try specific names if known or list available
        print("Available channels:")
        for group in tdms_file.groups():
            for channel in group.channels():
                print(f" - {group.name}/{channel.name}")
        raise ValueError("Could not find I/Q channels (ending in _I and _Q) in TDMS file.")
        
    data_i = channel_i[:]
    data_q = channel_q[:]
    
    # 3. Reshape Data
    # MATLAB flattened column-major (Fortran). 
    # Data stream: Pulse1, Pulse2, ...
    # Each pulse has num_range_samples.
    
    # Check total length
    expected_len = num_pulses * num_range_samples
    if len(data_i) != expected_len:
        print(f"Warning: Data length {len(data_i)} does not match expected {expected_len}. Adjusting num_pulses.")
        # Adjust num_pulses based on data length
        num_pulses = len(data_i) // num_range_samples
        # Truncate extra data if any (though usually it matches if logic is consistent)
        data_i = data_i[:num_pulses * num_range_samples]
        data_q = data_q[:num_pulses * num_range_samples]
    
    # Combine to complex
    raw_flat = data_i + 1j * data_q
    
    # Reshape
    # We want (FastTime, SlowTime).
    # The stream is [P1_S1, P1_S2, ..., P1_SN, P2_S1, ...]
    # reshape((num_pulses, num_range_samples)) -> Row 0 is Pulse 1.
    # Transpose -> Col 0 is Pulse 1. -> (num_range_samples, num_pulses)
    raw_signal = raw_flat.reshape((num_pulses, num_range_samples)).T
    
    # 4. Reconstruct Position (Ideal)
    # MATLAB: radarPlatform initialized at [0;0;2], Velocity [0;speed;0]
    # Loop runs num_pulses times with dt = 1/prf.
    # Position at step k (1-based) is: Initial + Velocity * (k * dt)
    
    dt = 1.0 / prf
    times = np.arange(1, num_pulses + 1) * dt
    
    # Positions: 3 x NumPulses
    # x = 0
    # y = speed * time
    # z = 2
    
    pos_x = np.zeros(num_pulses)
    pos_y = speed * times
    pos_z = np.full(num_pulses, 2.0)
    
    platform_pos = np.vstack((pos_x, pos_y, pos_z))
    
    return raw_signal, platform_pos

if __name__ == "__main__":
    # Default paths relative to this script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tdms_path = os.path.join(base_dir, 'output', 'simulation_raw.tdms')
    config_path = os.path.join(base_dir, 'conf', 'config.json')
    
    try:
        print(f"Reading {tdms_path}...")
        sig, pos = get_sar_data(tdms_path, config_path)
        print("Data loaded successfully.")
        print(f"Signal Shape: {sig.shape} (FastTime x SlowTime)")
        print(f"Position Shape: {pos.shape} (3 x SlowTime)")
        print(f"First Position: {pos[:,0]}")
        print(f"Last Position: {pos[:,-1]}")
    except Exception as e:
        print(f"Error: {e}")
