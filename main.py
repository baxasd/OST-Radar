import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from collections import deque

import radar.utils.parseCFG as cfg
from radar.utils.UARTconnection import (
    connectComPorts, 
    readAndParseUartDoubleCOMPort, 
    sendCfg
)

# ===================== USER CONFIG =====================
CLI_PORT = "COM10"
DATA_PORT = "COM11"
CFG_FILE = "radar/chirp_config/profile_3d_isk.cfg"

FRAME_DELAY = 0.05          # seconds
CLUTTER_ALPHA = 0.98       # closer to 1 = slower background
MIN_RANGE_METERS = 0.5      # ignore near-field junk
DB_MIN, DB_MAX = -80, 0     # color scale
DISPLAY_RANGE_MAX = 5.0     # meters to display
FPS_AVG_WINDOW = 30         # frames for FPS calculation
COLOR_SCALE_UPDATE_RATE = 10  # update scaling every N frames

# ======================================================

def setup_radar_connection():
    """Initialize radar connection and load configuration"""
    cli, data = connectComPorts(CLI_PORT, DATA_PORT)
    sendCfg(CFG_FILE, cli)
    cfg_params = cfg.cfg_params(CFG_FILE)
    
    print(f"Radar initialized:")
    print(f"  Range bins   : {cfg_params.ADCsamples}")
    print(f"  Doppler bins : {cfg_params.numLoops}")
    print(f"  Range res.   : {cfg_params.rangeRes:.3f} m")
    
    return cli, data, cfg_params

def create_axes(cfg_params):
    """Create range and velocity axes"""
    range_axis = cfg_params.rangeRes * np.arange(cfg_params.ADCsamples)
    velocity_axis = np.linspace(-cfg_params.dopMax, cfg_params.dopMax, cfg_params.numLoops)
    min_range_bin = max(1, int(MIN_RANGE_METERS / cfg_params.rangeRes))
    
    return range_axis, velocity_axis, min_range_bin

def process_radar_frame(raw_data, background, min_range_bin, cfg_params):
    """
    Process single radar frame: reshape, FFT shift, log scale, 
    background subtraction, range gating
    """
    if "RDHM" not in raw_data:
        return None, background
    
    # Reshape raw data to range-doppler matrix
    rd_matrix = np.array(raw_data["RDHM"], dtype=np.float32)
    rd_matrix = rd_matrix.reshape(cfg_params.ADCsamples, cfg_params.numLoops)
    
    # Center Doppler frequencies around zero
    rd_matrix = np.fft.fftshift(rd_matrix, axes=1)
    
    # Convert to dB scale
    rd_db = 20 * np.log10(np.abs(rd_matrix) + 1e-6)
    
    # Initialize or update background (clutter removal)
    if background is None:
        background = rd_db.copy()
    else:
        background = CLUTTER_ALPHA * background + (1.0 - CLUTTER_ALPHA) * rd_db
    
    # Subtract background and apply range gate
    rd_db_processed = rd_db - background
    rd_db_processed[:min_range_bin, :] = DB_MIN
    
    return rd_db_processed, background

def adaptive_color_scale(data, frame_count, update_rate=10):
    """
    Calculate color scale with adaptive updates to reduce flickering
    Updates less frequently for smoother display
    """
    if frame_count % update_rate == 0:
        # Use percentiles but exclude extreme values
        valid_data = data[data > DB_MIN + 10]  # Ignore very low values
        if len(valid_data) > 100:
            vmin = np.percentile(valid_data, 5)
            vmax = np.percentile(valid_data, 95)
        else:
            vmin, vmax = DB_MIN, DB_MAX
    return vmin, vmax

def main():
    """Main radar processing and visualization loop"""
    # Initialize radar connection
    cli, data, cfg_params = setup_radar_connection()
    range_axis, velocity_axis, min_range_bin = create_axes(cfg_params)
    
    # Set up visualization
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Create initial empty plot
    init_data = np.full((cfg_params.ADCsamples, cfg_params.numLoops), DB_MIN)
    mesh = ax.pcolormesh(velocity_axis, range_axis, init_data, 
                        shading='auto', vmin=DB_MIN, vmax=DB_MAX)
    
    # Configure plot
    plt.colorbar(mesh, ax=ax, label="Power (dB)")
    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("Range (m)")
    ax.set_ylim(0, DISPLAY_RANGE_MAX)
    ax.set_title("Range-Doppler Heatmap")
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Performance tracking
    background = None
    frame_count = 0
    frame_times = deque(maxlen=FPS_AVG_WINDOW)
    fps_text = ax.text(0.02, 0.98, 'FPS: --', transform=ax.transAxes, 
                      fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    print("\nStarting radar processing... Press Ctrl+C to exit.")
    
    try:
        while True:
            frame_start = time.time()
            
            # Read and process radar data
            raw_signal = readAndParseUartDoubleCOMPort(data)
            processed_data, background = process_radar_frame(
                raw_signal, background, min_range_bin, cfg_params
            )
            
            if processed_data is None:
                continue  # Skip invalid frame
            
            # Update visualization
            mesh.set_array(processed_data.ravel())
            
            # Update color scale periodically for smoother display
            vmin, vmax = adaptive_color_scale(processed_data, frame_count, COLOR_SCALE_UPDATE_RATE)
            mesh.set_clim(vmin, vmax)
            
            # Update FPS counter
            frame_times.append(time.time() - frame_start)
            fps = len(frame_times) / sum(frame_times) if frame_times else 0
            fps_text.set_text(f'FPS: {fps:.1f}')
            
            # Refresh display
            plt.pause(max(0.001, FRAME_DELAY))
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\nRadar processing stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        # Cleanup
        plt.ioff()
        plt.close('all')
        if 'cli' in locals() and cli:
            cli.close()
        if 'data' in locals() and data:
            data.close()
        print("Radar connection closed.")

if __name__ == "__main__":
    main()