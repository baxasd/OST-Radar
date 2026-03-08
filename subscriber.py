import sys                           # Used for passing arguments to QApplication and exiting
import time                          # Used to calculate the time between frames for the FPS counter
import argparse                      # Used to parse command-line arguments (like the Publisher's IP)
import logging                       # Used to gracefully log errors from the background thread
import zmq                           # The ZeroMQ library for high-performance network sockets
import numpy as np                   # Used to reconstruct the bytes back into a mathematical matrix
import scipy.ndimage as ndimage      # Used to smoothly upscale (bilinear zoom) the low-res radar image
import pyqtgraph as pg               # The fast, OpenGL-backed plotting library for our UI

# Import specific components from PyQt6 for multithreading and UI management
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow

# Import the RadarConfig parser so the UI knows how to draw the physical axis measurements
from core.base import RadarConfig

# Create a logger specifically for the viewer
log = logging.getLogger(__name__)

# Hardcoded settings for the UI display
CFG_FILE  = "core/config.cfg"        # The local copy of the config file used to map the axes
MAX_RANGE = 5.0                      # Clip the Y-axis to 5.0 meters to save screen space
CMAP             = "inferno"         # The colormap used for the heatmap (perceptually uniform)
DISPLAY_LOW_PCT  = 40                # The dynamic floor percentile to filter out background noise
DISPLAY_HIGH_PCT = 99.5              # The dynamic ceiling percentile to prevent hot-pixel washouts
SMOOTH_GRID_SIZE = 250               # Upscale the raw 32x64 matrix to at least 250x250 pixels


class ZmqRadarWorker(QThread):
    # Create a thread-safe signal to pass the fully processed Numpy matrix to the main UI thread
    new_frame = pyqtSignal(np.ndarray)
    
    # Create a thread-safe signal to pass error messages to the main thread's logger
    error     = pyqtSignal(str)

    def __init__(self, config: RadarConfig, max_range_m: float, publisher_ip: str):
        # Initialize the underlying Qt background thread
        super().__init__()
        
        # Store the configuration object
        self.cfg = config
        
        # A flag we can set to False to gracefully terminate the infinite while loop
        self.running = True

        # Extract the dimensions of the incoming matrix from the config file
        self.num_range_bins = config.numRangeBins
        self.num_vel_bins   = config.numLoops
        
        # Calculate exactly which row index corresponds to our MAX_RANGE cutoff
        self.max_bin = min(int(max_range_m / config.rangeRes), config.numRangeBins)
        
        # Pre-calculate the exact number of elements we expect in the 1D array
        self._expected_size = self.num_range_bins * self.num_vel_bins

        # Setup the ZeroMQ background context
        self.context = zmq.Context()
        
        # Create a SUB (Subscriber) socket designed to listen to incoming broadcasts
        self.socket = self.context.socket(zmq.SUB)
        
        # Connect to the Publisher using the IP address provided by the user (or localhost)
        self.socket.connect(f"tcp://{publisher_ip}:5555")
        
        # Tell ZeroMQ to subscribe to ALL topics (the empty string means no filtering)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

    def run(self):
        # This loop runs continuously in the background thread
        while self.running:
            try:
                # Attempt to read a packet from the network
                try:
                    # zmq.NOBLOCK ensures that if no data is arriving, the code immediately throws an error
                    # rather than freezing. This allows the thread to check 'self.running' and close cleanly.
                    msg = self.socket.recv(flags=zmq.NOBLOCK)
                    
                # Catch the specific exception ZeroMQ throws when NOBLOCK yields no data
                except zmq.Again:
                    # Sleep for 1 millisecond and loop back to try again
                    time.sleep(0.001)
                    continue

                # Convert the raw bytes received over the network directly back into a 1D uint16 Numpy array
                raw = np.frombuffer(msg, dtype=np.uint16)

                # Guard check: If the array size doesn't match our config, the hardware settings changed
                if raw.size != self._expected_size:
                    log.warning("Shape mismatch — got %d, expected %d.", raw.size, self._expected_size)
                    continue

                # Reshape the 1D array into a 2D matrix (Rows = Range, Columns = Velocity)
                rd = raw.astype(np.float32).reshape(self.num_range_bins, self.num_vel_bins)
                
                # Slice the matrix vertically to discard any data beyond MAX_RANGE
                rd = rd[:self.max_bin, :]

                # 1. fftshift: Move velocity=0 from the edges of the matrix to the center column
                # 2. abs: Ensure all values are positive magnitudes
                # 3. log10 * 20: Convert the linear magnitude into standard decibels (dB)
                display = 20.0 * np.log10(np.abs(np.fft.fftshift(rd, axes=1)) + 1e-6)
                
                # Emit the processed dB matrix to the main UI thread for drawing
                self.new_frame.emit(display)

            # Catch any generic mathematical or decoding errors so the thread doesn't crash
            except Exception as e:
                # Emit the error string to the main thread
                self.error.emit(str(e))

    def stop(self):
        # Break the infinite while loop
        self.running = False
        
        # Block the main thread until this background thread confirms it has finished executing
        self.wait()
        
        # Close the network socket to release the port
        self.socket.close()
        
        # Terminate the ZeroMQ context
        self.context.term()


class ViewerWindow(QMainWindow):
    def __init__(self, config: RadarConfig, publisher_ip: str):
        # Initialize the Qt Main Window base class
        super().__init__()
        
        # Store the radar configuration
        self.cfg = config
        
        # Store the network IP
        self.publisher_ip = publisher_ip
        
        # Keep track of the exact time the last frame arrived to calculate FPS
        self._prev_time = time.time()
        
        # Initialize the FPS variable
        self._fps = 0.0

        # Build the graph axes and image items
        self._build_plot()
        
        # Calculate the mathematical constants needed to upscale the image
        self._precompute_zoom()
        
        # Launch the network listening thread
        self._start_worker()
        
        # Set the text at the top of the window
        self._update_title()

    def _build_plot(self):
        # Calculate exactly how many meters the cropped matrix represents
        max_bin = min(int(MAX_RANGE / self.cfg.rangeRes), self.cfg.numRangeBins)
        actual_max_m = max_bin * self.cfg.rangeRes
        
        # Get the velocity resolution to offset the image so bin centers align with the grid
        dop_res = self.cfg.dopRes

        # Create the pyqtgraph 2D coordinate system
        self.plot = pg.PlotWidget(title="Live Range-Doppler Heatmap")
        
        # Label the axes with correct real-world units
        self.plot.setLabel("left",   "Range",    units="m")
        self.plot.setLabel("bottom", "Velocity", units="m/s")
        
        # Turn on a faint grid for easier reading
        self.plot.showGrid(x=True, y=True, alpha=0.3)

        # Draw a vertical dashed line perfectly down the center (0 m/s velocity = stationary objects)
        self.plot.addItem(pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("w", style=Qt.PenStyle.DashLine, width=1)))

        # Create the GPU-accelerated texture object that holds the pixels
        self.img = pg.ImageItem()
        
        # Apply the chosen color scale (inferno)
        self.img.setColorMap(pg.colormap.get(CMAP))
        
        # Add the texture into the 2D coordinate system
        self.plot.addItem(self.img)

        # Define the bounding box (in real-world meters) where the image should be stretched
        self._rect = pg.QtCore.QRectF(
            -self.cfg.dopMax - dop_res / 2.0,  # Left edge (Negative max velocity)
            0,                                 # Bottom edge (0 meters range)
            self.cfg.dopMax * 2.0,             # Total width (Full velocity span)
            actual_max_m,                      # Total height (Cropped max range)
        )
        
        # Lock the image to this exact bounding box
        self.img.setRect(self._rect)
        
        # Set this plot to fill the entire application window
        self.setCentralWidget(self.plot)
        
        # Set the default window dimensions in pixels
        self.resize(900, 650)

    def _precompute_zoom(self):
        # Find the raw pixel dimensions of the cropped matrix
        src_rows = min(int(MAX_RANGE / self.cfg.rangeRes), self.cfg.numRangeBins)
        src_cols = self.cfg.numLoops
        
        # Calculate the scalar multiplier required to stretch it to at least SMOOTH_GRID_SIZE
        self._zoom_y = max(SMOOTH_GRID_SIZE, src_rows) / src_rows
        self._zoom_x = max(SMOOTH_GRID_SIZE, src_cols) / src_cols

    def _start_worker(self):
        # Instantiate the background ZMQ thread, passing the target IP
        self.worker = ZmqRadarWorker(self.cfg, MAX_RANGE, self.publisher_ip)
        
        # Connect the worker's 'new_frame' signal to our local drawing function
        self.worker.new_frame.connect(self._on_frame)
        
        # Connect the worker's 'error' signal directly to the logger
        self.worker.error.connect(lambda e: log.error("Worker: %s", e))
        
        # Start the thread execution
        self.worker.start()

    def _on_frame(self, matrix: np.ndarray):
        # This function runs on the main thread every time ZMQ receives a packet

        # Calculate how much time passed since this function last ran
        now = time.time()
        dt = now - self._prev_time
        
        # Protect against division by zero, then calculate Frames Per Second
        self._fps = 1.0 / dt if dt > 0 else 0.0
        
        # Update the clock for the next frame
        self._prev_time = now

        # Upscale the blocky matrix using bilinear interpolation so it looks smooth
        smooth = ndimage.zoom(matrix, (self._zoom_y, self._zoom_x), order=1)

        # Calculate the noise floor based on our dynamic percentile (e.g., 40th percentile)
        lo = float(np.percentile(smooth, DISPLAY_LOW_PCT))
        
        # Calculate the peak ceiling based on our dynamic percentile (e.g., 99.5th percentile)
        hi = float(np.percentile(smooth, DISPLAY_HIGH_PCT))
        
        # Safeguard: If the frame is mathematically flat, force a tiny gap so the colormap doesn't crash
        if lo >= hi: 
            hi = lo + 0.1

        # Push the smoothed pixels to the GPU texture, explicitly defining the black/white cutoff points
        self.img.setImage(smooth, autoLevels=False, levels=(lo, hi))
        
        # Re-assert the physical bounding box (pyqtgraph occasionally forgets it during updates)
        self.img.setRect(self._rect)
        
        # Refresh the window title with the new FPS
        self._update_title()

    def _update_title(self):
        # Display the IP we are listening to alongside the live FPS
        self.setWindowTitle(f"OST Radar Subscriber [{self.publisher_ip}]  FPS: {self._fps:.1f}")

    def closeEvent(self, event):
        # This function is triggered by PyQt when the user clicks the 'X' button
        
        # Tell the background thread to safely close its network socket and exit
        self.worker.stop()
        
        # Accept the event, allowing the window to legally close and destroy itself
        event.accept()


def main():
    # Set the logging level to WARNING
    logging.basicConfig(level=logging.WARNING)
    
    # Set up argparse so the user can define the IP address in the terminal
    parser = argparse.ArgumentParser(description="Radar UI Subscriber")
    parser.add_argument(
        "--ip", 
        type=str, 
        default="localhost", 
        help="The IP address of the machine running the Publisher (e.g., 192.168.1.50). Defaults to localhost."
    )
    args = parser.parse_args()

    # Create the core PyQt application loop
    app = QApplication(sys.argv)
    
    # Optimize pyqtgraph: read arrays as row-major (standard for Numpy) and disable anti-aliasing for speed
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)

    # Parse the local config file to figure out the axis measurements
    config = RadarConfig(CFG_FILE)

    # Create the main window, passing in the config and the target IP address
    window = ViewerWindow(config, args.ip)
    
    # Make the window visible on the screen
    window.show()
    
    # Hand control over to the PyQt event loop; the script will block here until the window is closed
    sys.exit(app.exec())

if __name__ == "__main__":
    main()