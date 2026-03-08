import sys                           # for sys.exit() and sys.argv passed to QApplication
import time                          # for time.time() used in the FPS calculation
import logging                       # for routing worker errors to the terminal instead of crashing
import numpy as np                   # all matrix math — reshape, fftshift, log10, percentile
import scipy.ndimage as ndimage      # for ndimage.zoom() — bilinear upscale of the heatmap image
import pyqtgraph as pg               # fast OpenGL-backed image rendering inside a Qt window

from PyQt6.QtCore    import QThread, pyqtSignal, Qt   # QThread = background thread; pyqtSignal = thread-safe callback; Qt = style enums
from PyQt6.QtWidgets import QApplication, QMainWindow  # QApplication = the Qt event loop; QMainWindow = the top-level window

from core.radar import RadarSensor   # hardware abstraction — serial ports, framing, config

log = logging.getLogger(__name__)   # module-level logger, name = "viewer"

# ─────────────────────────────────────────────
#  SETTINGS  —  edit these before running
# ─────────────────────────────────────────────
CFG_FILE  = "core/config.cfg"   # path to the TI mmWave config file relative to this script
MAX_RANGE = 5.0                 # clip the range (Y) axis to this many metres — saves vertical space

CLI_PORT  = None                # set to e.g. "/dev/ttyACM0" or "COM3" to skip auto-detection
DATA_PORT = None                # set to e.g. "/dev/ttyACM1" or "COM4" to skip auto-detection

CMAP             = "inferno"    # pyqtgraph colormap name — "inferno" is perceptually uniform and good for radar
DISPLAY_LOW_PCT  = 40           # dB values below this percentile are mapped to the bottom of the colour scale
DISPLAY_HIGH_PCT = 99.5        # dB values above this percentile are clipped to the top — suppresses rare hot pixels
SMOOTH_GRID_SIZE = 250         # bilinear zoom target: output image will be at least 250×250 pixels


# ─────────────────────────────────────────────────────────────────────────────
#  RadarWorker  —  background thread
#
#  This thread owns all radar I/O.  The Qt main thread never touches the serial
#  ports.  Communication back to the UI happens via the new_frame signal which
#  Qt marshals safely across the thread boundary.
#
#  Flow per iteration:
#    read_raw_frame() → parse_standard_frame() → reshape → fftshift → emit
# ─────────────────────────────────────────────────────────────────────────────

class RadarWorker(QThread):
    new_frame = pyqtSignal(np.ndarray)   # carries the processed dB matrix to the main thread for rendering
    error     = pyqtSignal(str)          # carries error strings to the main thread for logging

    def __init__(self, radar: RadarSensor, max_range_m: float):
        super().__init__()               # must call QThread.__init__ before anything else
        self.radar   = radar             # the RadarSensor instance — owns the serial ports
        self.running = True              # setting this to False from outside will exit the run() loop cleanly

        cfg = radar.config   # shorthand — avoids repeated attribute lookups inside the hot loop

        self.num_range_bins = cfg.numRangeBins   # total range FFT bins produced by the DSP (64 here, padded to power of 2)
        self.num_vel_bins   = cfg.numLoops       # Doppler velocity bins = number of chirp loops per frame (32)

        # How many range bins correspond to MAX_RANGE metres — we crop the matrix to this height
        self.max_bin = min(int(max_range_m / cfg.rangeRes), cfg.numRangeBins)

        # Pre-compute the expected flat array size for the incoming RDHM payload
        # If the received size doesn't match this, the frame is corrupt and gets dropped
        self._expected_size = self.num_range_bins * self.num_vel_bins

    def run(self):
        # This method runs in the background thread — everything here is off the Qt main thread
        while self.running:
            try:
                raw_bytes = self.radar.read_raw_frame()   # returns bytes or None

                if raw_bytes is None:
                    # No complete frame available yet.
                    # Sleep 1 ms to yield the CPU — without this the loop would spin at 100%
                    # and starve the Qt rendering thread, causing visible FPS drops.
                    time.sleep(0.001)
                    continue

                # Import is at module level in practice; kept here to make the dependency explicit
                from core.base import parse_standard_frame
                frame = parse_standard_frame(raw_bytes)   # returns {"error": 0, "RDHM": array or None}

                if frame.get("RDHM") is None:
                    continue   # this packet had no RDHM TLV — nothing to display, skip it

                raw = frame["RDHM"]   # flat uint16 numpy array of length num_range_bins × num_vel_bins

                if raw.size != self._expected_size:
                    # Size mismatch means the DSP sent a different layout than we expected
                    # This can happen briefly after a config change or hardware reset
                    log.warning("Shape mismatch — got %d, expected %d. Dropping.", raw.size, self._expected_size)
                    continue

                # Reshape the flat array into a 2D matrix: rows = range bins, cols = velocity bins
                rd = raw.astype(np.float32).reshape(self.num_range_bins, self.num_vel_bins)

                # Crop to MAX_RANGE — discard range bins beyond the distance we care about
                rd = rd[:self.max_bin, :]

                # fftshift moves the zero-velocity bin from the edges to the centre column
                # abs() converts complex-like values to magnitude (the data is already magnitude here but abs is safe)
                # log10 + ×20 converts linear power to dB — this is the standard "dB scale" for radar displays
                # +1e-6 prevents log10(0) which would produce -inf and break the colour scaling
                display = 20.0 * np.log10(np.abs(np.fft.fftshift(rd, axes=1)) + 1e-6)

                # Send the processed matrix to the main thread via the Qt signal
                # Qt handles the thread-safe copy automatically
                self.new_frame.emit(display)

            except Exception as e:
                self.error.emit(str(e))   # send the error string to the main thread for logging — don't crash the thread

    def stop(self):
        self.running = False   # signal the run() loop to exit on its next iteration
        self.wait()            # block until the thread has actually finished


# ─────────────────────────────────────────────────────────────────────────────
#  ViewerWindow  —  Qt main window
#
#  Owns the plot widget and the ImageItem that gets updated each frame.
#  All rendering happens here on the Qt main thread via _on_frame().
# ─────────────────────────────────────────────────────────────────────────────

class ViewerWindow(QMainWindow):

    def __init__(self, radar: RadarSensor):
        super().__init__()           # initialise the QMainWindow base class
        self.radar      = radar      # kept so closeEvent() can call radar.close()
        self._prev_time = time.time()  # timestamp of the last received frame — used for FPS calculation
        self._fps       = 0.0          # displayed in the window title bar

        self._build_plot(radar)      # create the pyqtgraph widgets
        self._precompute_zoom(radar) # calculate zoom scale factors once (they never change)
        self._start_worker(radar)    # launch the background thread
        self._update_title()         # set the initial window title

    def _build_plot(self, radar):
        cfg = radar.config   # shorthand

        # How many range bins fit within MAX_RANGE metres
        max_bin      = min(int(MAX_RANGE / cfg.rangeRes), cfg.numRangeBins)
        # The actual maximum range in metres after rounding to bin boundaries
        actual_max_m = max_bin * cfg.rangeRes

        dop_res = cfg.dopRes   # velocity step size in m/s — used to offset the image rectangle by half a bin

        # PlotWidget is pyqtgraph's main 2D plot container
        self.plot = pg.PlotWidget(title="Live Range-Doppler Heatmap")
        self.plot.setLabel("left",   "Range",    units="m")       # Y axis label
        self.plot.setLabel("bottom", "Velocity", units="m/s")     # X axis label
        self.plot.showGrid(x=True, y=True, alpha=0.3)             # faint grid lines for readability

        # A vertical dashed white line at velocity=0 to mark the stationary target column
        self.plot.addItem(pg.InfiniteLine(
            pos=0, angle=90,
            pen=pg.mkPen("w", style=Qt.PenStyle.DashLine, width=1),
        ))

        # ImageItem is a fast GPU-accelerated texture — we update it every frame
        self.img = pg.ImageItem()
        self.img.setColorMap(pg.colormap.get(CMAP))   # apply the chosen colormap (inferno)
        self.plot.addItem(self.img)

        # QRectF maps the image pixel coordinates to real-world axis coordinates.
        # X spans from -dopMax to +dopMax (velocity axis, centred on zero).
        # Y spans from 0 to actual_max_m (range axis, starting at the sensor).
        # The -dop_res/2 offset shifts the image by half a bin so bin centres align with grid lines.
        self._rect = pg.QtCore.QRectF(
            -cfg.dopMax - dop_res / 2.0,   # left edge of the image in m/s
            0,                              # bottom edge in metres
            cfg.dopMax * 2.0,              # total width in m/s
            actual_max_m,                  # total height in metres
        )
        self.img.setRect(self._rect)         # lock the image to these axis coordinates
        self.setCentralWidget(self.plot)     # make the plot fill the window
        self.resize(900, 650)                # initial window size in pixels

    def _precompute_zoom(self, radar):
        cfg = radar.config

        # Source matrix dimensions — what the worker thread emits
        src_rows = min(int(MAX_RANGE / cfg.rangeRes), cfg.numRangeBins)   # range bins after cropping
        src_cols = cfg.numLoops                                            # velocity bins (32)

        # Target dimensions — zoom up to at least SMOOTH_GRID_SIZE in each direction
        # This makes the heatmap look smoother on high-DPI displays without aliasing artifacts
        # The scale factors are constant for a given config, so we compute them once here
        self._zoom_y = max(SMOOTH_GRID_SIZE, src_rows) / src_rows   # e.g. 250/32 ≈ 7.8×
        self._zoom_x = max(SMOOTH_GRID_SIZE, src_cols) / src_cols   # e.g. 250/32 ≈ 7.8×

    def _start_worker(self, radar):
        self.worker = RadarWorker(radar, MAX_RANGE)
        self.worker.new_frame.connect(self._on_frame)                          # wire frame signal to render slot
        self.worker.error.connect(lambda e: log.error("Worker: %s", e))       # wire error signal to logger
        self.worker.start()                                                     # launch the background thread

    def _on_frame(self, matrix: np.ndarray):
        # Called by Qt on the main thread each time the worker emits new_frame

        # ── FPS measurement ───────────────────────────────────────────────────
        now             = time.time()
        dt              = now - self._prev_time   # seconds since last frame arrived from the worker
        self._fps       = 1.0 / dt if dt > 0 else 0.0   # avoid division by zero on the very first frame
        self._prev_time = now

        # ── Bilinear upscale ──────────────────────────────────────────────────
        # ndimage.zoom resizes the matrix using bilinear interpolation (order=1).
        # Without this the 32×32-ish raw matrix would display as a blocky pixelated image.
        # The scale factors were pre-computed in _precompute_zoom — no recalculation here.
        smooth = ndimage.zoom(matrix, (self._zoom_y, self._zoom_x), order=1)

        # ── Dynamic colour scaling ─────────────────────────────────────────────
        # Instead of a fixed dB range, we scale to the actual content of each frame.
        # percentile(40) as the floor cuts out the noise floor; percentile(99.5) as the
        # ceiling prevents a single hot pixel from washing out the entire image.
        lo = float(np.percentile(smooth, DISPLAY_LOW_PCT))
        hi = float(np.percentile(smooth, DISPLAY_HIGH_PCT))
        if lo >= hi:
            hi = lo + 0.1   # guard: if the frame is completely uniform, make a tiny valid range

        # Push the new image data to the GPU texture — autoLevels=False means we control the colour range
        self.img.setImage(smooth, autoLevels=False, levels=(lo, hi))
        self.img.setRect(self._rect)   # re-apply the axis rectangle (pyqtgraph can reset it internally)

        self._update_title()   # refresh the FPS counter in the title bar

    def _update_title(self):
        # Window title shows live FPS so we can monitor performance without any on-screen overlay
        self.setWindowTitle(f"OST Radar  [LIVE]  FPS: {self._fps:.1f}")

    def closeEvent(self, event):
        # Called by Qt when the user closes the window (X button or OS close)

        self.worker.running = False   # tell the worker loop to exit on its next iteration

        # Close the serial ports immediately — this causes any blocking serial.read() in the
        # worker to raise an exception or return empty, which unblocks the thread so it can exit
        self.radar.close()

        self.worker.wait()   # wait for the worker thread to fully finish before Qt tears down the window
        event.accept()       # allow the close to proceed


# ─────────────────────────────────────────────
#  Startup helpers
# ─────────────────────────────────────────────

def connect_radar() -> RadarSensor:
    cli, data = CLI_PORT, DATA_PORT   # start with whatever the user set manually (may be None)

    if cli is None or data is None:
        print("Auto-detecting ports...")
        found_cli, found_data = RadarSensor.find_ti_ports()   # scan USB ports for TI device
        cli  = cli  or found_cli    # use manual value if set, otherwise use auto-detected
        data = data or found_data

    if not cli or not data:
        # Still None after auto-detection — print a helpful message and exit
        print(
            "ERROR: Ports not found.\n"
            "Set CLI_PORT / DATA_PORT manually at the top of viewer.py.\n"
            "Linux:   /dev/ttyACM0, /dev/ttyACM1\n"
            "Windows: COM3, COM4"
        )
        sys.exit(1)

    print(f"CLI: {cli}  DATA: {data}")
    radar = RadarSensor(cli, data, CFG_FILE)   # create the sensor object (parses config, doesn't open ports yet)
    radar.connect_and_configure()              # open serial ports and send the .cfg to the hardware
    return radar


def print_radar_info(radar: RadarSensor):
    # Print a formatted table to the terminal showing the derived radar parameters
    print("--- Radar Config ---")
    print(f"{'Max Range':<20}: {MAX_RANGE} m")   # show the user-configured clipping distance
    for k, v in radar.config.summary().items():
        print(f"{k:<20}: {str(v)}")   # add each derived parameter from RadarConfig.summary()
    print("--------------------")


def main():
    # Configure Python's logging to print WARNING and above to the terminal
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(name)s  %(message)s")

    radar = connect_radar()   # find ports, open serial, send config — blocks until the radar is ready

    app = QApplication(sys.argv)   # create the Qt application — must exist before any Qt widgets

    # row-major: numpy arrays are row-major by default; this tells pyqtgraph to match that convention
    # antialias=False: faster rendering — we don't need antialiasing on a heatmap
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)

    print_radar_info(radar)   # print the config table to the terminal before the window opens

    window = ViewerWindow(radar)   # create the main window (also starts the worker thread)
    window.show()                  # make the window visible

    sys.exit(app.exec())   # start the Qt event loop — blocks until the window is closed


if __name__ == "__main__":
    main()