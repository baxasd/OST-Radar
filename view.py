import sys
import os
import time
import logging
import zmq
import numpy as np
import scipy.ndimage as ndimage
import pyqtgraph as pg

from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow

from core.base import RadarConfig  # [cite: 2]

# Configure the logger to match the publisher's clean aesthetic
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("Subscriber")

CFG_FILE  = "core/config.cfg"  # [cite: 1, 3]
MAX_RANGE = 5.0
CMAP             = "inferno"
DISPLAY_LOW_PCT  = 40
DISPLAY_HIGH_PCT = 99.5
SMOOTH_GRID_SIZE = 250


# ─────────────────────────────────────────────────────────────────────────────
#  Pre-flight Connection Check
# ─────────────────────────────────────────────────────────────────────────────
def is_publisher_active(ip: str, timeout_ms: int = 2000) -> bool:
    """
    Briefly listens to the ZMQ port. If no data arrives within the timeout,
    we assume the publisher is not running or the IP is wrong.
    """
    log.info(f"Checking for active stream at tcp://{ip}:5555...")
    
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://{ip}:5555")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    
    # Block and wait for up to `timeout_ms` for a message
    events = poller.poll(timeout_ms)
    
    # Clean up the test connection
    socket.close()
    context.term()
    
    # If events is not empty, data is flowing
    return len(events) > 0


# ─────────────────────────────────────────────────────────────────────────────
#  Background Network Thread
# ─────────────────────────────────────────────────────────────────────────────
class ZmqRadarWorker(QThread):
    new_frame = pyqtSignal(np.ndarray)
    error     = pyqtSignal(str)

    def __init__(self, config: RadarConfig, max_range_m: float, publisher_ip: str):
        super().__init__()
        self.cfg = config
        self.running = True

        self.num_range_bins = config.numRangeBins
        self.num_vel_bins   = config.numLoops
        self.max_bin = min(int(max_range_m / config.rangeRes), config.numRangeBins)
        self._expected_size = self.num_range_bins * self.num_vel_bins

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{publisher_ip}:5555")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

    def run(self):
        while self.running:
            try:
                try:
                    msg = self.socket.recv(flags=zmq.NOBLOCK)
                except zmq.Again:
                    time.sleep(0.001)
                    continue

                raw = np.frombuffer(msg, dtype=np.uint16)

                if raw.size != self._expected_size:
                    continue  # Silently drop malformed frames

                rd = raw.astype(np.float32).reshape(self.num_range_bins, self.num_vel_bins)
                rd = rd[:self.max_bin, :]

                display = 20.0 * np.log10(np.abs(np.fft.fftshift(rd, axes=1)) + 1e-6)
                self.new_frame.emit(display)

            except Exception as e:
                self.error.emit(str(e))

    def stop(self):
        self.running = False
        self.wait()
        self.socket.close()
        self.context.term()


# ─────────────────────────────────────────────────────────────────────────────
#  Main Window UI
# ─────────────────────────────────────────────────────────────────────────────
class ViewerWindow(QMainWindow):
    def __init__(self, config: RadarConfig, publisher_ip: str):
        super().__init__()
        self.cfg = config
        self.publisher_ip = publisher_ip

        self._build_plot()
        self._precompute_zoom()
        self._start_worker()
        
        self.setWindowTitle(f"OST Radar Subscriber [{self.publisher_ip}]")

    def _build_plot(self):
        max_bin = min(int(MAX_RANGE / self.cfg.rangeRes), self.cfg.numRangeBins)
        actual_max_m = max_bin * self.cfg.rangeRes
        dop_res = self.cfg.dopRes

        self.plot = pg.PlotWidget(title="Live Range-Doppler Heatmap")  # 
        self.plot.setLabel("left",   "Range",    units="m")
        self.plot.setLabel("bottom", "Velocity", units="m/s")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.addItem(pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("w", style=Qt.PenStyle.DashLine, width=1)))

        self.img = pg.ImageItem()
        self.img.setColorMap(pg.colormap.get(CMAP))
        self.plot.addItem(self.img)

        self._rect = pg.QtCore.QRectF(
            -self.cfg.dopMax - dop_res / 2.0, 
            0,
            self.cfg.dopMax * 2.0,            
            actual_max_m,                     
        )
        self.img.setRect(self._rect)
        self.setCentralWidget(self.plot)
        self.resize(900, 650)

    def _precompute_zoom(self):
        src_rows = min(int(MAX_RANGE / self.cfg.rangeRes), self.cfg.numRangeBins)
        src_cols = self.cfg.numLoops
        self._zoom_y = max(SMOOTH_GRID_SIZE, src_rows) / src_rows
        self._zoom_x = max(SMOOTH_GRID_SIZE, src_cols) / src_cols

    def _start_worker(self):
        self.worker = ZmqRadarWorker(self.cfg, MAX_RANGE, self.publisher_ip)
        self.worker.new_frame.connect(self._on_frame)
        self.worker.error.connect(lambda e: log.error(f"Worker Error: {e}"))
        self.worker.start()

    def _on_frame(self, matrix: np.ndarray):
        smooth = ndimage.zoom(matrix, (self._zoom_y, self._zoom_x), order=1)

        lo = float(np.percentile(smooth, DISPLAY_LOW_PCT))
        hi = float(np.percentile(smooth, DISPLAY_HIGH_PCT))
        if lo >= hi: 
            hi = lo + 0.1

        self.img.setImage(smooth, autoLevels=False, levels=(lo, hi))
        self.img.setRect(self._rect)

    def closeEvent(self, event):
        log.info("Closing viewer window...")
        self.worker.stop()
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
#  Application Menu
# ─────────────────────────────────────────────────────────────────────────────
def launch_viewer(ip: str):
    """Checks the stream and launches the PyQt application if active."""
    if not is_publisher_active(ip):
        log.error(f"CONNECTION FAILED: No radar data detected at {ip}.")
        log.error("Make sure 'publisher.py' is running and actively streaming.\n")
        return

    log.info("Stream detected! Launching GUI...")
    
    app = QApplication.instance() or QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)

    try:
        config = RadarConfig(CFG_FILE)
    except Exception as e:
        log.error(f"Failed to load config: {e}")
        return

    window = ViewerWindow(config, ip)
    window.show()
    app.exec()


def main():
     
    print("=========================================")
    print("        OST RADAR SUBSCRIBER       ")
    print("=========================================")

    while True:
        print("\nCONNECTION MENU:")
        print("  1. Connect to Localhost")
        print("  2. Connect to External IP")
        print("  3. Exit")
        
        choice = input("\nSelect an option (1-3): ").strip()
        
        if choice == '1':
            launch_viewer("127.0.0.1")
        elif choice == '2':
            custom_ip = input("Enter the Publisher's IP address: ").strip()
            if custom_ip:
                launch_viewer(custom_ip)
            else:
                log.warning("No IP entered. Returning to menu.")
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()