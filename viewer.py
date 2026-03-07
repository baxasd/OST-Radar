import sys
import time
import logging
import numpy as np
import scipy.ndimage as ndimage
import pyqtgraph as pg

from PyQt6.QtCore    import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow

from core.sensor import RadarSensor

from rich.console import Console
from rich.table   import Table
from rich         import box

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  SETTINGS
# ─────────────────────────────────────────────
CFG_FILE  = "core/custom.cfg"
MAX_RANGE = 5.0          # metres — clips range axis

CLI_PORT  = None         # e.g. "/dev/ttyACM0" or "COM3" — None = auto-detect
DATA_PORT = None         # e.g. "/dev/ttyACM1" or "COM4" — None = auto-detect

CMAP             = "inferno"
DISPLAY_LOW_PCT  = 40    # percentile floor for colour scaling
DISPLAY_HIGH_PCT = 99.5  # percentile ceiling for colour scaling
SMOOTH_GRID_SIZE = 250   # minimum pixel dimension after bilinear upscale

console = Console()


# ─────────────────────────────────────────────
#  BACKGROUND THREAD
#  Reads USB bytes, parses the RDHM TLV, does
#  the log-FFT-shift, and emits one numpy array
#  per frame.  No queues, no locks — the thread
#  owns all radar state; the main thread only
#  renders what it receives via the Qt signal.
# ─────────────────────────────────────────────
class RadarWorker(QThread):
    new_frame = pyqtSignal(np.ndarray)
    error     = pyqtSignal(str)

    def __init__(self, radar: RadarSensor, max_range_m: float):
        super().__init__()
        self.radar   = radar
        self.running = True

        cfg = radar.config
        self.num_range_bins = cfg.numRangeBins
        self.num_vel_bins   = cfg.numLoops
        self.max_bin        = min(int(max_range_m / cfg.rangeRes), cfg.numRangeBins)
        self._expected_size = self.num_range_bins * self.num_vel_bins

    def run(self):
        while self.running:
            try:
                raw_bytes = self.radar.read_raw_frame()

                if raw_bytes is None:
                    # No complete frame yet — yield CPU briefly and retry.
                    # This prevents a busy-loop that would starve the Qt thread.
                    time.sleep(0.001)
                    continue

                from core.parser import parse_standard_frame
                frame = parse_standard_frame(raw_bytes)

                if frame.get("RDHM") is None:
                    continue

                raw = frame["RDHM"]

                if raw.size != self._expected_size:
                    log.warning("Shape mismatch — got %d, expected %d. Dropping.",
                                raw.size, self._expected_size)
                    continue

                rd = raw.astype(np.float32).reshape(self.num_range_bins, self.num_vel_bins)
                rd = rd[:self.max_bin, :]

                # Centre zero-velocity and convert to dB for display
                display = 20.0 * np.log10(np.abs(np.fft.fftshift(rd, axes=1)) + 1e-6)
                self.new_frame.emit(display)

            except Exception as e:
                self.error.emit(str(e))

    def stop(self):
        self.running = False
        self.wait()


# ─────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────
class ViewerWindow(QMainWindow):

    def __init__(self, radar: RadarSensor):
        super().__init__()
        self.radar      = radar
        self._prev_time = time.time()
        self._fps       = 0.0

        self._build_plot(radar)
        self._precompute_zoom(radar)
        self._start_worker(radar)
        self._update_title()

    # ── Plot setup ────────────────────────────
    def _build_plot(self, radar):
        cfg = radar.config

        max_bin      = min(int(MAX_RANGE / cfg.rangeRes), cfg.numRangeBins)
        actual_max_m = max_bin * cfg.rangeRes
        dop_res      = cfg.dopRes

        self.plot = pg.PlotWidget(title="Live Range-Doppler Heatmap")
        self.plot.setLabel("left",   "Range",    units="m")
        self.plot.setLabel("bottom", "Velocity", units="m/s")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.addItem(pg.InfiniteLine(
            pos=0, angle=90,
            pen=pg.mkPen("w", style=Qt.PenStyle.DashLine, width=1),
        ))

        self.img = pg.ImageItem()
        self.img.setColorMap(pg.colormap.get(CMAP))
        self.plot.addItem(self.img)

        # Image rectangle is fixed for the lifetime of the window
        self._rect = pg.QtCore.QRectF(
            -cfg.dopMax - dop_res / 2.0,
            0,
            cfg.dopMax * 2.0,
            actual_max_m,
        )
        self.img.setRect(self._rect)
        self.setCentralWidget(self.plot)
        self.resize(900, 650)

    def _precompute_zoom(self, radar):
        """Zoom scales are constant — compute once, not per frame."""
        cfg      = radar.config
        src_rows = min(int(MAX_RANGE / cfg.rangeRes), cfg.numRangeBins)
        src_cols = cfg.numLoops

        self._zoom_y = max(SMOOTH_GRID_SIZE, src_rows) / src_rows
        self._zoom_x = max(SMOOTH_GRID_SIZE, src_cols) / src_cols

    # ── Worker ────────────────────────────────
    def _start_worker(self, radar):
        self.worker = RadarWorker(radar, MAX_RANGE)
        self.worker.new_frame.connect(self._on_frame)
        self.worker.error.connect(lambda e: log.error("Worker: %s", e))
        self.worker.start()

    # ── Frame callback (Qt main thread) ───────
    def _on_frame(self, matrix: np.ndarray):
        now             = time.time()
        dt              = now - self._prev_time
        self._fps       = 1.0 / dt if dt > 0 else 0.0
        self._prev_time = now

        smooth = ndimage.zoom(matrix, (self._zoom_y, self._zoom_x), order=1)

        lo = float(np.percentile(smooth, DISPLAY_LOW_PCT))
        hi = float(np.percentile(smooth, DISPLAY_HIGH_PCT))
        if lo >= hi:
            hi = lo + 0.1

        self.img.setImage(smooth, autoLevels=False, levels=(lo, hi))
        self.img.setRect(self._rect)
        self._update_title()

    def _update_title(self):
        self.setWindowTitle(f"OST Radar  [LIVE]  FPS: {self._fps:.1f}")

    # ── Shutdown ──────────────────────────────
    def closeEvent(self, event):
        # Stop the worker loop first
        self.worker.running = False

        # Close hardware — this breaks any blocked serial.read() and lets
        # the worker's read_raw_frame() return None so the thread exits cleanly
        self.radar.close()

        # Now safe to join
        self.worker.wait()
        event.accept()


# ─────────────────────────────────────────────
#  STARTUP HELPERS
# ─────────────────────────────────────────────
def connect_radar() -> RadarSensor:
    cli, data = CLI_PORT, DATA_PORT
    if cli is None or data is None:
        console.print("[yellow]Auto-detecting ports...[/yellow]")
        found_cli, found_data = RadarSensor.find_ti_ports()
        cli  = cli  or found_cli
        data = data or found_data
    if not cli or not data:
        console.print(
            "[bold red]ERROR:[/bold red] Ports not found.\n"
            "Set CLI_PORT / DATA_PORT manually at the top of viewer.py.\n"
            "Linux:   /dev/ttyACM0, /dev/ttyACM1\n"
            "Windows: COM3, COM4"
        )
        sys.exit(1)
    console.print(f"CLI: [cyan]{cli}[/cyan]  DATA: [cyan]{data}[/cyan]")
    radar = RadarSensor(cli, data, CFG_FILE)
    radar.connect_and_configure()
    return radar


def print_radar_info(radar: RadarSensor):
    t = Table(title="Radar Config", box=box.ROUNDED, show_header=False)
    t.add_column("", style="dim")
    t.add_column("", style="cyan bold")
    t.add_row("Max Range",  f"{MAX_RANGE} m")
    for k, v in radar.config.summary().items():
        t.add_row(k, str(v))
    console.print(t)


def main():
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s  %(name)s  %(message)s")

    radar = connect_radar()

    app = QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)

    print_radar_info(radar)

    window = ViewerWindow(radar)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()