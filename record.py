"""
OST Radar Recorder — Research Tool
====================================
Edit the SESSION and RADAR SETTINGS constants below, then run:
    python record.py

Controls (in the spectrogram window):
    R  — Start / Stop recording
    Q  — Quit
"""

import sys
import time
import json
import threading
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import pyqtgraph as pg

from PyQt6.QtCore    import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtGui     import QShortcut, QKeySequence

try:
    from ti_radar.sensor import RadarSensor
    RADAR_AVAILABLE = True
except ImportError:
    RADAR_AVAILABLE = False

from rich.console import Console
from rich.table   import Table
from rich         import box

# ─────────────────────────────────────────────
#  SESSION  — edit before each run
# ─────────────────────────────────────────────
SUBJECT  = "S01"
ACTIVITY = "running"
TEMP     = 24.0         # °C

# ─────────────────────────────────────────────
#  RADAR SETTINGS  — change only when hardware/profile changes
# ─────────────────────────────────────────────
CFG_FILE  = "ti_radar/profile_3d_ISK.cfg"
MAX_RANGE = 5.0         # metres — crop beyond this distance
CLI_PORT  = "/dev/ttyUSB0"        # e.g. "COM3"  — None = auto-detect
DATA_PORT = "/dev/ttyUSB1"        # e.g. "COM4"  — None = auto-detect

# ─────────────────────────────────────────────
#  DISPLAY SETTINGS
# ─────────────────────────────────────────────
CMAP             = "inferno"
DISPLAY_LOW_PCT  = 40    # raise if spectrogram looks noisy
DISPLAY_HIGH_PCT = 99.5  # lower if spectrogram looks washed out

console = Console()


# ─────────────────────────────────────────────
#  WORKER THREAD
#  All serial I/O runs here so it never blocks
#  the Qt render loop.
# ─────────────────────────────────────────────
class RadarWorker(QThread):
    new_frame = pyqtSignal(np.ndarray)
    error     = pyqtSignal(str)

    def __init__(self, radar: "RadarSensor", max_range_m: float):
        super().__init__()
        self.radar     = radar
        self.running   = True
        self.recording = False

        # Lock guards _frames/_timestamps against concurrent access from
        # the main thread (start/stop_recording) and the worker (appending).
        self._lock       = threading.Lock()
        self._frames:     list[np.ndarray] = []
        self._timestamps: list[float]      = []

        cfg = radar.config
        self.num_range_bins = cfg.ADCsamples
        self.num_vel_bins   = cfg.numLoops
        self.max_bin        = min(int(max_range_m / cfg.rangeRes), cfg.ADCsamples)

    def start_recording(self):
        with self._lock:                         # clear while holding lock
            self._frames.clear()
            self._timestamps.clear()
        self.recording = True                    # set after clear: no frames lost

    def stop_recording(self) -> tuple[list, list]:
        self.recording = False                   # stop appends before acquiring lock
        with self._lock:
            return list(self._frames), list(self._timestamps)

    def stop(self):
        self.running = False
        self.wait()

    def run(self):
        while self.running:
            try:
                frame = self.radar.get_next_frame()
                if not frame or frame.get("RDHM") is None:
                    continue

                raw = frame["RDHM"]
                if raw.size != self.num_range_bins * self.num_vel_bins:
                    continue

                rd = raw.astype(np.float32).reshape(self.num_range_bins, self.num_vel_bins)
                rd = rd[:self.max_bin, :]

                # Timestamp immediately after frame arrives — before any numpy work.
                # Late timestamps skew inter-frame intervals and corrupt cadence FFT.
                ts = time.time()

                if self.recording:
                    with self._lock:
                        self._frames.append(rd.copy())
                        self._timestamps.append(ts)

                # Display only: fftshift centres 0 m/s, log scale for visibility.
                # Raw frames saved above are never fftshifted — keep physics clean.
                display = 20.0 * np.log10(np.abs(np.fft.fftshift(rd, axes=1)) + 1e-6)
                self.new_frame.emit(display)

            except Exception as e:
                self.error.emit(str(e))


# ─────────────────────────────────────────────
#  MAIN WINDOW  — spectrogram plot only
# ─────────────────────────────────────────────
class RecorderWindow(QMainWindow):
    def __init__(self, radar: "RadarSensor"):
        super().__init__()
        self.radar        = radar
        self.recording    = False
        self._prev_time   = time.time()
        self._fps         = 0.0
        self._frame_count = 0
        self._rec_frames  = 0

        self._build_plot(radar)
        self._bind_keys()
        self._start_worker(radar)
        self._update_title()

    def _build_plot(self, radar):
        cfg = radar.config

        # max_bin must match the worker exactly — any independent rounding
        # difference misaligns the displayed range axis against the actual data.
        max_bin      = min(int(MAX_RANGE / cfg.rangeRes), cfg.ADCsamples)
        actual_max_m = max_bin * cfg.rangeRes

        # Velocity spans ±dopMax after fftshift; half-bin offset aligns
        # pixel edges with axis tick positions correctly.
        max_v   = (cfg.numLoops / 2.0) * cfg.dopRes
        dop_res = cfg.dopRes

        self.plot = pg.PlotWidget(title="Live Micro-Doppler Spectrogram")
        self.plot.setLabel("left",   "Range",    units="m")
        self.plot.setLabel("bottom", "Velocity", units="m/s")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.addItem(pg.InfiniteLine(
            pos=0, angle=90,
            pen=pg.mkPen("w", style=Qt.PenStyle.DashLine, width=1)
        ))

        self.img = pg.ImageItem()
        self.img.setColorMap(pg.colormap.get(CMAP))
        self.plot.addItem(self.img)

        # Physical rect: x = velocity axis, y = range axis
        self._rect = pg.QtCore.QRectF(
            -max_v - dop_res / 2.0,  # left  (most negative velocity)
            0,                        # bottom (0 m)
            max_v * 2.0,              # width
            actual_max_m              # height
        )
        self.img.setRect(self._rect)
        self.setCentralWidget(self.plot)
        self.resize(900, 650)

    def _bind_keys(self):
        for key in ("R", "r"):
            QShortcut(QKeySequence(key), self).activated.connect(self._toggle_recording)
        for key in ("Q", "q"):
            QShortcut(QKeySequence(key), self).activated.connect(self.close)

    def _start_worker(self, radar):
        self.worker = RadarWorker(radar, MAX_RANGE)
        self.worker.new_frame.connect(self._on_frame)
        self.worker.error.connect(lambda e: console.print(f"[red]Worker:[/red] {e}"))
        self.worker.start()

    def _on_frame(self, matrix: np.ndarray):
        now = time.time()
        dt  = now - self._prev_time
        self._fps         = 1.0 / dt if dt > 0 else 0.0
        self._prev_time   = now
        self._frame_count += 1
        if self.recording:
            self._rec_frames += 1

        lo = float(np.percentile(matrix, DISPLAY_LOW_PCT))
        hi = float(np.percentile(matrix, DISPLAY_HIGH_PCT))
        self.img.setImage(matrix, autoLevels=False, levels=(lo, hi))
        self.img.setRect(self._rect)  # pyqtgraph can silently drift without this
        self._update_title()

    def _update_title(self):
        state = "[REC ●]" if self.recording else "[LIVE]"
        rec   = f"  Rec: {self._rec_frames}" if self.recording else ""
        self.setWindowTitle(
            f"OST Radar  {state}  FPS: {self._fps:.1f}  "
            f"Frames: {self._frame_count}{rec}  |  [R] Record   [Q] Quit"
        )

    def _toggle_recording(self):
        if not self.recording:
            self.recording   = True
            self._rec_frames = 0
            self.worker.start_recording()
            console.print(
                f"\n[bold red]● REC[/bold red]  "
                f"[cyan]{SUBJECT}[/cyan] / [cyan]{ACTIVITY}[/cyan]"
            )
        else:
            self.recording = False
            frames, timestamps = self.worker.stop_recording()
            console.print(f"[bold green]■ STOP[/bold green]  {len(frames)} frames")
            if frames:
                self._save(frames, timestamps)

    def _save(self, frames: list[np.ndarray], timestamps: list[float]):
        cfg      = self.radar.config
        filename = f"radar_{SUBJECT}_{ACTIVITY}_{int(time.time())}.parquet"

        meta = {
            "subject":     SUBJECT,
            "activity":    ACTIVITY,
            "temp":        TEMP,
            "range_res":   cfg.rangeRes,
            "doppler_res": cfg.dopRes,
            "dop_max":     cfg.dopMax,
        }

        df    = pd.DataFrame({
            "timestamp":  timestamps,
            "frame_data": [f.tobytes() for f in frames],
        })
        table = pa.Table.from_pandas(df)
        table = table.replace_schema_metadata({
            **(table.schema.metadata or {}),
            b"radar_metadata": json.dumps(meta).encode(),
            b"frame_shape":    json.dumps(list(frames[0].shape)).encode(),
            b"frame_dtype":    frames[0].dtype.str.encode(),  # e.g. b"<f4"
        })
        pq.write_table(table, filename)
        console.print(f"[bold green]✓[/bold green] {filename}")

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
        if self.radar:
            self.radar.close()
        event.accept()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
def connect_radar() -> "RadarSensor":
    cli, data = CLI_PORT, DATA_PORT
    if cli is None or data is None:
        console.print("[yellow]Auto-detecting ports...[/yellow]")
        found_cli, found_data = RadarSensor.find_ti_ports()
        cli  = cli  or found_cli
        data = data or found_data
    if not cli or not data:
        console.print(
            "[bold red]ERROR:[/bold red] Ports not found. "
            "Set CLI_PORT / DATA_PORT manually."
        )
        sys.exit(1)
    console.print(f"CLI: [cyan]{cli}[/cyan]  DATA: [cyan]{data}[/cyan]")
    radar = RadarSensor(cli, data, CFG_FILE)
    radar.connect_and_configure()
    return radar


def print_session_info(radar):
    t = Table(title="Session", box=box.ROUNDED, show_header=False)
    t.add_column("", style="dim")
    t.add_column("", style="cyan bold")
    t.add_row("Subject",      SUBJECT)
    t.add_row("Activity",     ACTIVITY)
    t.add_row("Temperature",  f"{TEMP} °C")
    t.add_row("Max Range",    f"{MAX_RANGE} m")
    for k, v in radar.config.summary().items():
        t.add_row(k, v)
    t.add_row("", "")
    t.add_row("Controls", "[R] Record / Stop   [Q] Quit")
    console.print(t)


def main():
    if not RADAR_AVAILABLE:
        console.print("[bold red]ERROR:[/bold red] ti_radar not found. Run from project root.")
        sys.exit(1)

    radar = connect_radar()

    app = QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)

    print_session_info(radar)

    window = RecorderWindow(radar)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()