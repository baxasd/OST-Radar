import sys
import time
import json
import numpy as np
import pyqtgraph as pg
import serial.tools.list_ports
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

try:
    from ti_radar.sensor import RadarSensor
    RADAR_LIB_AVAILABLE = True
except ImportError:
    RADAR_LIB_AVAILABLE = False

CFG_FILE = "ti_radar/profile_3d_isk.cfg"

# --- Shared UI Constants ---
WINDOW_WIDTH, WINDOW_HEIGHT = 1000, 750
STYLESHEET = """
QMainWindow { background-color: #1e1e1e; color: #ffffff; }
QFrame#Sidebar { background-color: #252526; border-right: 1px solid #333; }
QLabel { color: #e0e0e0; font-size: 11px; }
QLineEdit, QDoubleSpinBox, QComboBox { background-color: #3c3c3c; border: 1px solid #555; color: white; padding: 4px; border-radius: 3px; }
QLineEdit:disabled, QDoubleSpinBox:disabled, QComboBox:disabled { background-color: #2b2b2b; color: #888; }
QPushButton { background-color: #0e639c; color: white; border: none; border-radius: 4px; font-weight: bold; }
QPushButton:hover { background-color: #1177bb; }
QPushButton:disabled { background-color: #333333; color: #666666; }
QPushButton#RecBtn[recording="true"] { background-color: #d32f2f; }
"""

class RadarWorker(QThread):
    new_heatmap = pyqtSignal(np.ndarray)
    
    def __init__(self, radar_sensor, max_range_meters):
        super().__init__()
        self.radar = radar_sensor
        self.is_running = True
        self.is_recording = False
        
        self.recorded_frames = []
        self.recorded_timestamps = []
        
        self.num_range_bins = self.radar.config.ADCsamples
        self.num_vel_bins = self.radar.config.numLoops
        self.max_bin = min(int(max_range_meters / self.radar.config.rangeRes), self.num_range_bins)

    def run(self):
        while self.is_running:
            try:
                frame = self.radar.get_next_frame()
                if not frame or frame.get("RDHM") is None: continue
                
                raw_rdhm = frame["RDHM"]
                if raw_rdhm.size == (self.num_range_bins * self.num_vel_bins):
                    # 1. Reshape and crop to desired max distance
                    rd_matrix = np.array(raw_rdhm, dtype=np.float32).reshape(self.num_range_bins, self.num_vel_bins)
                    rd_matrix_cropped = rd_matrix[:self.max_bin, :]
                    
                    # 2. Store pure physics data if recording
                    if self.is_recording:
                        self.recorded_frames.append(np.copy(rd_matrix_cropped))
                        self.recorded_timestamps.append(time.time())

                    # 3. Prepare data for UI (Shift 0 velocity to center, convert to Log scale)
                    ui_matrix = np.fft.fftshift(rd_matrix_cropped, axes=1)
                    ui_matrix = 20 * np.log10(np.abs(ui_matrix) + 1e-6)
                    self.new_heatmap.emit(ui_matrix)

            except Exception as e:
                print(f"Worker Error: {e}")

    def toggle_recording(self, state):
        self.is_recording = state
        if state:
            self.recorded_frames.clear()
            self.recorded_timestamps.clear()
        return self.recorded_frames, self.recorded_timestamps

    def stop(self):
        self.is_running = False
        self.wait()


class RadarRecorderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OST Radar Recorder")
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet(STYLESHEET)
        
        self.is_recording = False
        self.frame_count = 0
        self.prev_time = time.time()
        self.worker = None
        
        self._build_ui()
        self._auto_populate_ports()

    # ================= UI SETUP =================
    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Build Panels
        main_layout.addWidget(self._create_sidebar())
        main_layout.addWidget(self._create_video_area())

    def _create_sidebar(self):
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(260)
        layout = QVBoxLayout(sidebar)
        layout.setSpacing(6)

        title = QLabel("OST Radar Recorder")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        # Connect Area
        self.cmb_cli = QComboBox()
        self.cmb_cli.setEditable(True)
        self.cmb_data = QComboBox()
        self.cmb_data.setEditable(True)
        
        self.btn_connect = QPushButton("Connect Radar")
        self.btn_connect.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_connect.clicked.connect(self._handle_connection)
        
        layout.addWidget(QLabel("CLI Port"))
        layout.addWidget(self.cmb_cli)
        layout.addWidget(QLabel("DATA Port"))
        layout.addWidget(self.cmb_data)
        layout.addWidget(self.btn_connect)
        layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine, styleSheet="color: #444; margin: 5px 0;"))

        # Config Area
        self.spin_max_range = QDoubleSpinBox()
        self.spin_max_range.setValue(5.0)
        self.spin_max_range.setSingleStep(0.5)
        self.spin_max_range.setRange(1.0, 20.0)
        
        self.cmb_cmap = QComboBox()
        self.cmb_cmap.addItems(['inferno', 'magma', 'viridis', 'plasma', 'turbo'])
        self.cmb_cmap.currentTextChanged.connect(lambda c: self.image_item.setColorMap(pg.colormap.get(c)))

        layout.addWidget(QLabel("Max Range Crop (Meters)"))
        layout.addWidget(self.spin_max_range)
        layout.addWidget(QLabel("Heatmap Color Map"))
        layout.addWidget(self.cmb_cmap)
        layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine, styleSheet="color: #444; margin: 5px 0;"))

        # Metadata Area
        self.inp_subject = QLineEdit()
        self.inp_subject.setPlaceholderText("e.g. S01")
        self.inp_activity = QLineEdit()
        self.inp_activity.setPlaceholderText("e.g. Running")
        self.inp_temp = QLineEdit()
        self.inp_temp.setPlaceholderText("24.0")
        
        layout.addWidget(QLabel("Subject ID *")); layout.addWidget(self.inp_subject)
        layout.addWidget(QLabel("Activity *")); layout.addWidget(self.inp_activity)
        layout.addWidget(QLabel("Temp (°C) *")); layout.addWidget(self.inp_temp)

        # Stats Area
        self.lbl_fps = QLabel("FPS: 0.0")
        self.lbl_fps.setFont(QFont("Consolas", 11))
        
        self.lbl_frames = QLabel("Frames: 0")
        self.lbl_frames.setFont(QFont("Consolas", 10))
        self.lbl_frames.setStyleSheet("color: gray;")
        
        self.lbl_error = QLabel("")
        self.lbl_error.setStyleSheet("color: #ff5555; font-size: 10px;")
        
        layout.addWidget(self.lbl_fps)
        layout.addWidget(self.lbl_frames)
        layout.addWidget(self.lbl_error)
        layout.addStretch()

        # Record Button
        self.btn_record = QPushButton("Record")
        self.btn_record.setObjectName("RecBtn")
        self.btn_record.setEnabled(False)
        self.btn_record.setFixedHeight(40)
        self.btn_record.setProperty("recording", False)
        self.btn_record.clicked.connect(self._toggle_recording)
        layout.addWidget(self.btn_record)

        return sidebar

    def _create_video_area(self):
        container = QWidget()
        container.setStyleSheet("background-color: black;")
        layout = QVBoxLayout(container)
        
        self.lbl_video = QLabel("Radar Disconnected.")
        self.lbl_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_video)

        pg.setConfigOptions(imageAxisOrder='row-major')
        self.plot_widget = pg.PlotWidget(title="Live Range-Doppler Heatmap")
        self.plot_widget.setLabel('left', 'Range', units='m')
        self.plot_widget.setLabel('bottom', 'Velocity', units='m/s')
        
        self.plot_widget.addItem(pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('w', style=Qt.PenStyle.DashLine, alpha=100)))
        
        self.image_item = pg.ImageItem()
        self.image_item.setColorMap(pg.colormap.get('inferno'))
        self.plot_widget.addItem(self.image_item)
        self.plot_widget.hide() 
        layout.addWidget(self.plot_widget)
        
        return container
    # ================= LOGIC =================
    def _auto_populate_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.cmb_cli.addItems(ports); self.cmb_data.addItems(ports)
        if RADAR_LIB_AVAILABLE:
            cli, data = RadarSensor.find_ti_ports()
            if cli: self.cmb_cli.setCurrentText(cli)
            if data: self.cmb_data.setCurrentText(data)

    def _handle_connection(self):
        cli, data = self.cmb_cli.currentText().strip(), self.cmb_data.currentText().strip()
        if not cli or not data: return self.lbl_error.setText("⚠ Select valid ports.")

        self.btn_connect.setEnabled(False)
        self.lbl_video.setText("Initializing...")
        QApplication.processEvents() 

        try:
            radar = RadarSensor(cli, data, CFG_FILE)
            radar.connect_and_configure()
            
            # Lock hardware UI state
            for w in [self.cmb_cli, self.cmb_data, self.spin_max_range]: w.setEnabled(False)
            self.btn_connect.setText("Connected")
            self.btn_connect.setStyleSheet("background-color: #2e7d32; color: white;")
            self.btn_record.setEnabled(True)
            self.lbl_error.setText("")
            
            self._start_worker(radar)

        except Exception as e:
            self.btn_connect.setEnabled(True)
            QMessageBox.critical(self, "Connection Error", str(e))

    def _start_worker(self, radar):
        max_m = self.spin_max_range.value()
        max_bin = min(int(max_m / radar.config.rangeRes), radar.config.ADCsamples)
        actual_max_range = max_bin * radar.config.rangeRes
        
        max_v = (radar.config.numLoops / 2.0) * radar.config.dopRes
        dop_res = radar.config.dopRes
        
        # Center bounding box
        self.rect = QRectF(-max_v - (dop_res / 2.0), 0, max_v * 2, actual_max_range)
        self.image_item.setRect(self.rect)
        
        self.lbl_video.hide()
        self.plot_widget.show()
        
        self.worker = RadarWorker(radar, max_m)
        self.worker.new_heatmap.connect(self._on_new_frame)
        self.worker.start()

    def _on_new_frame(self, matrix):
        # Update FPS safely
        now = time.time()
        self.lbl_fps.setText(f"FPS: {1.0 / (now - self.prev_time):.1f}" if now > self.prev_time else "FPS: 0.0")
        self.prev_time = now

        # Update Visuals
        self.image_item.setImage(matrix, autoLevels=False)
        self.image_item.setRect(self.rect) 
        self.image_item.setLevels((np.percentile(matrix, 40), np.percentile(matrix, 99.5)))

        if self.is_recording:
            self.frame_count += 1
            self.lbl_frames.setText(f"Frames: {self.frame_count}")
            self.lbl_frames.setStyleSheet("color: #ff5555; font-weight: bold;")

    def _toggle_recording(self):
        s, a, t = self.inp_subject.text().strip(), self.inp_activity.text().strip(), self.inp_temp.text().strip()
        
        if not self.is_recording:
            if not all([s, a, t]): return self.lbl_error.setText("⚠ MISSING FIELDS")
            
            self.is_recording = True
            self.frame_count = 0
            self.worker.toggle_recording(True)
            
            for w in [self.inp_subject, self.inp_activity, self.inp_temp]: w.setEnabled(False)
            self._update_btn_style(True, "STOP")
            
        else:
            self.is_recording = False
            frames, timestamps = self.worker.toggle_recording(False)
            
            for w in [self.inp_subject, self.inp_activity, self.inp_temp]: w.setEnabled(True)
            self._update_btn_style(False, "RECORD")
            self.lbl_frames.setStyleSheet("color: gray;")
            
            if frames: self._save_to_parquet(s, a, t, frames, timestamps)

    def _update_btn_style(self, is_rec, text):
        self.btn_record.setText(text)
        self.btn_record.setProperty("recording", is_rec)
        self.btn_record.style().unpolish(self.btn_record)
        self.btn_record.style().polish(self.btn_record)

    def _save_to_parquet(self, subject, activity, temp, frames, timestamps):
        """Migrated from .npz to Apache Parquet for high-performance columnar storage"""
        filename = f"radar_{subject}_{activity}_{int(time.time())}.parquet"
        
        # Serialize metadata and frame dimensions directly into the Parquet schema
        metadata = {
            "subject": subject, "activity": activity, "temp": temp,
            "range_res": self.worker.radar.config.rangeRes,
            "doppler_res": self.worker.radar.config.dopRes,
            "dop_max": self.worker.radar.config.dopMax
        }
        
        # Flattens raw physics arrays into bytes for columnar storage
        df = pd.DataFrame({
            "timestamp": timestamps,
            "frame_data": [f.tobytes() for f in frames]
        })
        
        table = pa.Table.from_pandas(df)
        custom_schema = {
            b"radar_metadata": json.dumps(metadata).encode(),
            b"frame_shape": json.dumps(frames[0].shape).encode(),
            b"frame_dtype": str(frames[0].dtype).encode()
        }
        
        table = table.replace_schema_metadata({**(table.schema.metadata or {}), **custom_schema})
        pq.write_table(table, filename)
        
        self.lbl_error.setStyleSheet("color: #4CAF50;")
        self.lbl_error.setText(f"Saved: {filename}")

    def closeEvent(self, event):
        if self.worker: self.worker.stop()
        if self.worker and self.worker.radar: self.worker.radar.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RadarRecorderApp() 
    window.show()
    sys.exit(app.exec())