import sys
import os
import time
import datetime
import traceback
import numpy as np
import pyqtgraph as pg

# UI Imports
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

# Radar Backend
try:
    from ti_radar import RadarSensor
    RADAR_LIB_AVAILABLE = True
except ImportError:
    RADAR_LIB_AVAILABLE = False

# ===================== USER CONFIG =====================
CLI_PORT = "COM10"
DATA_PORT = "COM11"
CFG_FILE = "ti_radar/chirp_config/profile_3d_isk.cfg"
MAX_RANGE_METERS = 5.0 
# ======================================================

WINDOW_WIDTH, WINDOW_HEIGHT = 1000, 600
MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT = 800, 500
TEXT_DIM = "#888888"
VERSION = "v1.3.0-Pure-Heatmap"

STYLESHEET = """
QMainWindow { background-color: #1e1e1e; color: #ffffff; }
QFrame#Sidebar { background-color: #252526; border-right: 1px solid #333; }
QLabel { color: #e0e0e0; }
QLineEdit { background-color: #3c3c3c; border: 1px solid #555; color: white; padding: 5px; border-radius: 3px; }
QLineEdit:disabled { background-color: #2b2b2b; color: #888; }
QComboBox { background-color: #3c3c3c; border: 1px solid #555; color: white; padding: 5px; border-radius: 3px; }
QComboBox:disabled { background-color: #2b2b2b; color: #888; }
QPushButton { background-color: #0e639c; color: white; border: none; border-radius: 4px; font-weight: bold; }
QPushButton:hover { background-color: #1177bb; }
QPushButton#RecBtn[recording="true"] { background-color: #d32f2f; }
QPushButton#RecBtn[recording="true"]:hover { background-color: #f44336; }
"""

class RadarWorker(QThread):
    # Emits the full 2D matrix: (Range, Velocity)
    new_heatmap = pyqtSignal(np.ndarray)
    
    def __init__(self, radar_sensor):
        super().__init__()
        self.radar = radar_sensor
        self.is_running = True
        
        # Recording state
        self.is_recording = False
        self.recorded_session_rdhm = []
        self.recorded_session_timestamps = []
        
        # Matrix Dimensions
        self.num_range_bins = self.radar.config.ADCsamples
        self.num_vel_bins = self.radar.config.numLoops

        self.num_range_bins = self.radar.config.ADCsamples
        self.num_vel_bins = self.radar.config.numLoops
        
        # Calculate exactly which row index equals MAX_RANGE_METERS
        self.max_bin = min(int(MAX_RANGE_METERS / self.radar.config.rangeRes), self.num_range_bins)

    def run(self):
        while self.is_running:
            try:
                frame = self.radar.get_next_frame()
                if not frame or frame.get("RDHM") is None: 
                    continue
                
                raw_rdhm = frame["RDHM"]
                
                if raw_rdhm.size == (self.num_range_bins * self.num_vel_bins):
                    # 1. Reshape to 2D first
                    rd_matrix = np.array(raw_rdhm, dtype=np.float32).reshape(self.num_range_bins, self.num_vel_bins)
                    
                    # 2. Crop the top off the matrix!
                    rd_matrix_cropped = rd_matrix[:self.max_bin, :]
                    
                    # 3. Save the *cropped* raw data (Saves 50%+ RAM and disk space)
                    if self.is_recording:
                        self.recorded_session_rdhm.append(np.copy(rd_matrix_cropped))
                        self.recorded_session_timestamps.append(time.time())

                    # 4. Process Data for UI Heatmap
                    # Use the cropped matrix for the FFT shift and log conversion
                    ui_matrix = np.fft.fftshift(rd_matrix_cropped, axes=1)
                    ui_matrix = 20 * np.log10(np.abs(ui_matrix) + 1e-6)
                    
                    self.new_heatmap.emit(ui_matrix)

            except Exception as e:
                print(f"Worker Error: {e}")


    def start_recording(self):
        self.recorded_session_rdhm = []
        self.recorded_session_timestamps = []
        self.is_recording = True

    def stop_recording(self):
        self.is_recording = False
        return self.recorded_session_rdhm, self.recorded_session_timestamps

    def stop(self):
        self.is_running = False
        self.wait()


class RadarRecorderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OST Radar Recorder")
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setMinimumSize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
        
        self.is_recording = False
        self.frame_count = 0
        self.prev_time = 0
        self.radar = None
        self.worker = None
        
        self._init_ui()
        self.setStyleSheet(STYLESHEET)
        QTimer.singleShot(100, self.load_heavy_components)

    def load_heavy_components(self):
        try:
            self.lbl_video.setText("Initializing Radar Sensor...")
            QApplication.processEvents() 

            if not RADAR_LIB_AVAILABLE:
                raise Exception("ti_radar library not found in directory.")

            if not self.connect_radar():
                self.close()
                return
            
            self.lbl_video.hide()
            self.plot_widget.show()

            # Calculate the exact physical range of the cropped bins
            max_bin = min(int(MAX_RANGE_METERS / self.radar.config.rangeRes), self.radar.config.ADCsamples)
            actual_max_range = max_bin * self.radar.config.rangeRes
            
            max_vel = (self.radar.config.numLoops / 2.0) * self.radar.config.dopRes
            dop_res = self.radar.config.dopRes
            
            # Use actual_max_range for the bounding box height
            self.rect = QRectF(-max_vel - (dop_res / 2.0), 0, max_vel * 2, actual_max_range)
            self.image_item.setRect(self.rect)
            
            self.worker = RadarWorker(self.radar)
            self.worker.new_heatmap.connect(self.update_loop)
            self.worker.start()

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Startup Error", f"Failed to initialize:\n{e}")

    def connect_radar(self):
        while True:
            try:
                self.radar = RadarSensor(CLI_PORT, DATA_PORT, CFG_FILE)
                self.radar.connect_and_configure()
                return True
            except Exception as e:
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Icon.Critical)
                msg.setWindowTitle("Radar Error")
                msg.setText("Radar Sensor not detected.")
                msg.setInformativeText(f"Error: {str(e)}\n\nPlease connect radar and retry.")
                msg.setStandardButtons(QMessageBox.StandardButton.Retry | QMessageBox.StandardButton.Cancel)
                if msg.exec() == QMessageBox.StandardButton.Cancel:
                    return False

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # SIDEBAR
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(220)
        side_layout = QVBoxLayout(self.sidebar)
        side_layout.setContentsMargins(15, 20, 15, 30)
        side_layout.setSpacing(5)

        title = QLabel("OST Radar Recorder")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        side_layout.addWidget(title)
        side_layout.addSpacing(10)

        self.inp_subject = self._add_input(side_layout, "Subject ID *", "e.g. S01")
        self.inp_activity = self._add_input(side_layout, "Activity *", "e.g. Running")
        self.inp_temp = self._add_input(side_layout, "Room Temp (°C) *", "24.0")

        side_layout.addWidget(QLabel("Date"))
        self.lbl_date = QLabel(datetime.datetime.now().strftime("%Y-%m-%d"))
        self.lbl_date.setStyleSheet("color: gray;")
        side_layout.addWidget(self.lbl_date)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #444;")
        side_layout.addWidget(line)

        side_layout.addWidget(QLabel("Radar Config"))
        self.cmb_model = QComboBox()
        self.cmb_model.addItems([CFG_FILE.split('/')[-1]])
        side_layout.addWidget(self.cmb_model)

        side_layout.addSpacing(10)
        self.lbl_fps = QLabel("FPS: 00.0")
        self.lbl_fps.setFont(QFont("Consolas", 12))
        side_layout.addWidget(self.lbl_fps)
        
        self.lbl_frames = QLabel("Frames: 0")
        self.lbl_frames.setFont(QFont("Consolas", 10))
        self.lbl_frames.setStyleSheet("color: gray;")
        side_layout.addWidget(self.lbl_frames)

        self.lbl_error = QLabel("")
        self.lbl_error.setStyleSheet("color: #ff5555; font-size: 10px;")
        side_layout.addWidget(self.lbl_error)

        self.btn_record = QPushButton("Record")
        self.btn_record.setObjectName("RecBtn")
        self.btn_record.setFixedHeight(40)
        self.btn_record.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_record.clicked.connect(self.toggle_recording)
        self.btn_record.setProperty("recording", False)
        side_layout.addWidget(self.btn_record)

        l_ver = QLabel(VERSION)
        l_ver.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px; border: none; margin-top: 15px;")
        l_ver.setAlignment(Qt.AlignmentFlag.AlignLeft)
        side_layout.addWidget(l_ver)

        main_layout.addWidget(self.sidebar)

        # 2D HEATMAP AREA
        self.video_container = QWidget()
        self.video_container.setStyleSheet("background-color: black;")
        video_layout = QVBoxLayout(self.video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        
        self.lbl_video = QLabel("Initializing...")
        self.lbl_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_layout.addWidget(self.lbl_video)

        # PyQtGraph Image Setup
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.plot_widget = pg.PlotWidget(title="Live Range-Doppler Heatmap")
        self.plot_widget.setLabel('left', 'Range (Distance)', units='m')
        self.plot_widget.setLabel('bottom', 'Velocity (Speed)', units='m/s')
        
        # Add a center line to easily see 0 m/s (Stationary objects)
        v_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('w', style=Qt.PenStyle.DashLine, alpha=100))
        self.plot_widget.addItem(v_line)
        
        self.image_item = pg.ImageItem()
        self.image_item.setColorMap(pg.colormap.get('inferno'))
        self.plot_widget.addItem(self.image_item)
        self.plot_widget.hide() 

        video_layout.addWidget(self.plot_widget)
        main_layout.addWidget(self.video_container)

    def _add_input(self, layout, label, placeholder):
        layout.addWidget(QLabel(label))
        inp = QLineEdit()
        inp.setPlaceholderText(placeholder)
        layout.addWidget(inp)
        return inp

    def toggle_recording(self):
        if not self.worker: return 

        # FIX: Grab the text values at the very beginning of the function
        s = self.inp_subject.text().strip()
        a = self.inp_activity.text().strip()
        t = self.inp_temp.text().strip()

        if not self.is_recording:
            if not all([s, a, t]):
                self.lbl_error.setText("⚠ MISSING FIELDS")
                return
            self.lbl_error.setText("")
            
            self.is_recording = True
            self.frame_count = 0
            self.worker.start_recording()
            self._set_inputs_enabled(False)
            
            self.btn_record.setText("STOP")
            self.btn_record.setProperty("recording", True)
            self.btn_record.style().unpolish(self.btn_record)
            self.btn_record.style().polish(self.btn_record)
            
            self.plot_widget.setTitle("<span style='color: red;'>RECORDING IN PROGRESS...</span>")
            
        else:
            self.is_recording = False
            rdhm_data, timestamps = self.worker.stop_recording()
            
            self._set_inputs_enabled(True)
            self.lbl_frames.setStyleSheet("color: gray;")
            self.btn_record.setText("RECORD")
            self.btn_record.setProperty("recording", False)
            self.btn_record.style().unpolish(self.btn_record)
            self.btn_record.style().polish(self.btn_record)
            self.plot_widget.setTitle("Live Range-Doppler Heatmap")
            
            if len(rdhm_data) > 0:
                # s, a, and t are now safely accessible here!
                filename = f"radar_{s}_{a}_{int(time.time())}.npz"
                np.savez_compressed(
                    filename,
                    rdhm=np.array(rdhm_data), 
                    timestamps=np.array(timestamps),
                    metadata={
                        "subject": s, 
                        "activity": a, 
                        "temp": t,
                        "range_res": self.radar.config.rangeRes,
                        "doppler_res": self.radar.config.dopRes,
                        "dop_max": self.radar.config.dopMax
                    }
                )
                self.lbl_error.setStyleSheet("color: #4CAF50;")
                self.lbl_error.setText(f"Saved: {filename}")
    def _set_inputs_enabled(self, enabled):
        for w in [self.inp_subject, self.inp_activity, self.inp_temp, self.cmb_model]:
            w.setEnabled(enabled)

    def update_loop(self, heatmap_matrix):
        # Calculate FPS
        t = time.time()
        dt = t - self.prev_time
        fps = 1.0 / dt if dt > 0 else 0
        self.prev_time = t
        self.lbl_fps.setText(f"FPS: {fps:.1f}")

        # Set the single 2D frame
        self.image_item.setImage(heatmap_matrix, autoLevels=False)
        
        # Fix: Reapply bounding box
        if hasattr(self, 'rect'):
            self.image_item.setRect(self.rect)
        
        # Dynamic contrast: Hide the bottom 40% noise so moving feet pop out
        vmin = np.percentile(heatmap_matrix, 40)
        vmax = np.percentile(heatmap_matrix, 99.5)
        self.image_item.setLevels((vmin, vmax))

        if self.is_recording:
            self.frame_count += 1
            self.lbl_frames.setText(f"Frames: {self.frame_count}")
            self.lbl_frames.setStyleSheet("color: #ff5555; font-weight: bold;")

    def closeEvent(self, event):
        if hasattr(self, 'worker') and self.worker is not None:
            self.worker.stop()
        if hasattr(self, 'radar') and self.radar is not None:
            try:
                self.radar.close()
            except: pass
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RadarRecorderApp() 
    window.show()
    sys.exit(app.exec())