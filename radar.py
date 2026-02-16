import sys
import time
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                             QHBoxLayout, QWidget, QPushButton, QLabel, QFrame)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QPalette, QColor

from ti_radar import RadarSensor

# ===================== USER CONFIG =====================
CLI_PORT = "COM10"
DATA_PORT = "COM11"
CFG_FILE = "ti_radar/chirp_config/profile_3d_isk.cfg"

TREADMILL_MIN_DIST = 0.5  # meters
TREADMILL_MAX_DIST = 1.0  # meters
HISTORY_FRAMES = 20      # Spectrogram history length
# ======================================================

class RadarWorker(QThread):
    new_spectrogram_line = pyqtSignal(np.ndarray)
    stats_update = pyqtSignal(float, int) # FPS, Error Count

    def __init__(self, radar_sensor):
        super().__init__()
        self.radar = radar_sensor
        self.is_running = True
        self.error_count = 0
        
        self.min_range_idx = int(TREADMILL_MIN_DIST / self.radar.config.rangeRes)
        self.max_range_idx = int(TREADMILL_MAX_DIST / self.radar.config.rangeRes)
        self.center_doppler_bin = self.radar.config.numLoops // 2

    def run(self):
        frames_processed = 0
        start_time = time.time()

        while self.is_running:
            try:
                frame = self.radar.get_next_frame()
                
                if frame and frame.get("RDHM"):
                    rd_matrix = np.array(frame["RDHM"], dtype=np.float32)
                    expected_size = self.radar.config.ADCsamples * self.radar.config.numLoops
                    
                    if rd_matrix.size != expected_size:
                        self.error_count += 1
                        continue 
                        
                    rd_matrix = rd_matrix.reshape(self.radar.config.ADCsamples, self.radar.config.numLoops)
                    rd_matrix = np.fft.fftshift(rd_matrix, axes=1)
                    
                    # --- CRITICAL NEW MATH: STATIC CLUTTER REMOVAL ---
                    # Zero out the center bins (stationary objects like walls/floor)
                    # This allows the moving feet to stand out in the color scale
                    rd_matrix[:, self.center_doppler_bin-2 : self.center_doppler_bin+3] = 1e-6
                    
                    # Convert to dB
                    rd_matrix = 20 * np.log10(np.abs(rd_matrix) + 1e-6)
                    
                    # Range Gating & Profile Extraction
                    target_area = rd_matrix[self.min_range_idx:self.max_range_idx, :]
                    velocity_profile = np.max(target_area, axis=0)
                    
                    self.new_spectrogram_line.emit(velocity_profile)
                    
                    # Calculate FPS
                    frames_processed += 1
                    elapsed = time.time() - start_time
                    if elapsed > 1.0:
                        fps = frames_processed / elapsed
                        self.stats_update.emit(fps, self.error_count)
                        frames_processed = 0
                        start_time = time.time()
                else:
                    self.error_count += 1
                    
            except Exception as e:
                self.error_count += 1

    def stop(self):
        self.is_running = False
        self.wait()


class RecorderWindow(QMainWindow):
    def __init__(self, radar_sensor):
        super().__init__()
        self.radar = radar_sensor
        self.setWindowTitle("OST Radar: Advanced Micro-Doppler Diagnostics")
        self.resize(1100, 650)
        
        # --- Data Buffers ---
        self.history_frames = HISTORY_FRAMES
        self.num_vel_bins = self.radar.config.numLoops
        self.max_vel = self.radar.config.dopMax
        self.spectrogram_buffer = np.zeros((self.history_frames, self.num_vel_bins), dtype=np.float32)

        self.init_ui()

        # --- Start Background Thread ---
        self.worker = RadarWorker(self.radar)
        self.worker.new_spectrogram_line.connect(self.update_plot)
        self.worker.stats_update.connect(self.update_stats)
        self.worker.start()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Dashboard UI ---
        dashboard_layout = QHBoxLayout()
        
        self.btn_toggle_stream = QPushButton("⏸ Pause Stream")
        self.btn_toggle_stream.setCheckable(True)
        self.btn_toggle_stream.clicked.connect(self.toggle_stream)
        self.btn_toggle_stream.setMinimumHeight(40)
        self.btn_toggle_stream.setMinimumWidth(150)
        
        self.lbl_fps = QLabel("FPS: --")
        self.lbl_fps.setStyleSheet("font-size: 16px; font-weight: bold; color: #4CAF50;")
        
        self.lbl_errors = QLabel("Errors: 0")
        self.lbl_errors.setStyleSheet("font-size: 16px; font-weight: bold; color: #F44336;")
        
        self.lbl_status = QLabel("Status: STREAMING")
        self.lbl_status.setStyleSheet("font-size: 14px; color: #00BCD4;")

        dashboard_layout.addWidget(self.btn_toggle_stream)
        dashboard_layout.addSpacing(20)
        dashboard_layout.addWidget(self.lbl_fps)
        dashboard_layout.addSpacing(20)
        dashboard_layout.addWidget(self.lbl_errors)
        dashboard_layout.addStretch()
        dashboard_layout.addWidget(self.lbl_status)
        
        main_layout.addLayout(dashboard_layout)
        
        # Divider Line
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(line)

        # --- PyQtGraph Plot Setup ---
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Velocity', units='m/s')
        self.plot_widget.setLabel('bottom', 'Time', units='Frames')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        self.image_item = pg.ImageItem()
        self.plot_widget.addItem(self.image_item)
        
        # Scale the image axes from "Bins" to actual "m/s"
        # setRect parameters: (x_start, y_start, x_width, y_height)
        rect = pg.QtCore.QRectF(0, -self.max_vel, self.history_frames, self.max_vel * 2)
        self.image_item.setRect(rect)
        
        # A better colormap for radar
        colormap = pg.colormap.get('magma') 
        self.image_item.setColorMap(colormap)
        
        main_layout.addWidget(self.plot_widget)

    def update_plot(self, velocity_profile):
        self.spectrogram_buffer = np.roll(self.spectrogram_buffer, -1, axis=0)
        self.spectrogram_buffer[-1, :] = velocity_profile
        
        # Smarter Auto-Leveling
        active_data = self.spectrogram_buffer[self.spectrogram_buffer > 0]
        if len(active_data) > 100:
            vmin = np.percentile(active_data, 20) # Hide the lower 20% noise
            vmax = np.percentile(active_data, 99)
        else:
            vmin, vmax = 0, 100 
            
        self.image_item.setImage(self.spectrogram_buffer, autoLevels=False, levels=(vmin, vmax))

    def update_stats(self, fps, errors):
        self.lbl_fps.setText(f"FPS: {fps:.1f}")
        self.lbl_errors.setText(f"Dropped Packets: {errors}")

    def toggle_stream(self):
        if self.btn_toggle_stream.isChecked():
            self.btn_toggle_stream.setText("▶ Resume Stream")
            self.lbl_status.setText("Status: PAUSED")
            self.lbl_status.setStyleSheet("font-size: 14px; color: #FF9800;")
            self.worker.new_spectrogram_line.disconnect(self.update_plot)
        else:
            self.btn_toggle_stream.setText("⏸ Pause Stream")
            self.lbl_status.setText("Status: STREAMING")
            self.lbl_status.setStyleSheet("font-size: 14px; color: #00BCD4;")
            self.worker.new_spectrogram_line.connect(self.update_plot)

    def closeEvent(self, event):
        self.worker.stop()
        self.radar.close()
        event.accept()

# --- Dark Theme ---
def apply_dark_theme(app):
    app.setStyle("Fusion")
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(18, 18, 18))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(40, 40, 40))
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(45, 45, 45))
    dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    app.setPalette(dark_palette)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    
    try:
        radar = RadarSensor(CLI_PORT, DATA_PORT, CFG_FILE)
        radar.connect_and_configure()
    except Exception as e:
        print(f"Failed to initialize radar: {e}")
        sys.exit(1)
        
    window = RecorderWindow(radar)
    window.show()
    sys.exit(app.exec())