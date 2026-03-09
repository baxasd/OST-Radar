import sys
import os
import ast
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.signal import butter, filtfilt, find_peaks

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, QFrame, QSizePolicy)

import pyqtgraph as pg

# Force pyqtgraph to use a modern white theme BEFORE creating any widgets
pg.setConfigOption('background', '#FFFFFF')
pg.setConfigOption('foreground', '#333333')
pg.setConfigOptions(antialias=True, imageAxisOrder='row-major')


# ─────────────────────────────────────────────────────────────────────────────
#  Signal Processing
# ─────────────────────────────────────────────────────────────────────────────
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


# ─────────────────────────────────────────────────────────────────────────────
#  Modern UI Application
# ─────────────────────────────────────────────────────────────────────────────
class AnalyzerWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("OST Radar | Gait Analysis Studio")
        self.resize(1300, 850)
        self.selected_file = None

        # Main Layout (Horizontal: Sidebar + Content)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self._build_sidebar()
        self._build_graph_area()

    def _build_sidebar(self):
        """Builds the modern left-hand control panel."""
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(280)
        self.sidebar.setStyleSheet("""
            QFrame { background-color: #F8F9FA; border-right: 1px solid #E0E0E0; }
            QLabel { color: #333333; }
            QPushButton {
                background-color: #FFFFFF; border: 1px solid #CCCCCC;
                border-radius: 6px; padding: 8px; font-weight: bold; color: #333;
            }
            QPushButton:hover { background-color: #E2E6EA; }
            QPushButton#analyzeBtn {
                background-color: #0078D4; border: none; color: white;
            }
            QPushButton#analyzeBtn:hover { background-color: #005A9E; }
            QPushButton#analyzeBtn:disabled { background-color: #A0CBE8; }
        """)
        
        layout = QVBoxLayout(self.sidebar)
        layout.setContentsMargins(20, 30, 20, 30)
        layout.setSpacing(15)

        # Title
        title = QLabel("Gait Analysis")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        layout.addWidget(title)

        # File Controls
        self.btn_browse = QPushButton("📁 Load Parquet File")
        self.btn_browse.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_browse.clicked.connect(self.browse_file)
        layout.addWidget(self.btn_browse)

        self.lbl_filename = QLabel("No file selected")
        self.lbl_filename.setStyleSheet("color: #777777; font-size: 11px;")
        self.lbl_filename.setWordWrap(True)
        layout.addWidget(self.lbl_filename)

        self.btn_analyze = QPushButton("▶ Run Analysis")
        self.btn_analyze.setObjectName("analyzeBtn")
        self.btn_analyze.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self.run_analysis)
        layout.addWidget(self.btn_analyze)

        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("border-top: 1px solid #D0D0D0; margin-top: 20px; margin-bottom: 20px;")
        layout.addWidget(divider)

        # Metrics Panel
        metrics_title = QLabel("Session Metrics")
        metrics_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        layout.addWidget(metrics_title)

        self.val_duration = self._create_metric_label("Duration", "--")
        self.val_steps = self._create_metric_label("Total Steps", "--")
        self.val_cadence = self._create_metric_label("Cadence", "-- SPM", color="#0078D4")
        self.val_symmetry = self._create_metric_label("Symmetry", "-- %", color="#107C10")
        self.val_dominant = self._create_metric_label("Dominant Leg", "--")

        layout.addWidget(self.val_duration)
        layout.addWidget(self.val_steps)
        layout.addWidget(self.val_cadence)
        layout.addWidget(self.val_symmetry)
        layout.addWidget(self.val_dominant)

        layout.addStretch() # Pushes everything to the top
        self.main_layout.addWidget(self.sidebar)

    def _create_metric_label(self, title, value, color="#333333"):
        lbl = QLabel(f"<b>{title}:</b> <span style='color:{color};'>{value}</span>")
        lbl.setFont(QFont("Segoe UI", 11))
        return lbl

    def _update_metric(self, label_widget, title, value, color="#333333"):
        label_widget.setText(f"<b>{title}:</b> <span style='color:{color};'>{value}</span>")

    def _build_graph_area(self):
        """Builds the main right-hand visualization area."""
        self.graph_panel = QWidget()
        self.graph_panel.setStyleSheet("background-color: #FFFFFF;")
        graph_layout = QVBoxLayout(self.graph_panel)
        graph_layout.setContentsMargins(20, 20, 20, 20)
        graph_layout.setSpacing(20)

        # 1. Spectrogram Plot (Completely independent)
        self.plot_spec = pg.PlotWidget(title="Micro-Doppler Spectrogram")
        self.plot_spec.setLabel('left', 'Velocity (m/s)')
        self.plot_spec.setLabel('bottom', 'Time', units='s')
        self.plot_spec.showGrid(x=True, y=True, alpha=0.2)
        
        self.img_spec = pg.ImageItem()
        # 'viridis' looks fantastic on white backgrounds
        self.img_spec.setColorMap(pg.colormap.get('viridis')) 
        self.plot_spec.addItem(self.img_spec)
        graph_layout.addWidget(self.plot_spec)

        # 2. Cadence Plot (Completely independent)
        self.plot_cadence = pg.PlotWidget(title="Step Detection Waveform")
        self.plot_cadence.setLabel('left', 'Energy Amplitude (Z)')
        self.plot_cadence.setLabel('bottom', 'Time', units='s')
        self.plot_cadence.showGrid(x=True, y=True, alpha=0.3)
        
        self.curve_raw = self.plot_cadence.plot(pen=pg.mkPen('#DDDDDD', width=1), name="Raw")
        self.curve_filt = self.plot_cadence.plot(pen=pg.mkPen('#0078D4', width=2), name="Filtered")
        
        # distinct markers for Leg 1 and Leg 2
        self.scatter_1 = pg.ScatterPlotItem(size=12, pen=pg.mkPen(None), brush=pg.mkBrush('#D13438'), symbol='o')
        self.scatter_2 = pg.ScatterPlotItem(size=12, pen=pg.mkPen(None), brush=pg.mkBrush('#FFB900'), symbol='o')
        self.plot_cadence.addItem(self.scatter_1)
        self.plot_cadence.addItem(self.scatter_2)

        graph_layout.addWidget(self.plot_cadence)
        self.main_layout.addWidget(self.graph_panel)

    def browse_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Radar Session", os.path.join(os.getcwd(), "records"), "Parquet Files (*.parquet);;All Files (*)"
        )
        if filepath:
            self.selected_file = filepath
            self.lbl_filename.setText(os.path.basename(filepath))
            self.btn_analyze.setEnabled(True)

    def run_analysis(self):
        if not self.selected_file:
            return

        self.btn_analyze.setText("Crunching Data...")
        self.btn_analyze.setEnabled(False)
        QApplication.processEvents() 

        try:
            # 1. Load Parquet Data
            table = pq.read_table(self.selected_file)
            df = table.to_pandas()
            
            schema = table.schema
            metadata_bytes = schema.metadata.get(b'session_meta', b'{}')
            try:
                meta = ast.literal_eval(metadata_bytes.decode('utf-8'))
            except:
                meta = {}

            num_range_bins = int(meta.get('Range FFT Bins', 64))
            num_vel_bins = 32 
            fps = float(meta.get('Frame rate', "15.1").split()[0])
            dop_max = float(meta.get('Max velocity', "20.16").replace('±','').replace(' m/s',''))
            
            total_frames = len(df)
            time_axis = np.arange(total_frames) / fps
            duration_minutes = total_frames / fps / 60.0

            # 2. Reconstruct Matrices
            rdhm_cube = np.zeros((total_frames, num_range_bins, num_vel_bins), dtype=np.float32)
            for i, row in df.iterrows():
                raw = np.frombuffer(row['rdhm_bytes'], dtype=np.uint16)
                if raw.size == num_range_bins * num_vel_bins:
                    rd = raw.astype(np.float32).reshape(num_range_bins, num_vel_bins)
                    rd = np.abs(np.fft.fftshift(rd, axes=1))
                    rdhm_cube[i] = rd

            rdhm_db = 20 * np.log10(rdhm_cube + 1e-6)
            spectrogram = np.max(rdhm_db, axis=1)

            # --- VISUAL FIX FOR SPECTROGRAM ---
            # The center bin (stationary clutter) is so loud it ruins the visual color scale.
            # We clone the spectrogram and aggressively mute the center 3 bins for drawing only.
            vis_spectrogram = spectrogram.copy()
            center_bin = num_vel_bins // 2
            floor_val = np.percentile(vis_spectrogram, 5) # Find the background noise floor
            vis_spectrogram[:, center_bin-1:center_bin+2] = floor_val

            # 3. Extract Cadence Signal (Skipping the center bins)
            moving_bins = list(range(0, center_bin - 2)) + list(range(center_bin + 3, num_vel_bins))
            movement_signal = np.sum(spectrogram[:, moving_bins], axis=1)

            normal_max = np.percentile(movement_signal, 99.5)
            movement_signal = np.clip(movement_signal, a_min=None, a_max=normal_max)
            movement_signal = (movement_signal - np.mean(movement_signal)) / (np.std(movement_signal) + 1e-6)

            # 4. Filter and Find Peaks
            filtered_signal = butter_bandpass_filter(movement_signal, lowcut=1.0, highcut=4.0, fs=fps)
            peaks, _ = find_peaks(filtered_signal, distance=int(fps / 4.0), prominence=0.4)
            
            total_steps = len(peaks)
            average_cadence = total_steps / duration_minutes if duration_minutes > 0 else 0

            # 5. Advanced Gait Symmetry & Dominance Logic
            symmetry_score = 0.0
            dominance_text = "N/A"
            sym_color = "#333333"
            
            if total_steps > 4:
                # Get the time duration of every single stride interval
                step_intervals = np.diff(time_axis[peaks])
                
                # Split into alternating leg swings
                intervals_1_to_2 = step_intervals[0::2]
                intervals_2_to_1 = step_intervals[1::2]
                
                median_1_to_2 = np.median(intervals_1_to_2)
                median_2_to_1 = np.median(intervals_2_to_1)

                symmetry_score = (min(median_1_to_2, median_2_to_1) / max(median_1_to_2, median_2_to_1)) * 100.0

                if symmetry_score >= 95:
                    dominance_text = "Balanced (Symmetrical)"
                    sym_color = "#107C10" # Green
                else:
                    sym_color = "#D13438" # Red
                    # A dominant leg produces a faster swing (shorter interval)
                    if median_1_to_2 < median_2_to_1:
                        dominance_text = "Leg 1 (Faster swing)"
                    else:
                        dominance_text = "Leg 2 (Faster swing)"

            # 6. Update Sidebar Metrics
            self._update_metric(self.val_duration, "Duration", f"{duration_minutes:.2f} min")
            self._update_metric(self.val_steps, "Total Steps", str(total_steps))
            self._update_metric(self.val_cadence, "Cadence", f"{average_cadence:.1f} SPM", color="#0078D4")
            self._update_metric(self.val_symmetry, "Symmetry", f"{symmetry_score:.1f}%", color=sym_color)
            self._update_metric(self.val_dominant, "Dominant Leg", dominance_text, color=sym_color)

            # 7. Draw Visuals
            
            # Draw Spectrogram
            self.img_spec.setImage(vis_spectrogram, autoLevels=True)
            # Map pixels perfectly to real world Time and physical Velocity
            self.img_spec.setRect(QRectF(0, -dop_max, time_axis[-1], dop_max * 2.0))

            # Draw Waves
            self.curve_raw.setData(x=time_axis, y=movement_signal)
            self.curve_filt.setData(x=time_axis, y=filtered_signal)

            # Draw Alternating Steps
            if total_steps > 0:
                peaks_leg_1 = peaks[0::2]
                peaks_leg_2 = peaks[1::2]
                self.scatter_1.setData(x=time_axis[peaks_leg_1], y=filtered_signal[peaks_leg_1])
                self.scatter_2.setData(x=time_axis[peaks_leg_2], y=filtered_signal[peaks_leg_2])

            self.plot_spec.autoRange()
            self.plot_cadence.autoRange()

        except Exception as e:
            self.lbl_filename.setText(f"Error processing data: {str(e)}")
            self.lbl_filename.setStyleSheet("color: #D13438; font-size: 11px;")
        
        finally:
            self.btn_analyze.setText("▶ Run Analysis")
            self.btn_analyze.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = AnalyzerWindow()
    window.show()
    sys.exit(app.exec())