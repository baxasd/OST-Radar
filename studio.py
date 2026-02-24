import sys
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyqtgraph as pg

from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

# --- Shared UI Constants ---
WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 800
STYLESHEET = """
QMainWindow { background-color: #1e1e1e; color: #ffffff; }
QFrame#Sidebar { background-color: #252526; border-right: 1px solid #333; }
QLabel { color: #e0e0e0; font-size: 12px; }
QLabel#Header { font-size: 14px; font-weight: bold; color: #ffffff; margin-top: 15px; }
QLabel#DataLabel { font-family: Consolas; color: #4CAF50; }
QLabel#HighlightLabel { font-family: Consolas; color: #ffeb3b; font-size: 18px; font-weight: bold; }
QPushButton { background-color: #0e639c; color: white; border: none; border-radius: 4px; font-weight: bold; padding: 10px; }
QPushButton:hover { background-color: #1177bb; }
"""

class RadarStudioApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OST Radar Studio")
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet(STYLESHEET)
        
        self._build_ui()

    # ================= UI SETUP =================
    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        main_layout.addWidget(self._create_sidebar())
        main_layout.addWidget(self._create_plots_area())

    def _create_sidebar(self):
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(260)
        layout = QVBoxLayout(sidebar)
        layout.setSpacing(8)

        layout.addWidget(QLabel("OST Radar Studio", font=QFont("Segoe UI", 16, QFont.Weight.Bold)))
        
        btn_load = QPushButton("Load .parquet File", cursor=Qt.CursorShape.PointingHandCursor)
        btn_load.clicked.connect(self._load_file)
        layout.addWidget(btn_load)
        layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine, styleSheet="color: #444; margin: 10px 0;"))

        # Results Group
        layout.addWidget(QLabel("Automated Findings", objectName="Header"))
        self.lbl_cadence = self._add_row(layout, "Cadence (SPM):", "HighlightLabel")
        self.lbl_cadence_hz = self._add_row(layout, "Step Freq (Hz):", "DataLabel")

        # Metadata Group
        layout.addWidget(QLabel("Metadata", objectName="Header"))
        self.lbl_subject = self._add_row(layout, "Subject:", "DataLabel")
        self.lbl_activity = self._add_row(layout, "Activity:", "DataLabel")
        
        # Stats Group
        layout.addWidget(QLabel("Recording Stats", objectName="Header"))
        self.lbl_duration = self._add_row(layout, "Duration:", "DataLabel")
        self.lbl_fps = self._add_row(layout, "Avg FPS:", "DataLabel")
        
        layout.addStretch()
        return sidebar

    def _add_row(self, layout, text, obj_name):
        row = QHBoxLayout()
        row.addWidget(QLabel(text))
        val_label = QLabel("-", objectName=obj_name, alignment=Qt.AlignmentFlag.AlignRight)
        row.addWidget(val_label)
        layout.addLayout(row)
        return val_label

    def _create_plots_area(self):
        container = QWidget(styleSheet="background-color: black;")
        layout = QVBoxLayout(container)
        pg.setConfigOptions(imageAxisOrder='row-major')

        # Micro-Doppler Plot
        self.plot_md = pg.PlotWidget(title="Micro-Doppler Spectrogram")
        self.plot_md.setLabel('left', 'Velocity', units='m/s')
        self.plot_md.setLabel('bottom', 'Time', units='s')
        
        self.img_md = pg.ImageItem(colorMap=pg.colormap.get('turbo'))
        self.plot_md.addItem(self.img_md)
        self.plot_md.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('w', style=Qt.PenStyle.DashLine, alpha=100)))

        # Frequency Plot
        self.plot_freq = pg.PlotWidget(title="Movement Frequency Analysis")
        self.plot_freq.setLabel('left', 'Magnitude')
        self.plot_freq.setLabel('bottom', 'Frequency', units='Hz')
        self.plot_freq.setFixedHeight(200) 
        
        self.curve_freq = self.plot_freq.plot(pen='y')
        self.peak_marker = pg.ScatterPlotItem(size=10, brush=pg.mkBrush('r'))
        self.plot_freq.addItem(self.peak_marker)

        layout.addWidget(self.plot_md)
        layout.addWidget(self.plot_freq)
        return container

    # ================= LOGIC =================
    def _load_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Radar Data", "", "Parquet Files (*.parquet)")
        if not filepath: return

        try:
            # 1. Read Parquet Table
            table = pq.read_table(filepath)
            df = table.to_pandas()
            
            # 2. Extract embedded custom schema metadata
            meta_json = table.schema.metadata[b'radar_metadata'].decode()
            shape_json = table.schema.metadata[b'frame_shape'].decode()
            dtype_str = table.schema.metadata[b'frame_dtype'].decode()
            
            meta = json.loads(meta_json)
            shape = tuple(json.loads(shape_json))
            dtype = np.dtype(dtype_str)

            # 3. Reconstruct the 3D numpy array from columnar bytes
            timestamps = df['timestamp'].values
            rdhm = np.array([np.frombuffer(b, dtype=dtype).reshape(shape) for b in df['frame_data']])

            # Update Metadata UI
            self.lbl_subject.setText(str(meta.get('subject', 'N/A')))
            self.lbl_activity.setText(str(meta.get('activity', 'N/A')))
            
            duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
            self.lbl_duration.setText(f"{duration:.2f} s")
            self.lbl_fps.setText(f"{(len(rdhm) / duration):.1f}" if duration > 0 else "0.0")

            self._process_and_plot(rdhm, duration, meta)

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load Parquet file:\n{str(e)}")

    def _process_and_plot(self, rdhm, duration, meta):
        # Shift 0 m/s velocity to the center of the array
        rdhm_shifted = np.fft.fftshift(rdhm, axes=2)
        dop_max, dop_res = meta.get('dop_max', 5.0), meta.get('doppler_res', 0.1)

        # ==============================================================
        # 1. Micro-Doppler Generation
        # ==============================================================
        # Sum reflections across the Range axis to get (Time vs Velocity)
        md_data = np.sum(np.abs(rdhm_shifted), axis=1) 
        md_data_db = 20 * np.log10(md_data + 1e-6).T # Transpose for PyQtGraph
        
        self.img_md.setImage(md_data_db, autoLevels=False)
        self.img_md.setRect(QRectF(0, -dop_max - (dop_res/2), duration, dop_max * 2))
        self.img_md.setLevels((np.percentile(md_data_db, 50), np.percentile(md_data_db, 99.9)))

        # ==============================================================
        # 2. Cadence Frequency Analysis (FFT)
        # ==============================================================
        # Identify the stationary clutter bin
        center_idx = md_data.shape[1] // 2
        ignore_bins = int(0.5 / dop_res) # Ignore anything moving slower than 0.5 m/s
        
        # Calculate total energy of fast-moving limbs over time
        moving_energy = np.sum(md_data[:, :center_idx-ignore_bins], axis=1) + \
                        np.sum(md_data[:, center_idx+ignore_bins:], axis=1)
        
        # Detrend (remove static DC offset) to stabilize the FFT
        moving_energy_detrended = moving_energy - np.mean(moving_energy)
        
        n_frames = len(moving_energy_detrended)
        dt = duration / n_frames if n_frames > 0 else 1
        
        # Fast Fourier Transform on the 1D energy wave
        freqs = np.fft.rfftfreq(n_frames, d=dt)
        fft_mag = np.abs(np.fft.rfft(moving_energy_detrended))
        
        # Human step frequency falls strictly between 0.8 Hz and 4.0 Hz
        human_freq_mask = np.where((freqs >= 0.8) & (freqs <= 4.0))[0]
        
        if len(human_freq_mask) > 0:
            best_idx = human_freq_mask[np.argmax(fft_mag[human_freq_mask])]
            cadence_hz = freqs[best_idx]
            
            self.lbl_cadence.setText(f"{cadence_hz * 60:.0f}")
            self.lbl_cadence_hz.setText(f"{cadence_hz:.2f} Hz")
            
            self.curve_freq.setData(freqs, fft_mag)
            self.plot_freq.setXRange(0, 5.0) 
            self.peak_marker.setData([cadence_hz], [fft_mag[best_idx]])
        else:
            self.lbl_cadence.setText("N/A")
            self.lbl_cadence_hz.setText("N/A")
            self.curve_freq.setData([], [])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RadarStudioApp()
    window.show()
    sys.exit(app.exec())