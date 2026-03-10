import sys
import time
import datetime
import os
import logging
import zmq
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import configparser

from core.radar import RadarSensor
from core.base import parse_standard_frame, VERSION

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("Publisher")

# ─────────────────────────────────────────────────────────────────────────────
#  Load Global Settings
# ─────────────────────────────────────────────────────────────────────────────
config = configparser.ConfigParser()
config.read('settings.ini')

HW_CFG_FILE = config['Hardware']['radar_cfg_file']
HW_CLI_PORT = config['Hardware']['cli_port']
HW_DATA_PORT = config['Hardware']['data_port']
ZMQ_PORT = config['Network']['zmq_port']
CHUNK_SIZE = int(config['Recording']['chunk_size'])

# ─────────────────────────────────────────────────────────────────────────────
#  RadarSessionWriter
# ─────────────────────────────────────────────────────────────────────────────
class RadarSessionWriter:
    def __init__(self, metadata=None):
        os.makedirs("records", exist_ok=True)
        
        self.start_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = f"records/session_{self.start_time_str}.parquet"
        
        self.metadata = metadata or {}
        self.metadata["session_start"] = self.start_time_str
        
        self.data_buffer = []
        self.chunk_size = CHUNK_SIZE
        self.writer = None
        self.total_frames = 0
        self.schema_columns = ['timestamp', 'rdhm_bytes']

    def write_frame(self, rdhm_array: np.ndarray):
        self.data_buffer.append({'timestamp': time.time(), 'rdhm_bytes': rdhm_array.tobytes()})
        self.total_frames += 1
        
        if len(self.data_buffer) >= self.chunk_size:
            self._flush_buffer()

    def _flush_buffer(self):
        if not self.data_buffer:
            return
            
        df = pd.DataFrame(self.data_buffer, columns=self.schema_columns)
        table = pa.Table.from_pandas(df)
        
        if self.writer is None:
            schema_with_meta = table.schema.with_metadata({b"session_meta": str(self.metadata).encode()})
            table = table.cast(schema_with_meta)
            self.writer = pq.ParquetWriter(self.filepath, schema_with_meta)
            
        self.writer.write_table(table)
        self.data_buffer.clear()

    def close(self):
        self._flush_buffer()
        if self.writer:
            self.writer.close()
            log.info(f"Session saved successfully: {self.filepath} ({self.total_frames} frames)")

# ─────────────────────────────────────────────────────────────────────────────
#  Hardware Connection Helper
# ─────────────────────────────────────────────────────────────────────────────
def connect_radar() -> RadarSensor:
    log.info("Connecting to Texas Instruments hardware...")
    
    cli, data = None, None
    if HW_CLI_PORT.lower() != 'auto' and HW_DATA_PORT.lower() != 'auto':
        cli, data = HW_CLI_PORT, HW_DATA_PORT
    else:
        log.info("Scanning for auto-assigned USB ports...")
        cli, data = RadarSensor.find_ti_ports()
    
    if not cli or not data:
        log.error("Auto-detection failed: No TI radar ports found. Please check connections and try again.")
        return None

    log.info(f"Using CLI: {cli} | DATA: {data}")
    
    radar = RadarSensor(cli, data, HW_CFG_FILE)
    radar.connect_and_configure()
    
    print("\n" + "="*40)
    print(" RADAR CONFIGURATION LOADED")
    print("="*40)
    for key, value in radar.config.summary().items():
        print(f" {key:<20}: {value}")
    print("="*40 + "\n")
    
    return radar

# ─────────────────────────────────────────────────────────────────────────────
#  Core Streaming Loop
# ─────────────────────────────────────────────────────────────────────────────
def run_stream(zmq_socket: zmq.Socket, record: bool):
    radar = connect_radar()
    if radar is None: return

    writer = RadarSessionWriter(metadata=radar.config.summary()) if record else None
    
    if record: log.info(f"RECORD MODE: Broadcasting over ZMQ and saving to {writer.filepath}")
    else: log.info("PREVIEW MODE: Broadcasting over ZMQ only (No disk writing).")

    print("\n>>> STREAM ACTIVE. Press Ctrl+C to stop and return to menu. <<<\n")
    
    try:
        while True:
            raw_bytes = radar.read_raw_frame()
            if raw_bytes is None:
                time.sleep(0.001)
                continue

            frame = parse_standard_frame(raw_bytes)
            rdhm = frame.get("RDHM")
            
            if rdhm is not None:
                zmq_socket.send(rdhm.tobytes())
                if record: writer.write_frame(rdhm)

    except KeyboardInterrupt:
        log.info("Ctrl+C detected. Stopping stream...")
    finally:
        radar.close()
        if writer: writer.close()
        time.sleep(0.5)

# ─────────────────────────────────────────────────────────────────────────────
#  Main Application Menu
# ─────────────────────────────────────────────────────────────────────────────
def main():
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{ZMQ_PORT}")
    
    print("=========================================")
    print(f"                OST RADAR v{VERSION}       ")
    print(f"        ZMQ Bound to tcp://*:{ZMQ_PORT}    ")
    print("=========================================")

    while True:
        print("\nMAIN MENU:")
        print("  1. Preview Mode")
        print("  2. Record Mode")
        print("  3. Exit Application")
        
        choice = input("\nSelect an option (1-3): ").strip()
        
        if choice == '1': run_stream(socket, record=False)
        elif choice == '2': run_stream(socket, record=True)
        elif choice == '3':
            print("Shutting down network and exiting...")
            break
        else: print("Invalid choice.")

    socket.close()
    context.term()
    sys.exit(0)

if __name__ == "__main__":
    main()