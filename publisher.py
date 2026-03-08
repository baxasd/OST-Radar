import sys                           # Used for system-level operations, like exiting the script
import time                          # Used for small sleep delays so we don't max out the CPU
import datetime                      # Used to generate unique timestamps for the Parquet filenames
import os                            # Used to check for and create the 'records' directory
import logging                       # Used to print warnings and errors to the terminal gracefully
import zmq                           # The ZeroMQ library for high-performance network sockets
import numpy as np                   # Used for handling the raw byte arrays from the radar
import pandas as pd                  # Used as an intermediary format to convert Python dicts to Parquet
import pyarrow as pa                 # The core memory format used by Parquet
import pyarrow.parquet as pq         # The actual writer that saves the data to the disk

from core.radar import RadarSensor   # Imports your hardware abstraction class to talk to the USB ports
from core.base import parse_standard_frame  # Imports the function that extracts the RDHM matrix from the raw packet

# Create a logger specifically for this script so our terminal output is labeled nicely
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  RadarSessionWriter — Handles high-speed, chunked disk writes
# ─────────────────────────────────────────────────────────────────────────────
class RadarSessionWriter:
    def __init__(self, subject_id="radar", activity="session", metadata=None):
        # Ensure the 'records' folder exists on the disk; if it doesn't, create it
        os.makedirs("records", exist_ok=True)
        
        # Get the exact current date and time to inject into the filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Construct the final path where this session's data will be saved
        self.filepath = f"records/{subject_id}_{activity}_{timestamp}.parquet"
        
        # Store any configuration metadata (like radar settings) to embed in the file later
        self.metadata = metadata or {}
        
        # Initialize an empty Python list to act as our RAM buffer
        self.data_buffer = []
        
        # Set the threshold for how many frames to hold in RAM before writing to the disk
        self.chunk_size = 100
        
        # We start with no active file writer; it will be created on the first disk flush
        self.writer = None
        
        # Keep a running tally of how many frames we've recorded in total
        self.total_frames = 0
        
        # Define the exact column names our Parquet file will use
        self.schema_columns = ['timestamp', 'rdhm_bytes']

    def write_frame(self, rdhm_array: np.ndarray):
        # Convert the 2D Numpy matrix into a raw, flat string of bytes (extremely fast)
        # and append it to our RAM buffer along with the current Unix timestamp
        self.data_buffer.append({
            'timestamp': time.time(),
            'rdhm_bytes': rdhm_array.tobytes()
        })
        
        # Increment our total frame counter
        self.total_frames += 1
        
        # Check if our RAM buffer has reached the chunk size limit (e.g., 100 frames)
        if len(self.data_buffer) >= self.chunk_size:
            # If it has, trigger the function to write these frames to the actual hard drive
            self._flush_buffer()

    def _flush_buffer(self):
        # If the buffer is empty (e.g., called during shutdown with no data), do nothing
        if not self.data_buffer:
            return
            
        # Convert our list of dictionaries into a Pandas DataFrame, enforcing our column names
        df = pd.DataFrame(self.data_buffer, columns=self.schema_columns)
        
        # Convert the Pandas DataFrame into a PyArrow Table (the native format for Parquet)
        table = pa.Table.from_pandas(df)
        
        # If this is the very first time we are writing to the disk...
        if self.writer is None:
            # Create a custom metadata dictionary containing our radar configuration
            schema_with_meta = table.schema.with_metadata({
                b"session_meta": str(self.metadata).encode()
            })
            
            # Apply this new metadata-enriched schema to our PyArrow Table
            table = table.cast(schema_with_meta)
            
            # Open the actual file stream on the disk, locking in the schema
            self.writer = pq.ParquetWriter(self.filepath, schema_with_meta)
            
        # Append the current chunk of data to the open file on the disk
        self.writer.write_table(table)
        
        # Clear out the RAM buffer so it's empty and ready for the next 100 frames
        self.data_buffer.clear()

    def close(self):
        # Force any remaining frames in the RAM buffer to write to the disk
        self._flush_buffer()
        
        # If we successfully opened a writer during the session...
        if self.writer:
            # Safely close the file stream so the Parquet file isn't corrupted
            self.writer.close()
            # Print a success message to the terminal showing the file location and total frame count
            print(f"✅ Session saved: {self.filepath} ({self.total_frames} frames)")


# ─────────────────────────────────────────────────────────────────────────────
#  Hardware Connection Helper (Brought over from your original viewer)
# ─────────────────────────────────────────────────────────────────────────────
def connect_radar() -> RadarSensor:
    # Attempt to automatically scan the USB ports to find the Texas Instruments radar
    cli, data = RadarSensor.find_ti_ports()
    
    # If the auto-detect failed to find both ports...
    if not cli or not data:
        # Print a helpful error message to the user explaining what went wrong
        print("ERROR: Ports not found. Check USB connection.")
        # Terminate the script with an error code (1)
        sys.exit(1)

    # Print the ports we successfully found to the terminal
    print(f"CLI: {cli}  DATA: {data}")
    
    # Create the radar sensor object, pointing it to your configuration file
    radar = RadarSensor(cli, data, "core/config.cfg")
    
    # Open the serial connections and send the configuration commands to the hardware
    radar.connect_and_configure()
    
    # Return the fully initialized and streaming hardware object
    return radar


# ─────────────────────────────────────────────────────────────────────────────
#  Main Publisher Loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Set the logging level to WARNING so we don't clutter the terminal with minor info logs
    logging.basicConfig(level=logging.WARNING)

    # 1. Setup ZeroMQ Publisher
    # Create the background ZeroMQ context that manages the I/O threads
    context = zmq.Context()
    
    # Create a PUB (Publisher) socket intended to broadcast data to anyone listening
    socket = context.socket(zmq.PUB)
    
    # Bind the socket to port 5555 on ALL network interfaces (localhost + LAN IP)
    socket.bind("tcp://*:5555")
    
    # Inform the user that the network broadcaster is active
    print("ZMQ Publisher bound to tcp://*:5555")

    # 2. Connect to Hardware
    # Call our helper function to initialize the USB serial ports
    radar = connect_radar()
    
    # 3. Setup Parquet Writer
    # Initialize our chunked disk writer, passing in the hardware's configuration summary
    writer = RadarSessionWriter(metadata=radar.config.summary())

    # Tell the user how to gracefully stop the recording
    print("Publishing and recording data... Press Ctrl+C to stop.")
    
    try:
        # Enter an infinite loop to process frames as fast as they arrive
        while True:
            # Pull exactly one complete binary frame packet from the USB serial buffer
            raw_bytes = radar.read_raw_frame()
            
            # If a full frame hasn't arrived over USB yet...
            if raw_bytes is None:
                # Sleep for 1 millisecond to yield the CPU and prevent 100% core utilization
                time.sleep(0.001)
                # Skip the rest of the loop and try reading again
                continue

            # Pass the raw binary packet into the parser to extract the TLV payloads
            frame = parse_standard_frame(raw_bytes)
            
            # Attempt to extract the Range-Doppler Heat Map matrix from the parsed dictionary
            rdhm = frame.get("RDHM")
            
            # If the frame actually contained a valid Heat Map matrix...
            if rdhm is not None:
                # 1. Network: Convert the Numpy array to raw bytes and broadcast it over ZMQ instantly
                socket.send(rdhm.tobytes())
                
                # 2. Disk: Append the matrix to our Parquet RAM buffer
                writer.write_frame(rdhm)

    # Catch the KeyboardInterrupt exception which happens when the user presses Ctrl+C
    except KeyboardInterrupt:
        # Print a clean shutdown message
        print("\nShutting down publisher...")
        
    # The 'finally' block ensures cleanup happens even if the script crashes unexpectedly
    finally:
        # Send the 'sensorStop' command to the radar and close the USB serial ports
        radar.close()
        
        # Flush the final chunk of data to the disk and close the Parquet file
        writer.close()
        
        # Close the ZeroMQ network socket
        socket.close()
        
        # Terminate the ZeroMQ context to cleanly release the background threads
        context.term()

# Standard Python boilerplate to ensure main() only runs if the script is executed directly
if __name__ == "__main__":
    main()