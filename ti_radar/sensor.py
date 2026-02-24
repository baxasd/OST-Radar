import serial
from serial.tools import list_ports
import time
import logging
from ti_radar.config import RadarConfig
from ti_radar.parser import parse_standard_frame

log = logging.getLogger(__name__)

class RadarSensor:
    # 8-byte sequence marking the absolute start of a new data frame
    UART_MAGIC_WORD = bytearray(b'\x02\x01\x04\x03\x06\x05\x08\x07')

    def __init__(self, cli_port, data_port, config_file):
        self.config = RadarConfig(config_file)
        self.cli_port_name = cli_port
        self.data_port_name = data_port
        self.cli_com = None
        self.data_com = None

    def connect_and_configure(self):
        """Opens serial ports and flashes the CFG profile to the radar's DSP"""
        self.cli_com = serial.Serial(self.cli_port_name, 115200, timeout=0.6)
        self.data_com = serial.Serial(self.data_port_name, 921600, timeout=0.6)
        self.data_com.reset_output_buffer()
        
        self._send_cfg()
        self.config.print_summary()

    def _send_cfg(self):
        """Sends commands line-by-line via the CLI port"""
        with open(self.config.file_path, "r") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('%')]

        for line in lines:
            time.sleep(0.03)
            self.cli_com.write((line + '\n').encode())
            
            # Flush acks from hardware
            self.cli_com.readline()
            self.cli_com.readline()

            if line.startswith("baudRate"):
                self.cli_com.baudrate = int(line.split()[1])
                
        time.sleep(0.03)
        self.cli_com.reset_input_buffer()

    def read_raw_frame(self):
        """Scans the UART stream until it locks onto a Magic Word, then reads exactly one frame."""
        index = 0
        frameData = bytearray()
        
        # 1. Hunt for the magic word to sync with the hardware stream
        while True:
            magicByte = self.data_com.read(1)
            if not magicByte:
                return None 

            if magicByte[0] == self.UART_MAGIC_WORD[index]:
                index += 1
                frameData.append(magicByte[0])
                if index == len(self.UART_MAGIC_WORD):
                    break
            else:
                # Mismatch found. Reset search, but check if this mismatched byte 
                # is actually the start of a brand new sequence to prevent dropped frames.
                index = 1 if magicByte[0] == self.UART_MAGIC_WORD[0] else 0
                frameData = bytearray([magicByte[0]]) if index == 1 else bytearray()

        # 2. Extract frame length
        versionBytes = self.data_com.read(4)
        lengthBytes = self.data_com.read(4)
        
        if len(versionBytes) < 4 or len(lengthBytes) < 4:
            return None 
            
        frameData.extend(versionBytes)
        frameData.extend(lengthBytes)
        
        frameLength = int.from_bytes(lengthBytes, byteorder='little')
        
        # 3. Size sanity check (reject corrupt packets)
        if not (16 <= frameLength <= 100000):
            return None
            
        # 4. Pull the exact payload size to ensure we don't bleed into the next frame
        bytes_to_read = frameLength - 16 
        while bytes_to_read > 0:
            chunk = self.data_com.read(bytes_to_read)
            if not chunk: 
                return None
            frameData.extend(chunk)
            bytes_to_read -= len(chunk)
            
        return frameData

    def get_next_frame(self):
        """Fetches and parses a single frame dictionary"""
        raw_bytes = self.read_raw_frame()
        return parse_standard_frame(raw_bytes) if raw_bytes else None

    def close(self):
        if self.cli_com and self.cli_com.is_open: self.cli_com.close()
        if self.data_com and self.data_com.is_open: self.data_com.close()

    @staticmethod
    def find_ti_ports():
        """Auto-detects active TI radar COM ports"""
        cli, data = None, None
        for p in list_ports.comports():
            if 'XDS110 Class Application/User UART' in p.description or 'Enhanced COM Port' in p.description:
                cli = p.device
            elif 'XDS110 Class Auxiliary Data Port' in p.description or 'Standard COM Port' in p.description:
                data = p.device
        return cli, data