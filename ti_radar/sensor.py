import serial
from serial.tools import list_ports
import time
import logging
import platform
from .config import RadarConfig
from .parser import parse_standard_frame

log = logging.getLogger(__name__)

class RadarSensor:
    UART_MAGIC_WORD = bytearray(b'\x02\x01\x04\x03\x06\x05\x08\x07')

    def __init__(self, cli_port, data_port, config_file):
        self.config = RadarConfig(config_file)
        self.cli_port_name = cli_port
        self.data_port_name = data_port
        self.cli_com = None
        self.data_com = None

    def connect_and_configure(self):
        """Opens serial ports and sends configuration"""
        self.cli_com = serial.Serial(self.cli_port_name, 115200, timeout=0.6)
        self.data_com = serial.Serial(self.data_port_name, 921600, timeout=0.6)
        self.data_com.reset_output_buffer()
        
        self._send_cfg()
        self.config.print_summary()

    def _send_cfg(self):
        with open(self.config.file_path, "r") as f:
            lines = [line for line in f.readlines() if line.strip() and not line.startswith('%')]

        for line in lines:
            time.sleep(0.03)
            self.cli_com.write((line.strip() + '\n').encode())
            # Read acks
            self.cli_com.readline()
            self.cli_com.readline()

            if line.startswith("baudRate"):
                self.cli_com.baudrate = int(line.split()[1])
                
        time.sleep(0.03)
        self.cli_com.reset_input_buffer()

    def read_raw_frame(self):
        """Hunts for the magic word and extracts exactly one raw byte frame"""
        index = 0
        frameData = bytearray()
        
        # 1. Hunt for the magic word sequence
        while True:
            magicByte = self.data_com.read(1)
            if not magicByte:
                return None # Timeout

            if magicByte[0] == self.UART_MAGIC_WORD[index]:
                index += 1
                frameData.append(magicByte[0])
                if index == 8:
                    break
            else:
                # Fix: Reset search, but check if the current byte starts a NEW sequence
                index = 0
                frameData = bytearray()
                if magicByte[0] == self.UART_MAGIC_WORD[0]:
                    index = 1
                    frameData.append(magicByte[0])

        # 2. Read Header Info (Version + Length)
        versionBytes = self.data_com.read(4)
        lengthBytes = self.data_com.read(4)
        
        if len(versionBytes) < 4 or len(lengthBytes) < 4:
            return None # Dropped payload immediately after magic word
            
        frameData += bytearray(versionBytes) + bytearray(lengthBytes)
        
        frameLength = int.from_bytes(lengthBytes, byteorder='little')
        
        # 3. Sanity check: If a frame claims to be an absurd size (corruption), reject it
        if frameLength > 100000 or frameLength < 16:
            return None
            
        bytes_to_read = frameLength - 16 # Subtract bytes already read
        payload = bytearray()
        
        # 4. CRITICAL FIX 3: Loop until the exact number of bytes are pulled from the buffer
        while bytes_to_read > 0:
            chunk = self.data_com.read(bytes_to_read)
            if not chunk: 
                return None # Hardware timeout halfway through frame
            payload += bytearray(chunk)
            bytes_to_read -= len(chunk)
            
        frameData += payload
        return frameData

    def get_next_frame(self):
        """Fetches raw bytes and parses them into usable data"""
        raw_bytes = self.read_raw_frame()
        if not raw_bytes:
            return None
        return parse_standard_frame(raw_bytes)

    def close(self):
        """Cleanly close ports"""
        if self.cli_com and self.cli_com.is_open:
            self.cli_com.close()
        if self.data_com and self.data_com.is_open:
            self.data_com.close()

    @staticmethod
    def find_ti_ports():
        """Helper static method to find ports automatically"""
        ports = list(list_ports.comports())
        cli, data = None, None
        for p in ports:
            if 'XDS110 Class Application/User UART' in p.description or 'Enhanced COM Port' in p.description:
                cli = p.device
            elif 'XDS110 Class Auxiliary Data Port' in p.description or 'Standard COM Port' in p.description:
                data = p.device
        return cli, data