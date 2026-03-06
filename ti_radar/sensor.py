import time
import logging
import serial
from serial.tools import list_ports

from ti_radar.config import RadarConfig
from ti_radar.parser import parse_standard_frame

log = logging.getLogger(__name__)

# TI IWR/AWR magic word — marks the start of every data frame
_MAGIC = bytearray(b"\x02\x01\x04\x03\x06\x05\x08\x07")
_MAGIC_LEN = len(_MAGIC)


class RadarSensor:

    def __init__(self, cli_port: str, data_port: str, config_file: str):
        self.config         = RadarConfig(config_file)
        self._cli_port_name = cli_port
        self._data_port_name= data_port
        self._cli           = None
        self._data          = None

    # ── connection ────────────────────────────────────────────────────────
    def connect_and_configure(self):
        """Opens serial ports and flashes the profile to the radar DSP."""
        self._cli  = serial.Serial(self._cli_port_name,  115200, timeout=0.6)
        self._data = serial.Serial(self._data_port_name, 921600, timeout=0.6)
        self._data.reset_output_buffer()
        self._send_cfg()

    def _send_cfg(self):
        """Streams config commands line-by-line over the CLI port."""
        with open(self.config.file_path) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("%")]

        for line in lines:
            time.sleep(0.03)
            self._cli.write((line + "\n").encode())
            self._cli.readline()   # ack line 1
            self._cli.readline()   # ack line 2
            # FIX: removed dead 'baudRate' branch — that command does not exist
            # in standard TI mmWave profiles and was never reachable.

        time.sleep(0.03)
        self._cli.reset_input_buffer()

    # ── frame acquisition ─────────────────────────────────────────────────
    def read_raw_frame(self) -> bytes | None:
        """Syncs to the magic word and reads exactly one complete frame."""
        idx  = 0
        buf  = bytearray()

        # Phase 1: hunt for the 8-byte magic word
        while True:
            b = self._data.read(1)
            if not b:
                return None
            byte = b[0]
            if byte == _MAGIC[idx]:
                buf.append(byte)
                idx += 1
                if idx == _MAGIC_LEN:
                    break
            else:
                # On mismatch, check if this byte restarts the sequence
                if byte == _MAGIC[0]:
                    buf = bytearray([byte])
                    idx = 1
                else:
                    buf.clear()
                    idx = 0

        # Phase 2: read version (4 bytes) + length (4 bytes)
        extra = self._data.read(8)
        if len(extra) < 8:
            return None
        buf.extend(extra)

        frame_len = int.from_bytes(extra[4:8], byteorder="little")
        if not (16 <= frame_len <= 100_000):
            return None  # reject corrupt/oversized packets

        # Phase 3: read remainder of frame without bleeding into the next
        remaining = frame_len - 16
        while remaining > 0:
            chunk = self._data.read(remaining)
            if not chunk:
                return None
            buf.extend(chunk)
            remaining -= len(chunk)

        return bytes(buf)

    def get_next_frame(self) -> dict | None:
        raw = self.read_raw_frame()
        return parse_standard_frame(raw) if raw else None

    # ── cleanup ───────────────────────────────────────────────────────────
    def close(self):
        for port in (self._cli, self._data):
            if port and port.is_open:
                port.close()

    # ── port detection ────────────────────────────────────────────────────
    @staticmethod
    def find_ti_ports() -> tuple[str | None, str | None]:
        """Auto-detects TI XDS110 CLI and data ports by USB descriptor."""
        cli = data = None
        for p in list_ports.comports():
            desc = p.description
            if "XDS110 Class Application/User UART" in desc or "Enhanced COM Port" in desc:
                cli = p.device
            elif "XDS110 Class Auxiliary Data Port" in desc or "Standard COM Port" in desc:
                data = p.device
        return cli, data