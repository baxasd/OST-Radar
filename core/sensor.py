import time
import logging
import serial
from serial.tools import list_ports

from ti_radar.config import RadarConfig
from ti_radar.parser import parse_standard_frame

log = logging.getLogger(__name__)

_MAGIC  = b"\x02\x01\x04\x03\x06\x05\x08\x07"
_TI_VID = 0x0451

class RadarSensor:

    def __init__(self, cli_port: str, data_port: str, config_file: str):
        self.config          = RadarConfig(config_file)
        self._cli_port_name  = cli_port
        self._data_port_name = data_port
        self._cli            = None
        self._data           = None
        self._buffer         = bytearray()

    def connect_and_configure(self):
        self._cli  = serial.Serial(self._cli_port_name,  115200, timeout=0.6)
        self._data = serial.Serial(self._data_port_name, 921600, timeout=1.0)
        self._data.reset_output_buffer()
        self._send_cfg()

    def _send_cfg(self):
        with open(self.config.file_path) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("%")]

        for line in lines:
            self._cli.write((line + "\n").encode())
            if line.startswith("sensorStop"):
                time.sleep(0.1)
                self._cli.reset_input_buffer()
                continue
            if line.startswith("sensorStart"):
                time.sleep(0.05)
                self._cli.reset_input_buffer()
                continue
            self._read_until_done()

        self._cli.reset_input_buffer()

    def _read_until_done(self, timeout: float = 0.3):
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self._cli.readline().decode(errors="ignore").strip()
            if "Done" in line:
                return
            if "Error" in line or "Ignored" in line:
                log.warning(f"CFG response: {line}")
                return

    def read_raw_frame(self) -> bytes | None:
        in_waiting = self._data.in_waiting
        if in_waiting > 0:
            self._buffer.extend(self._data.read(in_waiting))
        else:
            chunk = self._data.read(4096)
            if not chunk:
                return None
            self._buffer.extend(chunk)
        if len(self._buffer) > 16384:
            log.warning("Oversized buffer — flushing to next magic word.")
            idx = self._buffer.find(_MAGIC, 1)
            if idx != -1:
                self._buffer = self._buffer[idx:]
            else:
                self._buffer.clear()
            return None

        idx = self._buffer.find(_MAGIC)

        if idx == -1:
            if len(self._buffer) > 7:
                self._buffer = self._buffer[-7:]
            return None

        if idx > 0:
            self._buffer = self._buffer[idx:]

        if len(self._buffer) < 40:
            return None

        frame_len = int.from_bytes(self._buffer[12:16], byteorder="little")

        if not (16 <= frame_len <= 16384):
            self._buffer = self._buffer[8:]
            return None

        if len(self._buffer) < frame_len:
            return None

        frame_data = bytes(self._buffer[:frame_len])
        self._buffer = self._buffer[frame_len:]
        return frame_data

    def get_next_frame(self) -> dict | None:
        raw = self.read_raw_frame()
        return parse_standard_frame(raw) if raw else None

    def close(self):
        if self._cli and self._cli.is_open:
            try:
                self._cli.write(b"sensorStop\n")
                time.sleep(0.1)
            except Exception as e:
                log.error(f"Failed to send sensorStop: {e}")

        for port in (self._cli, self._data):
            if port and port.is_open:
                port.close()

    @staticmethod
    def find_ti_ports() -> tuple[str | None, str | None]:
        cli = data = None
        for p in list_ports.comports():
            desc      = p.description or ""
            vid_match = getattr(p, "vid", None) == _TI_VID

            if "Application/User UART" in desc or "Enhanced COM Port" in desc:
                cli = p.device
            elif "Auxiliary Data Port" in desc or "Standard COM Port" in desc:
                data = p.device
            elif vid_match:
                # No description match — fall back to arrival order
                if cli is None:
                    cli = p.device
                elif data is None:
                    data = p.device

        return cli, data