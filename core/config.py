import logging
import math

log = logging.getLogger(__name__)

class RadarConfig:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._parse(file_path)

    def _parse(self, file_path: str):
        with open(file_path) as f:
            lines = [l.split() for l in f if l.strip() and not l.startswith("%")]

        chirp, frame, rx_en, tx_en = {}, {}, 0, 0

        for val in lines:
            if not val:
                continue
            cmd = val[0]
            if cmd == "channelCfg":
                rx_en = int(val[1])
                tx_en = int(val[2])
            elif cmd == "profileCfg":
                if int(val[1]) == 0:
                    chirp = {
                        "startFreq":     float(val[2]),
                        "idleTime":      float(val[3]),
                        "rampEndTime":   float(val[5]),
                        "freqSlope":     float(val[8]),
                        "numADCsamples": int(val[10]),
                        "sampleRate":    float(val[11]),
                    }
            elif cmd == "frameCfg":
                frame = {
                    "chirpStartInd": int(val[1]),
                    "chirpEndInd":   int(val[2]),
                    "numLoops":      int(val[3]),
                    "periodicity":   int(val[5]),
                }

        if not chirp:
            raise ValueError(f"No profileCfg (profile 0) found in {file_path}")
        if not frame:
            raise ValueError(f"No frameCfg found in {file_path}")

        self.rxAntennas = bin(rx_en).count("1")
        self.txAntennas = bin(tx_en).count("1")

        self.ADCsamples = chirp["numADCsamples"]

        # Next power-of-two padded FFT size
        self.numRangeBins = 1 if self.ADCsamples == 0 else 2 ** math.ceil(math.log2(self.ADCsamples))

        self.BW = chirp["freqSlope"] * self.ADCsamples / chirp["sampleRate"] * 1e9

        self.rangeRes = 3e8 / (2 * self.BW)
        # FIX: use numRangeBins (padded FFT size), not raw ADC samples
        self.rangeMax = self.rangeRes * self.numRangeBins

        chirps_per_loop = frame["chirpEndInd"] - frame["chirpStartInd"] + 1
        self.numLoops   = frame["numLoops"]
        numChirps       = chirps_per_loop * self.numLoops

        Tc = (chirp["idleTime"] + chirp["rampEndTime"]) * 1e-6
        fc = chirp["startFreq"] * 1e9

        self.dopRes = 3e8 / (2 * fc * Tc * numChirps)
        # FIX: dopMax must use numChirps (full CPI), not just numLoops
        self.dopMax = numChirps * self.dopRes / 2

        self.T         = frame["periodicity"]
        self.frameRate = 1e3 / self.T

    def summary(self) -> dict:
        return {
            "TX / RX antennas":   f"{self.txAntennas} / {self.rxAntennas}",
            "Bandwidth":          f"{self.BW / 1e9:.3f} GHz",
            "Range resolution":   f"{self.rangeRes * 100:.2f} cm",
            "Range max":          f"{self.rangeMax:.2f} m",
            "Range FFT Bins":     f"{self.numRangeBins}",
            "Doppler resolution": f"{self.dopRes:.3f} m/s",
            "Max velocity":       f"±{self.dopMax:.2f} m/s",
            "Frame rate":         f"{self.frameRate:.1f} Hz ({self.T:.0f} ms)",
        }