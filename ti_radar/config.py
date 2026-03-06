import logging

log = logging.getLogger(__name__)

class RadarConfig:
    """Parses TI mmWave .cfg files and derives physical radar parameters."""

    def __init__(self, file_path):
        self.file_path = file_path
        self._parse(file_path)

    def _parse(self, file_path):
        with open(file_path) as f:
            lines = [l.split() for l in f if l.strip() and not l.startswith("%")]

        chirp, frame, rx_en, tx_en = {}, {}, 0, 0

        for val in lines:
            if not val:
                continue
            if val[0] == "channelCfg":
                # FIX: TI doc order is <rxChannelEn> <txChannelEn>
                # Previous code had these swapped — txAntennas/rxAntennas labels
                # were wrong even though Nv (their product) was unaffected.
                rx_en = int(val[1])
                tx_en = int(val[2])
            elif val[0] == "profileCfg":
                chirp = {
                    "startFreq":    float(val[2]),   # GHz
                    "idleTime":     float(val[3]),   # us
                    "rampEndTime":  float(val[5]),   # us
                    "freqSlope":    float(val[8]),   # MHz/us
                    "numADCsamples":  int(val[10]),
                    "sampleRate":   float(val[11]),  # ksps
                }
            elif val[0] == "frameCfg":
                frame = {
                    "chirpStartInd": int(val[1]),
                    "chirpEndInd":   int(val[2]),
                    "numLoops":      int(val[3]),
                    "periodicity":   int(val[5]),    # ms
                }

        # Antenna counts (popcount of enable bitmasks)
        self.rxAntennas = bin(rx_en).count("1")
        self.txAntennas = bin(tx_en).count("1")
        self.Nv         = self.txAntennas * self.rxAntennas

        # Bandwidth: slope [MHz/us] * sweep_time [us] -> MHz, then -> Hz
        # sweep_time [us] = numADCsamples / sampleRate [ksps] * 1e3
        self.ADCsamples = chirp["numADCsamples"]
        self.BW = chirp["freqSlope"] * self.ADCsamples / chirp["sampleRate"] * 1e9  # Hz

        # Range
        self.rangeRes = 3e8 / (2 * self.BW)            # m
        self.rangeMax = self.rangeRes * self.ADCsamples  # FIX: N bins, not N-1

        # Doppler
        # Total chirps per frame = chirps_per_loop * numLoops
        chirps_per_loop  = frame["chirpEndInd"] - frame["chirpStartInd"] + 1
        self.numLoops    = frame["numLoops"]
        numChirps        = chirps_per_loop * self.numLoops

        # Tc = time per chirp (idleTime + rampEndTime) in seconds
        Tc           = (chirp["idleTime"] + chirp["rampEndTime"]) * 1e-6
        fc           = chirp["startFreq"] * 1e9  # Hz

        # For TDM MIMO the effective slow-time PRI is chirps_per_loop * Tc
        # dopRes  = c / (2 * fc * Tc * numChirps)  [equivalent formulation]
        # dopMax  = c / (4 * fc * chirps_per_loop * Tc)
        self.dopRes  = 3e8 / (2 * fc * Tc * numChirps)
        self.dopMax  = self.numLoops * self.dopRes / 2

        # Frame timing
        self.T = frame["periodicity"]  # ms

    def summary(self) -> dict:
        """Returns key parameters as a plain dict for external display."""
        return {
            "TX / RX antennas":  f"{self.txAntennas} / {self.rxAntennas}",
            "Bandwidth":         f"{self.BW / 1e9:.3f} GHz",
            "Range resolution":  f"{self.rangeRes * 100:.2f} cm",
            "Range max":         f"{self.rangeMax:.2f} m",
            "Doppler resolution":f"{self.dopRes:.3f} m/s",
            "Max velocity":      f"±{self.dopMax:.2f} m/s",
            "Frame rate":        f"{1e3 / self.T:.1f} Hz ({self.T:.0f} ms)",
        }