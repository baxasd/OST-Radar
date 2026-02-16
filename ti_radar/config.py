import logging

log = logging.getLogger(__name__)

class RadarConfig:
    def __init__(self, file_path):
        self.file_path = file_path
        self._parse_file(file_path)

    def _parse_file(self, file_path):
        with open(file_path, "r") as f:
            lines = [l for l in f if not l.startswith("%") and not l.startswith("\n")]

        antennas = {'txAntennas': 0, 'rxAntenna': 0}
        self.chirpProf = {}
        self.frameCfg = {}

        for l in lines:
            val = l.split()
            if not val: continue
            
            if val[0] == "channelCfg":
                antennas.update({'txAntennas': int(val[1]), 'rxAntenna': int(val[2])})
            elif val[0] == "profileCfg":
                self.chirpProf.update({
                    'startFreq': float(val[2]), 'idleTime': float(val[3]), 
                    'adcStartTime': float(val[4]), 'rampEndTime': float(val[5]), 
                    'freqSlope': float(val[8]), 'numADCsamples': int(val[10]),
                    'sampleRate': float(val[11])
                })
            elif val[0] == "frameCfg":
                self.frameCfg.update({
                    'chirpStartInd': int(val[1]), 'chirpEndInd': int(val[2]),
                    'numLoops': int(val[3]), 'periodicity': int(val[5])
                })

        # Calculations
        self.txAntennas = bin(antennas['txAntennas']).count("1")
        self.rxAntennas = bin(antennas['rxAntenna']).count("1")
        self.Nv = self.txAntennas * self.rxAntennas

        self.BW = self.chirpProf['freqSlope'] * self.chirpProf['numADCsamples'] / self.chirpProf['sampleRate'] * 1e9
        self.BW2 = self.chirpProf['freqSlope'] * self.chirpProf['rampEndTime'] * 1e-3
        self.chirpTime = 1e3 * self.chirpProf['numADCsamples'] / self.chirpProf['sampleRate']
        self.rangeRes = 3e8 / (2 * self.BW)
        self.rangeMax = self.rangeRes * (self.chirpProf['numADCsamples'] - 1)

        self.numChirps = (self.frameCfg['chirpEndInd'] - self.frameCfg['chirpStartInd'] + 1) * self.frameCfg['numLoops']
        self.dopRes = 3e8 / (2 * self.chirpProf['startFreq'] * 1e9 * (self.chirpProf['idleTime'] + self.chirpProf['rampEndTime']) * 1e-6 * self.numChirps)
        
        self.numLoops = self.frameCfg['numLoops']
        self.dopMax = self.numLoops * self.dopRes / 2 
        self.T = self.frameCfg['periodicity']
        self.ADCsamples = self.chirpProf['numADCsamples']

    def print_summary(self):
        print("\n========== CFG Summary ==========")
        print(f"Frame periodicity : {self.T:.2f} ms ({1e3/self.T:.2f} Hz)")
        print(f"Effective BW      : {(self.BW*1e-9):.2f} GHz")
        print(f"Range Resolution  : {self.rangeRes*100:.2f} cm")
        print(f"Maximum range     : {self.rangeMax:.2f} m")
        print(f"Doppler Resolution: {self.dopRes:.2f} m/s")
        print(f"Maximum velocity  : {self.dopMax:.2f} m/s")
        print("=================================\n")