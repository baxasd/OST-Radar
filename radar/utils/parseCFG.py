
class cfg_params:

    def __init__(self, file):
        with open(file) as f:
            lines = []
            for l in f:
                if not l.startswith("%") and not l.startswith("\n"):
                    lines.append(l)

        antennas = {}
        for l in lines:
            if l.startswith("channelCfg"):
                val = l.split()
                for _ in val:
                    antennas.update({'txAntennas': int(val[1]), 'rxAntenna': int(val[2])})
            
        # Number of antennas
        self.txAntennas = bin(antennas['txAntennas']).count("1")
        self.rxAntennas = bin(antennas['rxAntenna']).count("1")
        self.Nv = self.txAntennas * self.rxAntennas

        self.chirpProf = {}
        for l in lines:
            if l.startswith("profileCfg"):
                val = l.split()
                for _ in val:
                    self.chirpProf.update({'startFreq': float(val[2]), 'idleTime': float(val[3]), 'adcStartTime': float(val[4]),
                                    'rampEndTime': float(val[5]), 'freqSlope': float(val[8]), 'numADCsamples': int(val[10]),
                                    'sampleRate': float(val[11])})
        # Bandwith acquisition (EFFECTIVE BW)
        self.BW = self.chirpProf['freqSlope']*self.chirpProf['numADCsamples']/self.chirpProf['sampleRate'] * 1e9
        # Bandwith emission (TOTAL BW)
        self.BW2 = self.chirpProf['freqSlope']*self.chirpProf['rampEndTime']*1e-3
        # Chirp duration
        self.chirpTime = 1e3*self.chirpProf['numADCsamples'] / self.chirpProf['sampleRate']
        # Range resolution
        self.rangeRes = 3e8 / (2*self.BW)
        # Max range
        self.rangeMax = self.rangeRes*(self.chirpProf['numADCsamples']-1)


        self.frameCfg = {}
        for l in lines:
            if l.startswith("frameCfg"):
                val = l.split()
                for v in val:
                    self.frameCfg.update({'chirpStartInd': int(val[1]), 'chirpEndInd': int(val[2]),
                        'numLoops': int(val[3]), 'periodicity': int(val[5])})
                    
        # Number of chirps per frame
        self.numChirps = (self.frameCfg['chirpEndInd'] - self.frameCfg['chirpStartInd'] + 1) * self.frameCfg['numLoops']

        # Doppler analysis
        self.dopRes = 3e8 / (2 * self.chirpProf['startFreq'] * 1e9
                        *(self.chirpProf['idleTime'] + self.chirpProf['rampEndTime']) * 1e-6
                        * self.numChirps)
        self.numLoops = self.frameCfg['numLoops']
        self.dopMax = self.numLoops*self.dopRes / 2 
        # Periodicity
        self.T = self.frameCfg['periodicity']
        self.ADCsamples = self.chirpProf['numADCsamples']

        self.print_summary()


    def checkCFG(self):
        # CHECK
        # Ramp end time
        ADC_time = self.chirpProf['numADCsamples']/self.chirpProf['sampleRate']*1e3 + self.chirpProf['adcStartTime']
        print("========== Conditions checks ==========")
        print('=====================')
        print('1st condition:')
        print('Ramp time = ' + str(self.chirpProf['rampEndTime']) + ' us > ADC time = ' + str(ADC_time) + ' us')
        if self.chirpProf['rampEndTime'] > ADC_time:
            print('OK')
        else:
            print('ERROR')

        # Emission vs Acquisition Times
        time1 = self.chirpProf['rampEndTime'] - self.chirpProf['adcStartTime']
        time2 = 1e3 * self.chirpProf['numADCsamples'] / self.chirpProf['sampleRate']
        print('=====================')
        print('2nd condition:')
        print('Emission Time = ' + str(time1) + ' us > Acquisition Time = ' + str(time2) + ' us')
        if time1 > time2:
            print('OK')
        else:
            print('ERROR')

        # Receiving BW (Total BW vs Effective BW)
        f0 = self.chirpProf["startFreq"]
        fend = f0 + self.BW2
        print('=====================')
        print('3rd condition:')
        print(f'Initial frequency = {f0} GHz')
        print('Final frequency = ' + str(fend) + ' GHz')
        print('Bandwidth = ' + str(fend-f0) + ' < 4 GHz')
        if fend-f0 < 4:
            print('OK')
        else:
            print('ERROR')


    def print_summary(self):
       print("========== CFG Summary ==========")
       print(f"Frame periodicity: {self.T:.2f} ms ({1e3/self.T:.2f} Hz)")
       # Emission Time (commented out for now)
       # etime = self.numChirps * (self.chirpProf['rampEndTime'] + self.chirpProf['idleTime']) * 1e-3
       # print(f"Total emission time: {etime:.2f} ms")
       print(f"Effective Bandwidth: {(self.BW*1e-9):.2f} GHz")
       print(f"Range Resolution: {self.rangeRes*100:.2f} cm")
       print(f"Maximum range: {self.rangeMax:.2f} m")
       print(f"Doppler Resolution: {self.dopRes:.2f} m/s ({self.dopRes*3.6:.2f} km/h)")
       print(f"Maximum velocity: {self.dopMax:.2f} m/s ({self.dopMax*3.6:.2f} km/h)")
       print("=================================")


if __name__ == "__main__":
    file = "radar/chirp_config/ISK_6m_default.cfg"
    my_file = cfg_params(file)
    my_file.checkCFG()