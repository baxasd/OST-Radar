import matplotlib.pyplot as plt
import numpy as np
import radar.utils.mmMaths as mm
# %% Connect to the COM ports
from radar.utils.UARTconnection import *
import radar.utils.parseCFG as cfg

ports = find_TI_ports()
# [cliCOM, dataCOM]  = connectComPorts(ports["UART"].device, ports["DATA"].device)
[cliCOM, dataCOM]  = connectComPorts("COM10", "COM11")
chirp_cfg_file = "radar/chirp_config/profile_3d_isk.cfg"
sendCfg(chirp_cfg_file, cliCOM)
# Parse CFG file
cfg_params = cfg.cfg_params(chirp_cfg_file)

# Parameters
r = cfg_params.rangeRes * np.arange(cfg_params.chirpProf['numADCsamples'])
v = np.linspace(-cfg_params.dopMax, cfg_params.dopMax, cfg_params.numLoops)


signal = readAndParseUartDoubleCOMPort(dataCOM)
plt.ion()
for i in range(500):
    signal = readAndParseUartDoubleCOMPort(dataCOM)
    # record signal
    y = np.reshape(signal["RDHM"],(cfg_params.ADCsamples,cfg_params.numLoops))
    y = np.fft.fftshift(y, axes=1)
    c = plt.pcolormesh(v,r,y)
    plt.colorbar(c)
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Range (m)')
    plt.draw()
    plt.pause(0.1)
    plt.clf()




###### RAHM #######

# Parameters
# numAzBins = 63
# r = cfg_params.rangeRes * np.arange(cfg_params.chirpProf['numADCsamples'])
# theta = np.linspace(-90, 90, numAzBins)

# X = r[4:34].reshape(-1,1) * np.sin(np.deg2rad(theta[3:-3]))
# Y = r[4:34].reshape(-1,1) * np.cos(np.deg2rad(theta[3:-3]))

# import matplotlib.pyplot as plt
# import numpy as np
# import radar.utils.mmMaths as mm

# signal = readAndParseUartDoubleCOMPort(dataCOM)
# plt.ion()
# for i in range(1000):
#     signal = readAndParseUartDoubleCOMPort(dataCOM)
#     rahm = np.reshape(signal["RAHM"],(cfg_params.ADCsamples,cfg_params.Nv))
#     y = abs(mm.RAHM_fft(rahm, numAzBins))
#     c = plt.pcolormesh(X,Y,y[4:34,3:-3])
#     plt.colorbar(c)
#     plt.xlabel('Range (m)')
#     plt.ylabel('Azimuth (deg)')
#     plt.draw()
#     plt.pause(0.1)
#     plt.clf()


# import matplotlib.pyplot as plt
# import numpy as np

# r = cfg_params.rangeRes * np.arange(cfg_params.chirpProf['numADCsamples'])
# signal = readAndParseUartDoubleCOMPort(dataCOM)
# bg = np.array(signal["rangeProfile"])

# plt.ion()
# for i in range(1000):
#     signal = readAndParseUartDoubleCOMPort(dataCOM)
#     y = np.subtract(np.array(signal["rangeProfile"]),bg)
#     # y = signal["rangeProfile"]
#     y[y<0] = 0
#     plt.ylim((0,1000))
#     plt.plot(r,y)
#     plt.xlabel("Range (m)")
#     plt.draw()
#     plt.pause(0.1)
#     plt.clf()





