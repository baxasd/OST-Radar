# %%
"""
Script to save data from the camera and the radar, with timestamps
"""
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(description="Capture data from mmWave radar.")
parser.add_argument("--filename", type=str, default="radar_data", help="Filename to save the radar data.")
parser.add_argument("--chirp_cfg_file", type=str, default="ISK_6m_default", help="Chirp configuration file.")
parser.add_argument("--num_samples", type=int, default=300, help="Number of samples to capture.")

args = parser.parse_args()

# %% RADAR CONFIG
import radar.utils.parseCFG as cfg
# Connect to the COM ports
from radar.utils.UARTconnection import *

ports = find_TI_ports()
[cliCOM, dataCOM]  = connectComPorts("COM18", "COM17")
# chirp_cfg_file = "radar/chirp_config/profile_3d_ISK.cfg"
# Send CFG file
sendCfg(f'radar/chirp_config/{args.chirp_cfg_file}.cfg', cliCOM)
# Parse CFG file
cfg_params = cfg.cfg_params(f'radar/chirp_config/{args.chirp_cfg_file}.cfg')


# %% Append data and timestamps
tradar = []
pc = []
tpc = []

import matplotlib.pyplot as plt
# import numpy as np
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax = fig.add_subplot(111)
signal = []

for k in range(args.num_samples):
    # Get the current time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    tradar.append(timestamp)
    # Read and parse radar data
    signal.append(readAndParseUartDoubleCOMPort(dataCOM))

    try:
        pc.append(signal[-1]["pointCloud"])
        ax.scatter(pc[-1][:,0], pc[-1][:,1], pc[-1][:,2], marker='.')
    except KeyError:
        pc.append(None)

    try:
        tpc.append(signal[-1]["trackData"])
        labels = tpc[-1][:,0]
        ax.scatter(tpc[-1][:,1], tpc[-1][:,2], tpc[-1][:,3], marker='p')
        # for i in range(len(labels)):
        #     ax.text(xt[i], yt[i], zt[i], str(int(labels[i])))
    except KeyError:
        tpc.append(None)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_xlim([-2,2])
    ax.set_ylim([0,3])
    ax.set_zlim([-2,2])
    

    plt.draw()
    plt.pause(0.001)
    ax.cla()


# import pickle
# with open(f"data/{args.filename}_radar.pkl", 'wb') as file: 
#     # A new file will be created 
#     pickle.dump(tradar, file) 
#     pickle.dump(signal, file)

# print(f"Data saved to 'data/{args.filename}_radar.pkl'")