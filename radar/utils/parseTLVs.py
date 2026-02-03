import struct
import logging
import numpy as np


# Local File Imports
from radar.utils.mmMaths import sphericalToCartesianPointCloud

log = logging.getLogger(__name__)

# ================================================== Parsing Functions For Individual TLV's ==================================================

# Point Cloud TLV from SDK
def parsePointCloudTLV(tlvData, tlvLength, outputDict):
    pointCloud = outputDict['pointCloud']
    pointStruct = '4f'  # X, Y, Z, and Doppler
    pointStructSize = struct.calcsize(pointStruct)
    numPoints = int(tlvLength/pointStructSize)

    for i in range(numPoints):
        try:
            x, y, z, doppler = struct.unpack(pointStruct, tlvData[:pointStructSize])
        except:
            numPoints = i
            log.error('Point Cloud TLV Parser Failed')
            break
        tlvData = tlvData[pointStructSize:]
        pointCloud[i,0] = x 
        pointCloud[i,1] = y
        pointCloud[i,2] = z
        pointCloud[i,3] = doppler
    outputDict['numDetectedPoints'], outputDict['pointCloud'] = numPoints, pointCloud


# Point Cloud Ext TLV from SDK for xWRL6432
def parsePointCloudExtTLV(tlvData, tlvLength, outputDict):
    pointCloud = outputDict['pointCloud']
    pUnitStruct = '4f2h' # Units for the 5 results to decompress them
    pointStruct = '4h2B' # x y z doppler snr noise
    pUnitSize = struct.calcsize(pUnitStruct)
    pointSize = struct.calcsize(pointStruct)

    # Parse the decompression factors
    try:
        pUnit = struct.unpack(pUnitStruct, tlvData[:pUnitSize])
    except:
            log.error('Point Cloud TLV Parser Failed')
            outputDict['numDetectedPoints'], outputDict['pointCloud'] = 0, pointCloud
    # Update data pointer
    tlvData = tlvData[pUnitSize:]

    # Parse each point
    numPoints = int((tlvLength-pUnitSize)/pointSize)
    for i in range(numPoints):
        try:
            x, y, z, doppler, snr, noise = struct.unpack(pointStruct, tlvData[:pointSize])
        except:
            numPoints = i
            log.error('Point Cloud TLV Parser Failed')
            break
        
        tlvData = tlvData[pointSize:]
        # Decompress values
        pointCloud[i,0] = x * pUnit[0]          # x
        pointCloud[i,1] = y * pUnit[0]          # y
        pointCloud[i,2] = z * pUnit[0]          # z
        pointCloud[i,3] = doppler * pUnit[1]    # Doppler
        pointCloud[i,4] = snr * pUnit[2]        # SNR
        pointCloud[i,5] = noise * pUnit[3]      # Noise
    outputDict['numDetectedPoints'], outputDict['pointCloud'] = numPoints, pointCloud


# Side info TLV from SDK
def parseSideInfoTLV(tlvData, tlvLength, outputDict):
    pointCloud = outputDict['pointCloud']
    pointStruct = '2H'  # Two unsigned shorts: SNR and Noise
    pointStructSize = struct.calcsize(pointStruct)
    numPoints = int(tlvLength/pointStructSize)

    for i in range(numPoints):
        try:
            snr, noise = struct.unpack(pointStruct, tlvData[:pointStructSize])
        except:
            numPoints = i
            log.error('Side Info TLV Parser Failed')
            break
        tlvData = tlvData[pointStructSize:]
        # SNR and Noise are sent as uint16_t which are measured in 0.1 dB Steps
        pointCloud[i,4] = snr * 0.1
        pointCloud[i,5] = noise * 0.1
    outputDict['pointCloud'] = pointCloud

# Range Profile Parser
def parseRangeProfileTLV(tlvData, tlvLength, outputDict):
    rangeProfile = []
    # rangeDataStruct = 'I' # Every range bin gets a uint32_t
    rangeDataStruct = 'h'
    rangeDataSize = struct.calcsize(rangeDataStruct)

    numRangeBins = int(len(tlvData)/rangeDataSize)
    for i in range(numRangeBins):
        # Read in single range bin data
        try:
            rangeBinData = struct.unpack(rangeDataStruct, tlvData[:rangeDataSize])
        except:
            log.error(f'Range Profile TLV Parser Failed To Parse Range Bin Number ${i}')
            break
        rangeProfile.append(rangeBinData[0])
        # Move to next value
        tlvData = tlvData[rangeDataSize:]
    outputDict['rangeProfile'] = rangeProfile


# Spherical Point Cloud TLV Parser
def parseSphericalPointCloudTLV(tlvData, tlvLength, outputDict):
    pointCloud = outputDict['pointCloud']
    pointStruct = '4f'  # Range, Azimuth, Elevation, and Doppler
    pointStructSize = struct.calcsize(pointStruct)
    numPoints = int(tlvLength/pointStructSize)

    for i in range(numPoints):
        try:
            rng, azimuth, elevation, doppler = struct.unpack(pointStruct, tlvData[:pointStructSize])
        except:
            numPoints = i
            log.error('Point Cloud TLV Parser Failed')
            break
        tlvData = tlvData[pointStructSize:]
        pointCloud[i,0] = rng
        pointCloud[i,1] = azimuth
        pointCloud[i,2] = elevation
        pointCloud[i,3] = doppler
    
    # Convert from spherical to cartesian
    pointCloud[:,0:3] = sphericalToCartesianPointCloud(pointCloud[:, 0:3])
    outputDict['numDetectedPoints'], outputDict['pointCloud'] =  numPoints, pointCloud

# Point Cloud TLV from Capon Chain
def parseCompressedSphericalPointCloudTLV(tlvData, tlvLength, outputDict):
    pointCloud = outputDict['pointCloud']
    pUnitStruct = '5f' # Units for the 5 results to decompress them
    pointStruct = '2bh2H' # Elevation, Azimuth, Doppler, Range, SNR
    pUnitSize = struct.calcsize(pUnitStruct)
    pointSize = struct.calcsize(pointStruct)

    # Parse the decompression factors
    try:
        pUnit = struct.unpack(pUnitStruct, tlvData[:pUnitSize])
    except:
            log.error('Point Cloud TLV Parser Failed')
            outputDict['numDetectedPoints'], outputDict['pointCloud'] = 0, pointCloud
    # Update data pointer
    tlvData = tlvData[pUnitSize:]

    # Parse each point
    numPoints = int((tlvLength-pUnitSize)/pointSize)
    for i in range(numPoints):
        try:
            elevation, azimuth, doppler, rng, snr = struct.unpack(pointStruct, tlvData[:pointSize])
        except:
            numPoints = i
            log.error('Point Cloud TLV Parser Failed')
            break
        
        tlvData = tlvData[pointSize:]
        if (azimuth >= 128):
            log.error('Az greater than 127')
            azimuth -= 256
        if (elevation >= 128):
            log.error('Elev greater than 127')
            elevation -= 256
        if (doppler >= 32768):
            log.error('Doppler greater than 32768')
            doppler -= 65536
        # Decompress values
        pointCloud[i,0] = rng * pUnit[3]          # Range
        pointCloud[i,1] = azimuth * pUnit[1]      # Azimuth
        pointCloud[i,2] = elevation * pUnit[0]    # Elevation
        pointCloud[i,3] = doppler * pUnit[2]      # Doppler
        pointCloud[i,4] = snr * pUnit[4]          # SNR

    # Convert from spherical to cartesian
    pointCloud[:,0:3] = sphericalToCartesianPointCloud(pointCloud[:, 0:3])
    outputDict['numDetectedPoints'] = numPoints
    outputDict['pointCloud'] = pointCloud


def parseTrackTLV(tlvData, tlvLength, outputDict):
    targetStruct = 'I27f'
    targetSize = struct.calcsize(targetStruct)
    numDetectedTargets = int(tlvLength/targetSize)
    targets = np.empty((numDetectedTargets,16))
    for i in range(numDetectedTargets):
        try:
            targetData = struct.unpack(targetStruct,tlvData[:targetSize])
        except:
            log.error('Target TLV parsing failed')
            outputDict['numDetectedTracks'], outputDict['trackData'] = 0, targets

        targets[i,0] = targetData[0] # Target ID
        targets[i,1] = targetData[1] # X Position
        targets[i,2] = targetData[2] # Y Position
        targets[i,3] = targetData[3] # Z Position
        targets[i,4] = targetData[4] # X Velocity
        targets[i,5] = targetData[5] # Y Velocity
        targets[i,6] = targetData[6] # Z Velocity
        targets[i,7] = targetData[7] # X Acceleration
        targets[i,8] = targetData[8] # Y Acceleration
        targets[i,9] = targetData[9] # Z Acceleration
        targets[i,10] = targetData[26] # G
        targets[i,11] = targetData[27] # Confidence Level
        
        # Throw away EC
        tlvData = tlvData[targetSize:]
    outputDict['numDetectedTracks'], outputDict['trackData'] = numDetectedTargets, targets


# Vital Signs
def parseVitalSignsTLV (tlvData, tlvLength, outputDict):
    vitalsStruct = '2H33f'
    vitalsSize = struct.calcsize(vitalsStruct)
    
    # Initialize struct in case of error
    vitalsOutput = {}
    vitalsOutput ['id'] = 999
    vitalsOutput ['rangeBin'] = 0
    vitalsOutput ['breathDeviation'] = 0
    vitalsOutput ['heartRate'] = 0
    vitalsOutput ['breathRate'] = 0
    vitalsOutput ['heartWaveform'] = []
    vitalsOutput ['breathWaveform'] = []

    # Capture data for active patient
    try:
        vitalsData = struct.unpack(vitalsStruct, tlvData[:vitalsSize])
    except:
        log.error('ERROR: Vitals TLV Parsing Failed')
        outputDict['vitals'] = vitalsOutput
    
    # Parse this patient's data
    vitalsOutput ['id'] = vitalsData[0]
    vitalsOutput ['rangeBin'] = vitalsData[1]
    vitalsOutput ['breathDeviation'] = vitalsData[2]
    vitalsOutput ['heartRate'] = vitalsData[3]
    vitalsOutput ['breathRate'] = vitalsData [4]
    vitalsOutput ['heartWaveform'] = np.asarray(vitalsData[5:20])
    vitalsOutput ['breathWaveform'] = np.asarray(vitalsData[20:35])

    # Advance tlv data pointer to end of this TLV
    tlvData = tlvData[vitalsSize:]
    outputDict['vitals'] = vitalsOutput



def parseRahmTLV (tlvData, tlvLength, outputDict):
    rahm = []
    # rangeDataStruct = 'I' # Every range bin gets a uint32_t
    rahmDataStruct = '2h'
    rahmDataSize = struct.calcsize(rahmDataStruct)

    numRahmBins = int(len(tlvData)/rahmDataSize)
    for i in range(numRahmBins):
        # Read in single range bin data
        try:
            # rangeBinData = struct.unpack(rahmDataStruct, tlvData[:rahmDataSize])
            imagRahm, realRahm = struct.unpack(rahmDataStruct, tlvData[:rahmDataSize])
        except:
            log.error(f'RAHM TLV Parser Failed To Parse Range Bin Number ${i}')
            break
        rahm.append(complex(realRahm, imagRahm))
        # Move to next value
        tlvData = tlvData[rahmDataSize:]
    
    outputDict['RAHM'] = rahm




def parseDopplerTLV (tlvData, tlvLength, outputDict):
    rdhm = []
    # rangeDataStruct = 'I' # Every range bin gets a uint32_t
    rdhmDataStruct = 'H'
    rdhmDataSize = struct.calcsize(rdhmDataStruct)

    numRdhmBins = int(len(tlvData)/rdhmDataSize)
    for i in range(numRdhmBins):
        # Read in single range bin data
        try:
            # rangeBinData = struct.unpack(rahmDataStruct, tlvData[:rahmDataSize])
            rdhmSample = struct.unpack(rdhmDataStruct, tlvData[:rdhmDataSize])
        except:
            log.error(f'RAHM TLV Parser Failed To Parse Range Bin Number ${i}')
            break
        rdhm.append(rdhmSample[0])
        # Move to next value
        tlvData = tlvData[rdhmDataSize:]


    outputDict['RDHM'] = rdhm