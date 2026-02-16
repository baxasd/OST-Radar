import struct
import math
import logging
import numpy as np
from .math_utils import sphericalToCartesianPointCloud

log = logging.getLogger(__name__)

# --- TLV Defines ---
TLV_DETECTED_POINTS = 1
TLV_RANGE_DOPPLER_HEAT_MAP = 5

def parsePointCloudTLV(tlvData, tlvLength, outputDict):
    # (Keep your existing point cloud parser here if you have it)
    pass

def parseDopplerTLV(tlvData, tlvLength, outputDict):
    try:
        # SUPER FAST UNPACKING: 
        # Instantly reads the byte array as uint16 instead of a slow for-loop
        rdhm = np.frombuffer(tlvData, dtype=np.uint16).tolist()
        outputDict['RDHM'] = rdhm
    except Exception as e:
        log.error(f'RDHM TLV Parser Failed: {e}')

PARSER_FUNCTIONS = {
    TLV_DETECTED_POINTS: parsePointCloudTLV,
    TLV_RANGE_DOPPLER_HEAT_MAP: parseDopplerTLV
}

def parse_standard_frame(frameData):
    """Parses raw byte array of a single frame into a dictionary"""
    headerStruct = 'Q8I'
    frameHeaderLen = struct.calcsize(headerStruct)
    tlvHeaderLength = 8
    
    outputDict = {'error': 0, 'pointCloud': None, 'RDHM': None}

    # Verify we even have enough bytes for the main header
    if len(frameData) < frameHeaderLen:
        outputDict['error'] = 1
        return outputDict

    try:
        magic, version, totalPacketLen, platform, frameNum, timeCPUCycles, numDetectedObj, numTLVs, subFrameNum = struct.unpack(headerStruct, frameData[:frameHeaderLen])
    except Exception as e:
        outputDict['error'] = 1
        return outputDict

    frameData = frameData[frameHeaderLen:]
    outputDict['frameNum'] = frameNum

    for _ in range(numTLVs):
        # CRITICAL FIX 1: Check if we have enough bytes for the TLV Header (8 bytes)
        if len(frameData) < tlvHeaderLength:
            break
            
        tlvType, tlvLength = struct.unpack('2I', frameData[:tlvHeaderLength])
        frameData = frameData[tlvHeaderLength:]

        # CRITICAL FIX 2: Check if we have enough bytes for the TLV Payload
        if len(frameData) < tlvLength:
            break # Drop the rest of this corrupted frame safely

        # Parse valid TLV
        if tlvType in PARSER_FUNCTIONS:
            PARSER_FUNCTIONS[tlvType](frameData[:tlvLength], tlvLength, outputDict)

        # Move to next TLV
        frameData = frameData[tlvLength:]

    return outputDict