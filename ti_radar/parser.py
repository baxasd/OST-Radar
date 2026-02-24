import struct
import logging
import numpy as np

log = logging.getLogger(__name__)

# --- TLV (Type-Length-Value) Identifiers ---
TLV_DETECTED_POINTS = 1
TLV_RANGE_DOPPLER_HEAT_MAP = 5

# Main Frame Header: 1 unsigned long long (magic word), 8 unsigned ints
FRAME_HEADER_STRUCT = 'Q8I'
FRAME_HEADER_LEN = struct.calcsize(FRAME_HEADER_STRUCT)
TLV_HEADER_LEN = 8

def parsePointCloudTLV(tlvData, tlvLength, outputDict):
    """Placeholder for future 3D point cloud parsing"""
    pass

def parseDopplerTLV(tlvData, tlvLength, outputDict):
    """Instantly maps byte buffer to uint16 numpy array for maximum speed"""
    try:
        outputDict['RDHM'] = np.frombuffer(tlvData, dtype=np.uint16).copy()
    except Exception as e:
        log.error(f'RDHM TLV Parser Failed: {e}')

# Route TLV types to their respective parser functions
PARSER_FUNCTIONS = {
    TLV_DETECTED_POINTS: parsePointCloudTLV,
    TLV_RANGE_DOPPLER_HEAT_MAP: parseDopplerTLV
}

def parse_standard_frame(frameData):
    """Parses raw byte array of a single frame into a structured dictionary"""
    outputDict = {'error': 0, 'pointCloud': None, 'RDHM': None}

    if len(frameData) < FRAME_HEADER_LEN:
        outputDict['error'] = 1
        return outputDict

    try:
        header_data = struct.unpack(FRAME_HEADER_STRUCT, frameData[:FRAME_HEADER_LEN])
        # Unpack indices: 4 is frameNum, 7 is numTLVs
        outputDict['frameNum'] = header_data[4]
        numTLVs = header_data[7]
    except struct.error:
        outputDict['error'] = 1
        return outputDict

    frameData = frameData[FRAME_HEADER_LEN:]

    for _ in range(numTLVs):
        # Ensure enough bytes exist for TLV header
        if len(frameData) < TLV_HEADER_LEN:
            break
            
        tlvType, tlvLength = struct.unpack('2I', frameData[:TLV_HEADER_LEN])
        frameData = frameData[TLV_HEADER_LEN:]

        # Prevent buffer overflows if frame is truncated
        if len(frameData) < tlvLength:
            break 

        if tlvType in PARSER_FUNCTIONS:
            PARSER_FUNCTIONS[tlvType](frameData[:tlvLength], tlvLength, outputDict)

        frameData = frameData[tlvLength:]

    return outputDict