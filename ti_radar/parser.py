import struct
import logging
import numpy as np

log = logging.getLogger(__name__)

# TLV type identifiers (TI mmWave SDK)
TLV_RANGE_DOPPLER_HEAT_MAP = 5

# Frame header: magic word (Q) + 8 unsigned ints
_HEADER_FMT = "Q8I"
_HEADER_LEN = struct.calcsize(_HEADER_FMT)
_TLV_HDR_LEN = 8  # 2 x uint32: tlvType, tlvLength


def _parse_rdhm(data: bytes, length: int, out: dict):
    """Maps raw bytes directly to a uint16 numpy array (range-doppler heatmap)."""
    try:
        out["RDHM"] = np.frombuffer(data, dtype=np.uint16).copy()
    except Exception as e:
        log.error(f"RDHM parse failed: {e}")


# Only register TLVs that are actually handled.
# FIX: removed no-op parsePointCloudTLV from dispatch table — it was
# consuming a dict lookup and function call on every frame for zero benefit.
_PARSERS = {
    TLV_RANGE_DOPPLER_HEAT_MAP: _parse_rdhm,
}


def parse_standard_frame(data: bytes) -> dict:
    """Parses a raw TI mmWave frame into a structured dict."""
    out = {"error": 0, "RDHM": None}

    if len(data) < _HEADER_LEN:
        out["error"] = 1
        return out

    try:
        header = struct.unpack(_HEADER_FMT, data[:_HEADER_LEN])
        out["frameNum"] = header[4]
        num_tlvs        = header[7]
    except struct.error:
        out["error"] = 1
        return out

    data = data[_HEADER_LEN:]

    for _ in range(num_tlvs):
        if len(data) < _TLV_HDR_LEN:
            break
        tlv_type, tlv_len = struct.unpack("2I", data[:_TLV_HDR_LEN])
        data = data[_TLV_HDR_LEN:]
        if len(data) < tlv_len:
            break
        if tlv_type in _PARSERS:
            _PARSERS[tlv_type](data[:tlv_len], tlv_len, out)
        data = data[tlv_len:]

    return out