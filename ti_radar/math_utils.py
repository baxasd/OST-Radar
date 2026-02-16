import numpy as np
import logging

log = logging.getLogger(__name__)

def RAHM_fft(azimRangeCompArray, numAzBins):
    """Computes FFT for Range Azimuth Heatmap"""
    return np.fft.fftshift(np.fft.fft(azimRangeCompArray, axis=1, n=numAzBins), axes=1)

def sphericalToCartesianPointCloud(sphericalPointCloud):
    """
    Convert 3D Spherical Points (Range, Elevation, Azimuth) to Cartesian (X, Y, Z)
    """
    shape = sphericalPointCloud.shape
    cartesianPointCloud = sphericalPointCloud.copy()
    if shape[1] < 3:
        log.error('Failed to convert spherical point cloud: too few dimensions')
        return sphericalPointCloud

    # X = Range * sin(azimuth) * cos(elevation)
    cartesianPointCloud[:, 0] = sphericalPointCloud[:, 0] * np.sin(sphericalPointCloud[:, 1]) * np.cos(sphericalPointCloud[:, 2]) 
    # Y = Range * cos(azimuth) * cos(elevation)
    cartesianPointCloud[:, 1] = sphericalPointCloud[:, 0] * np.cos(sphericalPointCloud[:, 1]) * np.cos(sphericalPointCloud[:, 2]) 
    # Z = Range * sin(elevation)
    cartesianPointCloud[:, 2] = sphericalPointCloud[:, 0] * np.sin(sphericalPointCloud[:, 2])
    
    return cartesianPointCloud