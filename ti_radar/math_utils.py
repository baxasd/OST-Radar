import numpy as np
import logging

log = logging.getLogger(__name__)

def RAHM_fft(azimRangeCompArray, numAzBins):
    """Computes the 2D Fast Fourier Transform for Range-Azimuth (Angle) Heatmaps."""
    return np.fft.fftshift(np.fft.fft(azimRangeCompArray, axis=1, n=numAzBins), axes=1)

def sphericalToCartesianPointCloud(sphericalPointCloud):
    """
    Converts raw spherical radar coordinates into standard 3D space.
    Input format : (Range, Elevation, Azimuth)
    Output format: (X, Y, Z) in meters
    """
    if sphericalPointCloud.shape[1] < 3:
        log.error('Failed to convert spherical point cloud: too few dimensions')
        return sphericalPointCloud

    cartesianPointCloud = sphericalPointCloud.copy()
    
    ranges = sphericalPointCloud[:, 0]
    elevations = sphericalPointCloud[:, 1]
    azimuths = sphericalPointCloud[:, 2]

    # X = Distance projected horizontally, then mapped to left/right angle
    cartesianPointCloud[:, 0] = ranges * np.sin(azimuths) * np.cos(elevations) 
    
    # Y = Distance projected straight forward (Depth)
    cartesianPointCloud[:, 1] = ranges * np.cos(azimuths) * np.cos(elevations) 
    
    # Z = Distance projected vertically (Height)
    cartesianPointCloud[:, 2] = ranges * np.sin(elevations)
    
    return cartesianPointCloud