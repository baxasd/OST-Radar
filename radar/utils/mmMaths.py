import numpy as np


def RAHM_fft(azimRangeCompArray, numAzBins):

    R = np.fft.fftshift(np.fft.fft(azimRangeCompArray,axis=1,n=numAzBins),axes=1)

    return R


def sphericalToCartesianPointCloud(sphericalPointCloud):
    """
    Convert 3D Spherical Points to Cartesian
    Assumes sphericalPointCloud is an numpy array with at LEAST 3 dimensions
    Order should be Range, Elevation, Azimuth
    """
    import logging
    log = logging.getLogger(__name__)
    
    shape = sphericalPointCloud.shape
    cartesianPointCloud = sphericalPointCloud.copy()
    if (shape[1] < 3):
        log.error('Error: Failed to convert spherical point cloud to cartesian due to numpy array with too few dimensions')
        return sphericalPointCloud

    # Compute X
    # Range * sin (azimuth) * cos (elevation)
    cartesianPointCloud[:,0] = sphericalPointCloud[:,0] * np.sin(sphericalPointCloud[:,1]) * np.cos(sphericalPointCloud[:,2]) 
    # Compute Y
    # Range * cos (azimuth) * cos (elevation)
    cartesianPointCloud[:,1] = sphericalPointCloud[:,0] * np.cos(sphericalPointCloud[:,1]) * np.cos(sphericalPointCloud[:,2]) 
    # Compute Z
    # Range * sin (elevation)
    cartesianPointCloud[:,2] = sphericalPointCloud[:,0] * np.sin(sphericalPointCloud[:,2])
    return cartesianPointCloud