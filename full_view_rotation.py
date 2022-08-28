
import cv2
import numpy as np
import os
# from scipy.spatial.transform import Rotation as R

from .planar_as_base import PlanarAsBase
from .register import (SAMPLERS, register)

@register(SAMPLERS)
class FullViewRotation(PlanarAsBase):
    def __init__(self, camera_model, R_raw_fisheye):
        '''
        Note: Full view is the Unreal Engine's setting. It is NOT the same as the conventional 
        equirectangular projection. In Unreal Engine, the forward direction (that is where the 
        AirSim x-axis points when you set the camera pose) is located at the 3/4 of the image 
        width. 
        
        Note: R_raw_fisheye is the rotation matrix measured in the camera panorama frame. 
        The coordinates of a 3D point in the fisheye camera image frame x_f can 
        be transformed to the point in the camera panorama frame x_p by
        x_p = R_raw_fisheye @ x_f.

        R_raw_fisheye is following the naming converntion. This means that CIF's orientation
        is measure in CPF.

        Arguments:
        R_raw_fisheye (array): 3x3 rotation matrix. 
        camera_model (camera_model.CameraModel): The camera model. '''
        super().__init__(
            camera_model.fov_degree, camera_model=camera_model, R_raw_fisheye=R_raw_fisheye)

        # Get the longitude and latitude coordinates.
        self.lon_lat, invalid_mask = self.get_lon_lat()

        # Reshape out_of_fov.
        self.invalid_mask = invalid_mask.reshape(self.shape)

    def get_lon_lat(self):
        # Get the rays in xyz coordinates in the fisheye camera image frame (CIF).
        # The valid mask is saved in self.temp_valid_mask for compatibility concern.
        xyz, valid_mask = self.get_xyz()

        # # Already transformed.
        # # Transform the rays from the fisheye CIF to the CPF.
        # xyz = self.R_raw_fisheye @ xyz

        # The distance projected into the xz plane in the panorama frame.
        d = np.linalg.norm( xyz[[0, 2], :], axis=0 )

        # Longitude and latitude.
        lon_lat = np.zeros( (2, xyz.shape[1]), dtype=np.float32 )
        lon_lat[0, :] = np.pi - np.arctan2( xyz[2, :], xyz[0, :] ) # Longitude.
        lon_lat[1, :] = np.pi - np.arctan2( d, xyz[1, :] )         # Latitude.

        return lon_lat, np.logical_not( valid_mask )

    def pad(self, img):
        H, W = img.shape[:2]
        if ( img.ndim == 3 ):
            padded = np.zeros((H+1, W+1, img.shape[2]), dtype=img.dtype)
        else:
            padded = np.zeros((H+1, W+1), dtype=img.dtype)

        padded[:H, :W, ...] = img
        padded[ H, :W, ...] = img[0, :, ...]
        padded[:H,  W, ...] = img[:, 0, ...]
        padded[-1, -1, ...] = 0.5 * ( padded[-1, -2, ...] + padded[-2, -1, ...] )
        return padded

    def __call__(self, img, interpolation=cv2.INTER_LINEAR):
        # Get the shape of the input image.
        H, W = img.shape[:2]

        # The following does not make sense to me now. 20220701.
        # # Pad.
        # img = self.pad(img)
        # # Get the sample location.
        # # sx = ( lon_lat[0, :] / ( 2 * np.pi ) * (W-1) ).reshape(self.shape)
        # # sy = ( lon_lat[1, :] / np.pi * (H-1) ).reshape(self.shape)
        # # With padding.
        # sx = ( self.lon_lat[0, :] / ( 2 * np.pi ) * W ).reshape(self.shape)
        # sy = ( self.lon_lat[1, :] / np.pi * H ).reshape(self.shape)

        # Get the sample location. 20220701.
        sx = ( self.lon_lat[0, :] / ( 2 * np.pi ) * ( W - 1 ) ).reshape(self.shape)
        sy = ( self.lon_lat[1, :] / np.pi * ( H - 1 ) ).reshape(self.shape)

        # Sample.
        sampled = cv2.remap(img, sx, sy, interpolation=interpolation)

        # Handle invalid pixels.
        sampled[self.invalid_mask, ...] = 0.0

        return sampled, self.invalid_mask

    def compute_mean_samping_diff(self, img_shape):
        H, W = img_shape[:2]

        # Get the sample location. 20220701.
        sx = ( self.lon_lat[0, :] / ( 2 * np.pi ) * ( W - 1 ) ).reshape(self.shape)
        sy = ( self.lon_lat[1, :] / np.pi * ( H - 1 ) ).reshape(self.shape)

        valid_mask = np.logical_not( self.invalid_mask )

        d = self.compute_8_way_sample_msr_diff( np.stack( (sx, sy), axis=-1 ), valid_mask )
        return d, valid_mask

def read_image(fn):
    assert( os.path.isfile(fn) ), \
        f'{fn} does not exist. '

    return cv2.imread(fn, cv2.IMREAD_UNCHANGED)

def circle_mask(shape, c, r):
    '''
    shape (2-element): H, W.
    c (2-element): center coordinate, (x, y)
    r (float): the radius.
    '''

    # Get a meshgrid of pixel coordinates.
    H, W = shape[:2]
    x = np.arange(W, dtype=np.float32)
    y = np.arange(H, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing='xy')

    # Get the distance to the center.
    d = np.sqrt( ( xx - c[0] )**2 + ( yy - c[1] )**2 )

    return d <= r
