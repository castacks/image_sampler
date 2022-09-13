
import cv2
import numpy as np
import os
# from scipy.spatial.transform import Rotation as R

import torch
import torch.functional as F

from .planar_as_base import PlanarAsBase
from .register import (SAMPLERS, register)

@register(SAMPLERS)
class CameraModelRotation(PlanarAsBase):
    def __init__(self, camera_model_raw, camera_model_target, R_raw_fisheye):
        '''
        The raw image is a planer image that described by a camera model. 
        
        We create the target image by sampling from the raw image.

        R_raw_fisheye is the rotation matrix measured in the raw image frame. 
        The coordinates of a 3D point in the target camera image frame x_f can 
        be transformed to the point in the raw image frame x_p by
        x_p = R_raw_fisheye @ x_f.

        R_raw_fisheye is following the naming converntion. This means that CIF's orientation
        is measure in CPF.

        The camera model assumes that the raw image frame has its z-axis pointing forward,
        x-axis to the right, and y-axis downwards.

        Arguments:
        R_raw_fisheye (array): 3x3 rotation matrix. 
        camera_model_raw (camera_model.CameraModel): The camera model for the raw image. 
        camera_model_target (camera_model.CameraModel): The camera model for the target image. '''

        # TODO: Use torch overall.
        assert camera_model_raw.out_to_numpy, f'Currently only supports numpy version of raw camera model. '
        assert not camera_model_target.out_to_numpy, f'Currently only supports pytorch version of target camera model. '

        super().__init__(
            camera_model_target.fov_degree, camera_model=camera_model_target, R_raw_fisheye=R_raw_fisheye)

        self.camera_model_raw = camera_model_raw

        # Get the rays in xyz coordinates in the target camera image frame (CIF).
        # The rays has been already transformed to the target image frame.
        xyz, valid_mask_target = self.get_xyz()

        # Get the sample location in the raw image.
        pixel_coord_raw, valid_mask_raw = camera_model_raw.point_3d_2_pixel( xyz )

        # Reshape the sample location.
        self.sx = pixel_coord_raw[0, :].reshape( camera_model_target.ss.shape )
        self.sy = pixel_coord_raw[1, :].reshape( camera_model_target.ss.shape )

        # Compute the valid mask.
        self.invalid_mask = np.logical_not( np.logical_and( valid_mask_raw, valid_mask_target ) )
        pixel_coord_raw[:, self.invalid_mask] = -1

        self.invalid_mask_reshaped = self.invalid_mask.reshape( camera_model_target.ss.shape )

    def check_input_shape(self, img_shape):
        # Get the shape of the input image.
        H, W = img_shape[:2]
        ss = self.camera_model_raw.ss
        assert H == ss.H and W == ss.W, f'Wrong input image shape. Expect {ss}, got {img_shape[:2]}'

    def __call__(self, img, interpolation=cv2.INTER_LINEAR):
        self.check_input_shape(img.shape)

        # Sample.
        sampled = cv2.remap(img, self.sx, self.sy, interpolation=interpolation)

        # Handle invalid pixels.
        sampled[self.invalid_mask_reshaped, ...] = 0.0

        return sampled, self.invalid_mask_reshaped

    def compute_mean_samping_diff(self, img_shape):
        self.check_input_shape(img_shape)

        valid_mask = np.logical_not( self.invalid_mask )

        d = self.compute_8_way_sample_msr_diff( np.stack( (self.sx, self.sy), axis=-1 ), valid_mask )
        return d, valid_mask
