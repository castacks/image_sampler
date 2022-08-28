
# Author: 
# Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date:
# 2021-05-06

import cv2
import numpy as np
import torch

from ..mvs_utils.ftensor import FTensor, f_eye

# IDENTITY_ROT = np.eye(3, dtype=np.float32)
IDENTITY_ROT = f_eye(3, f0='raw', f1='fisheye', rotation=True, dtype=torch.float32)

INTER_MAP = {
    'nearest': cv2.INTER_NEAREST,
    'linear': cv2.INTER_LINEAR,
}

class PlanarAsBase(object):
    def __init__(self, fov, camera_model, R_raw_fisheye=IDENTITY_ROT):
        '''
        Arguments:
        fov (float): Full FoV of the lens in degrees.
        camera_model (camera_model.CameraModel): This is used if dsc is None. 
        R_raw_fisheye (FTensor): The orientation of the fisheye camera.
        '''
        super(PlanarAsBase, self).__init__()
        
        self.fov = fov # Degree.

        self.flag_cuda = False

        self.camera_model = camera_model
        self.shape = self.camera_model.shape

        # The rotation matrix of the fisheye camera.
        # The notation is R_<to>_<from> or R_<measured in>_<be measured>.
        # This rotation matrix is the orientation of the fisheye camera w.r.t
        # the frame where we take the raw images. And the orientation is measured
        # in the raw image frame.
        self.R_raw_fisheye = R_raw_fisheye

    def enable_cuda(self):
        self.flag_cuda = True

    def mesh_grid_pixels(self, shape, dimensionless=False, flag_flatten=False):
        '''Get a mesh grid of the pixel coordinates. 
        shape (two-element): H, W.
        '''

        x = np.arange( shape[1], dtype=np.int32 ) # W
        y = np.arange( shape[0], dtype=np.int32 ) # H

        xx, yy = np.meshgrid(x, y)
        
        if dimensionless:
            xx = xx / ( shape[1] - 1 )
            yy = yy / ( shape[0] - 1 )

        if ( flag_flatten ):
            return xx.reshape((-1)), yy.reshape((-1))
        else:
            return xx, yy

    def get_xyz(self):
        '''
        Compute the ray vectors for all valid pixels in the fisheye image.
        A ray vector is represented as a unit vector.
        All ray vectors will be transformed such that their coordiantes are
        measured in the raw frame where z-forward, x-right, and y-downward.
        
        Some pixels are not going to have valide rays. There is a mask of valid
        pixels that is also returned by this function.
        '''
        # The pixel coordinates.
        xx, yy = self.mesh_grid_pixels(self.shape, flag_flatten=True) # 1D.
        pixel_coor = np.stack( (xx, yy), axis=0 ) # 2xN

        xyz, valid_mask = \
            self.camera_model.pixel_2_ray(pixel_coor)
        
        # xyz and valid_mask are torch.Tensor.
        # xyz = xyz.astype(np.float32)
        
        xyz = FTensor(xyz, f0='fisheye', f1=None).to(dtype=torch.float32)
        
        # Change of reference frame.
        xyz = self.R_raw_fisheye @ xyz

        # Conver back to numpy.
        return xyz.cpu().numpy(), valid_mask.cpu().numpy()
    
    def compute_8_way_sample_msr_diff(self, s, valid_mask):
        '''
        This function computes the 8-way mean-square-root of the sampling location
        differences specified by s. 
        
        s (array): The sampling location. H x W x 2.
        
        Returns:
        A H x W array showing the mean of 8-way msr diff.
        '''
        
        assert s.ndim == 3, f's.ndim = {s.ndim}'
        
        # Augment the s array by 1s.
        all_ones = np.zeros(s.shape[:2], dtype=s.dtype)
        all_ones[valid_mask] = 1
        all_ones = np.expand_dims(all_ones, axis=-1)
        
        a = np.concatenate( ( s, all_ones ), axis=2 )
        
        # Make a sampling grid.
        xx, yy = self.mesh_grid_pixels( s.shape[:2] )
        
        shifts = [
            [  1,  0 ], # 0
            [  1,  1 ], # 1
            [  0,  1 ], # 2
            [ -1,  1 ], # 3
            [ -1,  0 ], # 4
            [ -1, -1 ], # 5
            [  0, -1 ], # 6
            [  1, -1 ], # 7
        ]
        
        acc_d = 0
        H, W = s.shape[:2]
        
        for shift in shifts:
            s_xx = ( xx + shift[0] ).astype(np.float32)
            s_yy = ( yy + shift[1] ).astype(np.float32)

            s_a = cv2.remap( a, s_xx, s_yy, interpolation=cv2.INTER_NEAREST )
            
            d = ( s[:, :, :2] - s_a[:, :, :2] ) * s_a[:, :, 2].reshape( (H, W, 1) )
            d = np.linalg.norm(d, axis=2)
            acc_d = d + acc_d
            
        return acc_d / len(shifts)