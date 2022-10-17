
# Author: 
# Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date:
# 2021-05-06

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .ocv_torch import ( ocv_2_torch, torch_2_ocv )
from ..mvs_utils.ftensor import FTensor, f_eye

IDENTITY_ROT = f_eye(3, f0='raw', f1='fisheye', rotation=True, dtype=torch.float32)

# INTER_MAP = {
#     'nearest': cv2.INTER_NEAREST,
#     'linear': cv2.INTER_LINEAR,
# }

INTER_MAP = {
    'nearest': 'nearest',
    'linear': 'bilinear',
}

def input_2_torch(img, device):
    '''
    img can be a single image represented as a NumPy array, or it could
    be a collection of NumPy arrays, or it could already be a PyTorch Tensor.
    '''
    
    if isinstance(img, (list, tuple)):
        flag_uint8 = img[0].dtype == np.uint8
        return torch.cat( [ ocv_2_torch(i, keep_dtype=False) for i in img ], dim=0 ).to(device=device), flag_uint8
    else:
        return ocv_2_torch(img, keep_dtype=False).to(device=device), img.dtype == np.uint8

def torch_2_output(t, flag_uint8=True):
    if flag_uint8:
        return torch_2_ocv(t, scale=True, dtype=np.uint8)
    else:
        return torch_2_ocv(t, scale=False, dtype=np.float32)

class PlanarAsBase(object):
    def __init__(self, fov, camera_model, R_raw_fisheye=IDENTITY_ROT):
        '''
        Arguments:
        fov (float): Full FoV of the lens in degrees.
        camera_model (camera_model.CameraModel): This is used if dsc is None. 
        R_raw_fisheye (FTensor): The orientation of the fisheye camera.
        '''
        # TODO: Fixe the naming of R_raw_fisheye. Target can be any kind of image.
        super(PlanarAsBase, self).__init__()
        
        self.fov = fov # Degree.

        self._device = 'cpu'

        self.camera_model = camera_model
        self.camera_model.device = self._device
        self.shape = self.camera_model.shape

        # The rotation matrix of the fisheye camera.
        # The notation is R_<to>_<from> or R_<measured in>_<be measured>.
        # This rotation matrix is the orientation of the fisheye camera w.r.t
        # the frame where we take the raw images. And the orientation is measured
        # in the raw image frame.
        self.R_raw_fisheye = R_raw_fisheye

    @property
    def align_corners(self):
        return False

    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, device):
        self._device = device
        self.camera_model.device = device
        self.R_raw_fisheye = self.R_raw_fisheye.to(device=device)

    def mesh_grid_pixels(self, shape, dimensionless=False, flag_flatten=False):
        '''Get a mesh grid of the pixel coordinates. 
        shape (two-element): H, W.
        '''

        x = torch.arange( shape[1], dtype=torch.float32, device=self.device ) + 0.5 # W
        y = torch.arange( shape[0], dtype=torch.float32, device=self.device ) + 0.5 # H

        xx, yy = torch.meshgrid(x, y, indexing='xy')
        
        # Make contiguous.
        xx = xx.contiguous()
        yy = yy.contiguous()
        
        if dimensionless:
            xx = xx / shape[1] * 2 - 1
            yy = yy / shape[0] * 2 - 1

        if ( flag_flatten ):
            return xx.view((-1)), yy.view((-1))
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
        pixel_coor = torch.stack( (xx, yy), dim=0 ) # 2xN

        xyz, valid_mask = \
            self.camera_model.pixel_2_ray(pixel_coor)
        
        # xyz and valid_mask are torch.Tensor.
        # xyz = xyz.astype(np.float32)
        
        xyz = FTensor(xyz, f0='fisheye', f1=None).to(dtype=torch.float32)
        
        # Change of reference frame.
        xyz = self.R_raw_fisheye @ xyz

        return xyz, valid_mask
    
    def compute_8_way_sample_msr_diff(self, s, valid_mask):
        '''
        This function computes the 8-way mean-square-root of the sampling location
        differences specified by s. 
        
        s (Tensor): The sampling location. N x H x W x 2.
        valid_mask: N x 1 x H x W.
        
        Returns:
        A N x 1 x H x W array showing the mean of 8-way msr diff. Measured in the unit of s.
        '''
        
        assert s.ndim == 4, f's.ndim = {s.ndim}'
        
        s = s.permute((0, 3, 1, 2))
        N, _, H, W = s.shape
        
        # Augment the s array by 1s.
        all_ones = torch.zeros(( N, 1, H, W ), dtype=s.dtype, device=self.device)
        all_ones[valid_mask] = 1

        a = torch.cat( ( s, all_ones ), dim=1 )
        
        # Make a sampling grid.
        xx, yy = self.mesh_grid_pixels( (H, W), dimensionless=True )
        grid = torch.stack( (xx, yy), dim=-1 ).unsqueeze(0).repeat(N, 1, 1, 1)
        
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
        
        shifts = torch.Tensor(shifts).to(dtype=torch.float32, device=self.device)
        shifts[:, 0] /= W
        shifts[:, 1] /= H
        
        acc_d = torch.zeros((N, 1, H, W), dtype=torch.float32, device=self.device)

        for shift in shifts:
            grid_shifted = grid + shift

            s_a = F.grid_sample( a, 
                                 grid_shifted, 
                                 mode='nearest', 
                                 align_corners=self.align_corners, 
                                 padding_mode='reflection' )
            
            d = ( s[:, :2, :, :] - s_a[:, :2, :, :] ) * s_a[:, 2, :, :].unsqueeze(1)
            d = torch.linalg.norm( d, dim=1, keepdim=True )
            acc_d = d + acc_d
            
        return acc_d / shifts.shape[0]