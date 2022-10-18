
# Author: 
# Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date:
# 2022-06-17

import cupy
import cv2
import math
import numpy as np
import time
import torch
import torch.nn.functional as F

# Local package.
from .planar_as_base import ( PlanarAsBase, IDENTITY_ROT, INTER_MAP, torch_2_output )
from .register import (SAMPLERS, register)

from .six_images_common import (FRONT, 
                               OFFSETS, 
                               make_image_cross_npy, make_image_cross_torch )
from .six_images_numba import ( sample_coor, sample_coor_cuda )

@register(SAMPLERS)
class SixPlanarNumba(PlanarAsBase):
    def __init__(self, fov, camera_model, R_raw_fisheye=IDENTITY_ROT):
        '''
        Arguments:
        fov (float): Full FoV of the lens in degrees.
        camera_model: A camera model for the fisheye camera.
        '''
        super().__init__(fov, camera_model=camera_model, R_raw_fisheye=R_raw_fisheye)
        
        # The 3D coordinates of the hyper-surface.
        # xyz, valid_mask = self.get_xyz(back_shift_pixel=True)
        xyz, valid_mask = self.get_xyz(back_shift_pixel=False)
        self.xyz = xyz.cpu().numpy()
        self.valid_mask = valid_mask.cpu().numpy().astype(np.bool)
        
        self.INTER_MAP_OCV = {
            'linear': cv2.INTER_LINEAR,
            'nearest': cv2.INTER_NEAREST
        }

    @PlanarAsBase.device.setter
    def device(self, device):
        PlanarAsBase.device.fset(self, device)
        # Do nothing.

    def __repr__(self):
        s = f'''SixPlanarNumba
fov = {self.fov}
shape = {self.shape}
'''
        return s

    def __call__(self, imgs, interpolation='linear', invalid_pixel_value=127):
        '''
        Arguments:
        imgs (list of arrays): The six images in the order of front, right, bottom, left, top, and back.
        interpolation (str): The interpolation method, could be linear or nearest.
        invalid_pixel_value (int): The value of the invalid pixel. For RGB images, it is normally 127. For depth, it is -1.
        
        Returns:
        The generated fisheye image.
        '''

        global FRONT

        # Get the original shape of the input images.
        img_shape = np.array(imgs[FRONT].shape[:2], dtype=np.float32).reshape((2, 1))
        H, W = img_shape
        
        # Make the image cross.
        img_cross = make_image_cross_npy( imgs )
        # cv2.imwrite('img_cross.png', img_cross)

        # Get the sample locations.
        if ( self.device != 'cpu' ):
            m, offsets = sample_coor_cuda( self.xyz, self.valid_mask )
        else:
            m, offsets = sample_coor(self.xyz, self.valid_mask)

        # We need to properly scale the dimensionless values in m to use cv2.remap().
        m[0, :] = W / ( W - 0.5 ) * ( m[0, :] - 0.5 ) + 0.5
        m[1, :] = H / ( H - 0.5 ) * ( m[1, :] - 0.5 ) + 0.5

        m = m * ( img_shape - 1 ) + offsets * img_shape

        mx = m[0, :].reshape(self.shape)
        my = m[1, :].reshape(self.shape)

        # Get the interpolation method.
        interp_method = self.INTER_MAP_OCV[interpolation]

        # Sample.
        sampled = cv2.remap( 
            img_cross, 
            mx, my, 
            interpolation=interp_method )

        # Apply gray color on invalid coordinates.
        invalid = np.logical_not(self.valid_mask).reshape(self.shape)
        sampled[invalid, ...] = invalid_pixel_value

        return sampled, self.valid_mask

    def compute_mean_samping_diff(self, img_shape):
        img_shape = np.array(img_shape, dtype=np.float32).reshape((2, 1))

        # Get the sample locations.
        if ( self.device != 'cpu' ):
            m, offsets = sample_coor_cuda( self.xyz, self.valid_mask )
        else:
            m, offsets = sample_coor(self.xyz, self.valid_mask)

        m = m * ( img_shape - 1 ) + offsets * img_shape

        mx = m[0, :].reshape(self.shape)
        my = m[1, :].reshape(self.shape)
        valid_mask = self.valid_mask.reshape(self.shape)

        # compute_8_way_sample_msr_diff only supports torch now.
        s = torch.from_numpy(np.stack( (mx, my), axis=-1 )).unsqueeze(0).to(self.device)
        vm = torch.from_numpy(valid_mask).unsqueeze(0).unsqueeze(0).to(self.device)
        d = self.compute_8_way_sample_msr_diff( s, vm )
        
        return torch_2_output(d, flag_uint8=False), valid_mask

@register(SAMPLERS)
class SixPlanarTorch(PlanarAsBase):
    def __init__(self, fov, camera_model, R_raw_fisheye=IDENTITY_ROT):
        '''
        Arguments:
        fov (float): Full FoV of the lens in degrees.
        camera_model: A camera model for the fisheye camera.
        '''
        super().__init__(fov, camera_model=camera_model, R_raw_fisheye=R_raw_fisheye)
        
        # The 3D coordinates of the hyper-surface.
        self.xyz, self.valid_mask = self.get_xyz()
        
        self.image_cross_layout = [3, 4]
        self.image_cross_layout_device = \
            torch.Tensor(self.image_cross_layout).to(dtype=torch.float32)
            
        self.OFFSETS_TORCH = torch.from_numpy(OFFSETS).to(dtype=torch.float32).permute((1,0)).contiguous()
        
        # === For the CuPy module. ===
        import os
        
        # Read the CUDA source.
        _CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
        with open( os.path.join( _CURRENT_PATH, 'six_images_cupy.cu' ), 'r' ) as fp:
            cuda_source = fp.read()
        
        # Compile the CUDA source.
        cupy_module = cupy.RawModule(code=cuda_source)
        self.sample_coor_cuda = cupy_module.get_function('cu_sample_coor')
        
        # Make a copy of xyz. Make it become Nx3.
        self.xyz_T = self.xyz.permute((1,0)).contiguous()
        
        self.cuda_block_size = 256
        self.cuda_grid_size = int( math.ceil( self.xyz_T.shape[0]*self.xyz_T.shape[1] / self.cuda_block_size ) )

    @PlanarAsBase.device.setter
    def device(self, device):
        PlanarAsBase.device.fset(self, device)
        
        self.xyz = self.xyz.to(device=self.device)
        self.valid_mask = self.valid_mask.to(device=self.device)
        self.image_cross_layout_device = self.image_cross_layout_device.to(device=self.device)
        self.OFFSETS_TORCH = self.OFFSETS_TORCH.to(device=self.device)
        self.xyz_T = self.xyz_T.to(device=self.device)

    def __repr__(self):
        s = f'''SixPlanarTorch
fov = {self.fov}
shape = {self.shape}
'''
        return s

    def __call__(self, imgs, interpolation='linear', invalid_pixel_value=127):
        '''
        Arguments:
        imgs (dict of arrays or list of dicts): The six images in the order of front, right, bottom, left, top, and back.
        interpolation (str): The interpolation method, could be linear or nearest.
        invalid_pixel_value (int): The value of the invalid pixel. For RGB images, it is normally 127. For depth, it is -1.
        
        Returns:
        The generated fisheye image. The image might be inside a list.
        '''

        global FRONT
        
        # Make the image cross.
        img_cross, flag_uint8 = make_image_cross_torch( imgs, device=self.device )

        # Get the sample locations.
        if ( self.device != 'cpu' ):
            start_time = time.time()
            # Allocate m and offsets.
            m = torch.zeros( (self.xyz_T.shape[0], 2), dtype=torch.float32, device=self.device )
            offsets = m.detach().clone()
            
            # Call the CUDA function.
            self.sample_coor_cuda(
                block=(self.cuda_block_size, ),
                grid=(self.cuda_grid_size, ),
                args=(
                    cupy.int32(self.xyz_T.shape[0]),
                    self.xyz_T.data_ptr(),
                    self.OFFSETS_TORCH.data_ptr(),
                    m.data_ptr(),
                    offsets.data_ptr()
                )
            )
            
            # Handle the valid mask.
            invalid_mask = torch.logical_not(self.valid_mask)
            m[invalid_mask, :] = -1 # NOTE: This might be a bug.
            d = cupy.cuda.Device()
            d.synchronize()
            print(f'Time for CUDA: {time.time() - start_time}s. ')
        else:
            m, offsets = sample_coor(self.xyz.cpu().numpy(), self.valid_mask.cpu().numpy().astype(np.bool))

        m[:, 0] = ( m[:, 0] + offsets[:, 0]) / self.image_cross_layout_device[1] * 2 - 1
        m[:, 1] = ( m[:, 1] + offsets[:, 1]) / self.image_cross_layout_device[0] * 2 - 1
        m = m.view( ( 1, *self.shape, 2 ) )
        
        N = img_cross.shape[0]
        sampled = F.grid_sample( 
                                img_cross, 
                                m.repeat((N, 1, 1, 1)), 
                                mode=INTER_MAP[interpolation], 
                                align_corners=self.align_corners )

        # Apply gray color on invalid coordinates.
        valid_mask = self.valid_mask.view(self.shape)
        invalid_mask = torch.logical_not(valid_mask)
        
        if flag_uint8:
            invalid_pixel_value /= 255.0
        
        sampled[..., invalid_mask] = invalid_pixel_value

        start_time = time.time()
        output_sampled = torch_2_output(sampled, flag_uint8)
        output_mask = valid_mask.cpu().numpy().astype(np.bool)
        print(f'Transfer from GPU to CPU: {time.time() - start_time}s. ')
        
        return output_sampled, output_mask

    def compute_mean_samping_diff(self, img_shape):
        # Get the sample locations.
        if ( self.device != 'cpu' ):
            # Allocate m and offsets.
            m = torch.zeros( (self.xyz_T.shape[0], 2), dtype=torch.float32, device=self.device )
            offsets = m.detach().clone()
            
            # Call the CUDA function.
            self.sample_coor_cuda(
                block=(self.cuda_block_size, ),
                grid=(self.cuda_grid_size, ),
                args=(
                    cupy.int32(self.xyz_T.shape[0]),
                    self.xyz_T.data_ptr(),
                    self.OFFSETS_TORCH.data_ptr(),
                    m.data_ptr(),
                    offsets.data_ptr()
                )
            )
            
            # Handle the valid mask.
            invalid_mask = torch.logical_not(self.valid_mask)
            m[invalid_mask, :] = -1 # NOTE: This might be a bug.
        else:
            m, offsets = sample_coor(self.xyz.cpu().numpy(), self.valid_mask.cpu().numpy().astype(np.bool))

        m = m + offsets
        m = m.view( ( 1, *self.shape, 2 ) )
        
        # Convert back to pixel coordinates.
        m[..., 0] *= img_shape[1]
        m[..., 1] *= img_shape[0]
        
        valid_mask = self.valid_mask.view(self.shape)

        d = self.compute_8_way_sample_msr_diff( m, valid_mask.unsqueeze(0).unsqueeze(0) )
        
        return torch_2_output(d, flag_uint8=False), valid_mask.cpu().numpy().astype(np.bool)