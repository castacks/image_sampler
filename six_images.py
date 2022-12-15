
# Author: 
# Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date:
# 2022-06-17

# import cupy
import cv2
import math
import numpy as np
import time
import torch
import torch.nn.functional as F

# Local package.
from .planar_as_base import ( PlanarAsBase, IDENTITY_ROT, INTER_MAP_OCV, INTER_MAP, torch_2_output )
from .register import (SAMPLERS, register)

from .six_images_common import (FRONT, 
                               OFFSETS, 
                               make_image_cross_npy, make_image_cross_torch )
from .six_images_numba import ( sample_coor, sample_coor_cuda )

def dummy_debug_callback(blend_factor_ori, blend_factor_sampled):
    pass
@register(SAMPLERS)
class SixPlanarNumba(PlanarAsBase):
    def __init__(self, fov, camera_model, R_raw_fisheye=IDENTITY_ROT, cached_raw_shape=(640, 640)):
        '''
        Arguments:
        fov (float): Full FoV of the lens in degrees.
        camera_model: A camera model for the fisheye camera.
        '''
        super().__init__(
            fov, 
            camera_model=camera_model, 
            R_raw_fisheye=R_raw_fisheye,
            cached_raw_shape=cached_raw_shape )
        
        # The 3D coordinates of the hyper-surface.
        # xyz, valid_mask = self.get_xyz(back_shift_pixel=True)
        xyz, valid_mask = self.get_xyz(back_shift_pixel=False)
        self.xyz = xyz.cpu().numpy()
        self.valid_mask = valid_mask.view(self.shape).cpu().numpy().astype(bool)
        
        # Explicity set the device to 'cuda' for better speed during construction.
        # Specifically, the call to self.update_remap_coordinates().
        self.device = 'cuda'
        
        # The remap coordinates.
        self.mx = None
        self.my = None
        self.update_remap_coordinates( self.cached_raw_shape )

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

    def update_remap_coordinates(self, support_shape):
        H, W = support_shape
        
        if not isinstance(support_shape, np.ndarray):
            support_shape = np.array( [H, W], dtype=np.float32 ).reshape((2, 1))

        # Get the sample locations.
        if ( self.device != 'cpu' ):
            m, offsets = sample_coor_cuda( self.xyz, self.valid_mask.reshape((-1,)) )
        else:
            m, offsets = sample_coor(self.xyz, self.valid_mask.reshape((-1,)))

        # We need to properly scale the dimensionless values in m to use cv2.remap().
        # Refer to self.convert_dimensionless_torch_grid_2_ocv_remap_format() for consistency.
        m[0, :] = W / ( W - 0.5 ) * ( m[0, :] - 0.5 ) + 0.5
        m[1, :] = H / ( H - 0.5 ) * ( m[1, :] - 0.5 ) + 0.5

        m = m * ( support_shape - 1 ) + offsets * support_shape

        self.mx = m[0, :].reshape(self.shape)
        self.my = m[1, :].reshape(self.shape)

    def check_shape_and_make_image_cross(self, imgs):
        global FRONT

        # Get the original shape of the input images.
        img_shape = np.array(imgs[FRONT].shape[:2], dtype=np.float32).reshape((2, 1))
        
        # Check the input shape.
        if not self.is_same_as_cached_shape( img_shape ):
            self.update_remap_coordinates( img_shape )
            self.cached_raw_shape = img_shape
        
        # Make the image cross.
        img_cross = make_image_cross_npy( imgs )
        # cv2.imwrite('img_cross.png', img_cross)
        
        return img_cross

    def __call__(self, imgs, interpolation='linear', invalid_pixel_value=127):
        '''
        Arguments:
        imgs (list of arrays): The six images in the order of front, right, bottom, left, top, and back.
        interpolation (str): The interpolation method, could be linear or nearest.
        invalid_pixel_value (int): The value of the invalid pixel. For RGB images, it is normally 127. For depth, it is -1.
        
        Returns:
        The generated fisheye image.
        '''

        global INTER_MAP_OCV
        
        img_cross = self.check_shape_and_make_image_cross(imgs)

        # Get the interpolation method.
        interp_method = INTER_MAP_OCV[interpolation]

        # Sample.
        sampled = cv2.remap( 
            img_cross, 
            self.mx, self.my, 
            interpolation=interp_method )

        # Apply gray color on invalid coordinates.
        invalid = np.logical_not(self.valid_mask)
        sampled[invalid, ...] = invalid_pixel_value

        return sampled, self.valid_mask
    
    def blend_interpolation(self, imgs, blend_func, invalid_pixel_value=127):
        '''
        This function blends the results of linear interpolation and nearest neighbor interpolation. 
        The user is supposed to provide a callable object, blend_func, which takes in img and produces
        a blending factor. The blending factor is a float number between 0 and 1. 1 means only nearest.
        '''
        
        img_cross = self.check_shape_and_make_image_cross(imgs)

        # Sample.
        sampled_linear = cv2.remap( 
            img_cross, 
            self.mx, self.my, 
            interpolation=cv2.INTER_LINEAR )
        
        sampled_nearest = cv2.remap( 
            img_cross, 
            self.mx, self.my, 
            interpolation=cv2.INTER_NEAREST )
        
        # Blend factor.
        f = blend_func(img_cross)
        
        # Sample from the blend factor.
        f = cv2.remap(
            f,
            self.mx, self.my,
            interpolation=cv2.INTER_NEAREST )
        
        sampled = f * sampled_nearest.astype(np.float32) + (1 - f) * sampled_linear.astype(np.float32)

        # Apply gray color on invalid coordinates.
        invalid = np.logical_not(self.valid_mask)
        sampled[invalid, ...] = invalid_pixel_value

        return sampled, self.valid_mask

    def compute_mean_samping_diff(self, support_shape):
        support_shape = np.array(support_shape, dtype=np.float32).reshape((2, 1))

        # Check the input shape.
        if not self.is_same_as_cached_shape( support_shape ):
            self.update_remap_coordinates( support_shape )
            self.cached_raw_shape = support_shape

        # compute_8_way_sample_msr_diff only supports torch now.
        s = torch.from_numpy(np.stack( (self.mx, self.my), axis=-1 )).unsqueeze(0).to(self.device)
        vm = torch.from_numpy(self.valid_mask).unsqueeze(0).unsqueeze(0).to(self.device)
        d = self.compute_8_way_sample_msr_diff( s, vm )
        
        return torch_2_output(d, flag_uint8=False), self.valid_mask

@register(SAMPLERS)
class SixPlanarTorch(PlanarAsBase):
    def __init__(self, fov, camera_model, R_raw_fisheye=IDENTITY_ROT, cached_raw_shape=(640, 640)):
        '''
        Arguments:
        fov (float): Full FoV of the lens in degrees.
        camera_model: A camera model for the fisheye camera.
        '''
        super().__init__(
            fov, 
            camera_model=camera_model, 
            R_raw_fisheye=R_raw_fisheye,
            cached_raw_shape=cached_raw_shape )
        
        # The 3D coordinates of the hyper-surface.
        self.xyz, self.valid_mask = self.get_xyz()
        self.valid_mask = self.valid_mask.view(self.shape)
        
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
        
        # Create the grid.
        # First dummy self.grid value then create the real one.
        self.grid = torch.zeros((1, 1, 1, 2), dtype=torch.float32, device=self.device)
        
        # Explicity set device to 'cuda' for faster computation during the construction.
        self.device = 'cuda'
        
        self.create_grid()

    @PlanarAsBase.device.setter
    def device(self, device):
        PlanarAsBase.device.fset(self, device)
        
        self.xyz = self.xyz.to(device=self.device)
        self.valid_mask = self.valid_mask.to(device=self.device)
        self.image_cross_layout_device = self.image_cross_layout_device.to(device=self.device)
        self.OFFSETS_TORCH = self.OFFSETS_TORCH.to(device=self.device)
        self.xyz_T = self.xyz_T.to(device=self.device)
        self.grid = self.grid.to(device=self.device)

    def __repr__(self):
        s = f'''SixPlanarTorch
fov = {self.fov}
shape = {self.shape}
'''
        return s

    def create_grid(self):
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
            invalid_mask = torch.logical_not(self.valid_mask).view((-1,))
            m[invalid_mask, :] = -1 # NOTE: This might be a bug.
            d = cupy.cuda.Device()
            d.synchronize()
            print(f'Time for CUDA: {time.time() - start_time}s. ')
        else:
            m, offsets = sample_coor(
                self.xyz.cpu().numpy(), 
                self.valid_mask.view((-1,)).cpu().numpy().astype(bool))

        m[:, 0] = ( m[:, 0] + offsets[:, 0]) / self.image_cross_layout_device[1] * 2 - 1
        m[:, 1] = ( m[:, 1] + offsets[:, 1]) / self.image_cross_layout_device[0] * 2 - 1
        m = m.view( ( 1, *self.shape, 2 ) )
        
        self.grid = m

    def __call__(self, imgs, interpolation='linear', invalid_pixel_value=127):
        '''
        Arguments:
        imgs (dict of arrays or list of dicts): The six images in the order of front, right, bottom, left, top, and back.
        interpolation (str): The interpolation method, could be linear or nearest.
        invalid_pixel_value (int): The value of the invalid pixel. For RGB images, it is normally 127. For depth, it is -1.
        
        Returns:
        The generated fisheye image. The image might be inside a list.
        '''

        # Make the image cross.
        img_cross, flag_uint8, single_support_shape = \
            make_image_cross_torch( imgs, device=self.device )
        N = img_cross.shape[0]
        
        if not self.is_same_as_cached_shape( single_support_shape ):
            self.cached_raw_shape = single_support_shape
        
        sampled = self.grid_sample( 
                                img_cross, 
                                self.grid.repeat((N, 1, 1, 1)), 
                                mode=INTER_MAP[interpolation],
                                padding_mode='border')

        # Apply gray color on invalid coordinates.
        invalid_mask = torch.logical_not(self.valid_mask)
        
        if flag_uint8:
            invalid_pixel_value /= 255.0
        
        sampled[..., invalid_mask] = invalid_pixel_value

        start_time = time.time()
        output_sampled = torch_2_output(sampled, flag_uint8)
        output_mask = self.valid_mask.cpu().numpy().astype(bool)
        print(f'Transfer from GPU to CPU: {time.time() - start_time}s. ')
        
        return output_sampled, output_mask

    def blend_interpolation(self, imgs, blend_func, invalid_pixel_value=127, debug_callback=dummy_debug_callback):
        '''
        This function blends the results of linear interpolation and nearest neighbor interpolation. 
        The user is supposed to provide a callable object, blend_func, which takes in img and produces
        a blending factor. The blending factor is a float number between 0 and 1. 1 means only nearest.
        '''
        
        # Make the image cross.
        img_cross, flag_uint8, single_support_shape = \
            make_image_cross_torch( imgs, device=self.device )
        N = img_cross.shape[0]
        
        if not self.is_same_as_cached_shape( single_support_shape ):
            self.cached_raw_shape = single_support_shape
        
        # Sample the images.
        grid = self.grid.repeat((N, 1, 1, 1))
        sampled_linear  = self.grid_sample( img_cross, grid, mode='bilinear', padding_mode='border' )
        sampled_nearest = self.grid_sample( img_cross, grid, mode='nearest' , padding_mode='border' )

        # The blend factor.
        bf = blend_func(img_cross)
        
        # Sample from the blend factor.
        f = self.grid_sample( bf, grid, mode='nearest', padding_mode='border' )
        
        # Debug.
        debug_callback(bf, f)
        
        # Blend.
        sampled = f * sampled_nearest + (1 - f) * sampled_linear

        # Apply gray color on invalid coordinates.
        invalid_mask = torch.logical_not(self.valid_mask)
        
        if flag_uint8:
            invalid_pixel_value /= 255.0
        
        sampled[..., invalid_mask] = invalid_pixel_value

        start_time = time.time()
        output_sampled = torch_2_output(sampled, flag_uint8)
        output_mask = self.valid_mask.cpu().numpy().astype(bool)
        print(f'Transfer from GPU to CPU: {time.time() - start_time}s. ')
        
        return output_sampled, output_mask

    def compute_mean_samping_diff(self, support_shape):
        if not self.is_same_as_cached_shape( support_shape ):
            self.cached_raw_shape = support_shape
        
        # Get to [0, 1] range.
        m = self.grid.detach().clone()
        m = ( m + 1 ) / 2
        
        # Convert back to pixel coordinates.
        m[..., 0] *= support_shape[1]
        m[..., 1] *= support_shape[0]

        d = self.compute_8_way_sample_msr_diff( m, self.valid_mask.unsqueeze(0).unsqueeze(0) )
        
        return torch_2_output(d, flag_uint8=False), self.valid_mask.cpu().numpy().astype(bool)