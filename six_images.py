
# Author: 
# Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date:
# 2022-06-17

import cv2
import math
from numba import (jit, cuda)
import numpy as np

# Local package.
from .planar_as_base import PlanarAsBase, IDENTITY_ROT, INTER_MAP
from .register import (SAMPLERS, register)

# -->       +---------+
# |  x      |0,W      |0,2W
# v         |    4    |
#  y        |   top   |
# +---------+---------+---------+---------+
# |H,0      |H,W      |H,2W     |H,3W     |H,4W
# |    3    |    0    |    1    |    5    |
# |   left  |  front  |  right  |   back  |
# +---------+---------+---------+---------+
#  2H,0     |2H,W     |2H,2W     2H,3W     2H,4W
#           |    2    |
#           |  bottom |
#           +---------+
#           3H,W     3H,2W

FRONT  = 'front'
BACK   = 'back'
LEFT   = 'left'
RIGHT  = 'right'
TOP    = 'top'
BOTTOM = 'bottom'

OFFSETS = offsets=np.array( 
            [ [1, 2, 1, 0, 1, 3],
              [1, 1, 2, 1, 0, 1] ], dtype=np.int32)

@jit(nopython=True)
def sample_coor(xyz, 
    valid_mask,
    offsets=OFFSETS):
    output = np.zeros( ( 2, xyz.shape[1] ), dtype=np.float32 )
    out_offsets = np.zeros_like(output)

    one_fourth_pi   = np.pi / 4
    half_pi         = np.pi / 2
    three_fourth_pi = one_fourth_pi + half_pi
    
    for i in range(xyz.shape[1]):
        # already rotated to the raw frame, z-forward, x-right, y-downwards.
        x = xyz[0, i]
        y = xyz[1, i]
        z = xyz[2, i]

        a_y     = math.atan2(x, y) # Angle w.r.t. y+ axis projected to the x-y plane.
        a_z     = math.atan2(z, y) # Angle w.r.t. y+ axis projected to the y-z plane.
        azimuth = math.atan2(z, x) # Angle w.r.t. x+ axis projected to the z-x plane.

        if ( -one_fourth_pi < a_y and a_y < one_fourth_pi and \
             -one_fourth_pi < a_z and a_z < one_fourth_pi ):
            # Bottom.
            output[0, i] = min( max( ( 1 + x/y ) / 2, 0 ), 1 )
            output[1, i] = min( max( ( 1 - z/y ) / 2, 0 ), 1 )
            out_offsets[0, i] = offsets[0][2]
            out_offsets[1, i] = offsets[1][2]
        elif ( (three_fourth_pi < a_y or a_y < -three_fourth_pi) and \
               (three_fourth_pi < a_z or a_z < -three_fourth_pi) ):
            # Top.
            output[0, i] = min( max( ( 1 - x/y ) / 2, 0 ), 1 )
            output[1, i] = min( max( ( 1 - z/y ) / 2, 0 ), 1 )
            out_offsets[0, i] = offsets[0][4]
            out_offsets[1, i] = offsets[1][4]
        elif ( one_fourth_pi <= azimuth and azimuth < three_fourth_pi ):
            # Front.
            output[0, i] = min( max( ( 1 + x/z ) / 2, 0 ), 1 )
            output[1, i] = min( max( ( 1 + y/z ) / 2, 0 ), 1 )
            out_offsets[0, i] = offsets[0][0]
            out_offsets[1, i] = offsets[1][0]
        elif ( -one_fourth_pi <= azimuth and azimuth < one_fourth_pi ):
            # Right.
            output[0, i] = min( max( ( 1 - z/x ) / 2, 0 ), 1 )
            output[1, i] = min( max( ( 1 + y/x ) / 2, 0 ), 1 )
            out_offsets[0, i] = offsets[0][1]
            out_offsets[1, i] = offsets[1][1]
        elif ( -three_fourth_pi <= azimuth and azimuth < -one_fourth_pi ):
            # Back.
            output[0, i] = min( max( ( 1 + x/z ) / 2, 0 ), 1 )
            output[1, i] = min( max( ( 1 - y/z ) / 2, 0 ), 1 )
            out_offsets[0, i] = offsets[0][5]
            out_offsets[1, i] = offsets[1][5]
        elif ( three_fourth_pi <= azimuth or azimuth < -three_fourth_pi ):
            # Left.
            output[0, i] = min( max( ( 1 - z/x ) / 2, 0 ), 1 )
            output[1, i] = min( max( ( 1 - y/x ) / 2, 0 ), 1 )
            out_offsets[0, i] = offsets[0][3]
            out_offsets[1, i] = offsets[1][3]
        else:
            raise Exception('xy invalid.')
            # raise Exception(f'x = {x}, y = {y}, z = {z}, a_y = {a_y}, a_z = {a_z}, one_fourth_pi = {one_fourth_pi}, half_pi = {half_pi}')

    output[:, np.logical_not(valid_mask)] = -1

    return output, out_offsets

@cuda.jit()
def k_sample_coor(
    output, out_offsets, xyz, offsets):
    # Prepare the index.
    x_idx    = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    x_stride = cuda.blockDim.x * cuda.gridDim.x

    # Constants.
    one_fourth_pi   = np.pi / 4
    half_pi         = np.pi / 2
    three_fourth_pi = one_fourth_pi + half_pi

    # Dimensionless image size.
    dls = 2 # 90 degrees of FOV.
    # dls = 1.9969 # 89.912 degrees of FOV.
    dls_half = dls / 2

    # Loop.
    for i in range( x_idx, xyz.shape[1], x_stride ):
        # already rotated to the raw frame, z-forward, x-right, y-downwards.
        x = xyz[0, i]
        y = xyz[1, i]
        z = xyz[2, i]

        a_y     = math.atan2(x, y) # Angle w.r.t. y+ axis projected to the x-y plane.
        a_z     = math.atan2(z, y) # Angle w.r.t. y+ axis projected to the y-z plane.
        azimuth = math.atan2(z, x) # Angle w.r.t. x+ axis projected to the z-x plane.

        if ( -one_fourth_pi < a_y and a_y < one_fourth_pi and \
             -one_fourth_pi < a_z and a_z < one_fourth_pi ):
            # Bottom.
            output[0, i] = min( max( ( dls_half + x/y ) / dls, 0 ), 1 )
            output[1, i] = min( max( ( dls_half - z/y ) / dls, 0 ), 1 )
            out_offsets[0, i] = offsets[0][2]
            out_offsets[1, i] = offsets[1][2]
        elif ( (three_fourth_pi < a_y or a_y < -three_fourth_pi) and \
               (three_fourth_pi < a_z or a_z < -three_fourth_pi) ):
            # Top.
            output[0, i] = min( max( ( dls_half - x/y ) / dls, 0 ), 1 )
            output[1, i] = min( max( ( dls_half - z/y ) / dls, 0 ), 1 )
            out_offsets[0, i] = offsets[0][4]
            out_offsets[1, i] = offsets[1][4]
        elif ( one_fourth_pi <= azimuth and azimuth < three_fourth_pi ):
            # Front.
            output[0, i] = min( max( ( dls_half + x/z ) / dls, 0 ), 1 )
            output[1, i] = min( max( ( dls_half + y/z ) / dls, 0 ), 1 )
            out_offsets[0, i] = offsets[0][0]
            out_offsets[1, i] = offsets[1][0]
        elif ( -one_fourth_pi <= azimuth and azimuth < one_fourth_pi ):
            # Right.
            output[0, i] = min( max( ( dls_half - z/x ) / dls, 0 ), 1 )
            output[1, i] = min( max( ( dls_half + y/x ) / dls, 0 ), 1 )
            out_offsets[0, i] = offsets[0][1]
            out_offsets[1, i] = offsets[1][1]
        elif ( -three_fourth_pi <= azimuth and azimuth < -one_fourth_pi ):
            # Back.
            output[0, i] = min( max( ( dls_half + x/z ) / dls, 0 ), 1 )
            output[1, i] = min( max( ( dls_half - y/z ) / dls, 0 ), 1 )
            out_offsets[0, i] = offsets[0][5]
            out_offsets[1, i] = offsets[1][5]
        elif ( three_fourth_pi <= azimuth or azimuth < -three_fourth_pi ):
            # Left.
            output[0, i] = min( max( ( dls_half - z/x ) / dls, 0 ), 1 )
            output[1, i] = min( max( ( dls_half - y/x ) / dls, 0 ), 1 )
            out_offsets[0, i] = offsets[0][3]
            out_offsets[1, i] = offsets[1][3]

def sample_coor_cuda( 
    xyz, 
    valid_mask,
    offsets=OFFSETS):

    output      = np.zeros((2, xyz.shape[1]), dtype=xyz.dtype)
    out_offsets = np.zeros_like(output)
    
    # Prepare the memory.
    d_xyz         = cuda.to_device(xyz)
    d_output      = cuda.to_device(output)
    d_out_offsets = cuda.to_device(out_offsets)
    d_offsets     = cuda.to_device(offsets)

    cuda.synchronize()
    k_sample_coor[[1024,1,1],[256,1,1]]( d_output, d_out_offsets, d_xyz, d_offsets )
    cuda.synchronize()

    output = d_output.copy_to_host()
    out_offsets = d_out_offsets.copy_to_host()
    # print(f'output.dtype = {output.dtype}')
    
    invalid_mask = np.logical_not(valid_mask)
    output[:, invalid_mask] = -1

    return output, out_offsets

@register(SAMPLERS)
class SixPlanarAsBase(PlanarAsBase):
    def __init__(self, fov, camera_model, R_raw_fisheye=IDENTITY_ROT):
        '''
        Arguments:
        fov (float): Full FoV of the lens in degrees.
        camera_model: A camera model for the fisheye camera.
        '''
        super(SixPlanarAsBase, self).__init__(fov, camera_model=camera_model, R_raw_fisheye=R_raw_fisheye)

    def make_image_cross(self, imgs):
        '''
        Arguments:
        imgs (dict of arrays): The six images with the keys as front, back, left, right, top, and bottom.

        Returns:
        A image cross with shape (3*H, 4*W).
        '''
        
        global FRONT, BACK, LEFT, RIGHT, TOP, BOTTOM
        
        H, W = imgs[FRONT].shape[:2]
        d_type = imgs[FRONT].dtype

        if ( imgs[FRONT].ndim == 3 ):
            # Get the last dimension of the input image.
            last_dim = imgs[FRONT].shape[2]
            canvas = np.zeros( ( 3*H, 4*W, last_dim ), dtype=d_type )
        elif ( imgs[FRONT].ndim == 2 ):
            canvas = np.zeros( ( 3*H, 4*W ), dtype=d_type )
        else:
            raise Exception(f'Wrong dimension of the input images. imgs[FRONT].shape = {imgs[FRONT].shape}')

        canvas[  H:2*H,   W:2*W, ...] = imgs[FRONT]  # Front.
        canvas[  H:2*H, 2*W:3*W, ...] = imgs[RIGHT]  # Right.
        canvas[2*H:3*H,   W:2*W, ...] = imgs[BOTTOM] # Bottom.
        canvas[  H:2*H,   0:W,   ...] = imgs[LEFT]   # Left.
        canvas[  0:H,     W:2*W, ...] = imgs[TOP]    # Top.
        canvas[  H:2*H, 3*W:4*W, ...] = imgs[BACK]   # Top.

        # Padding.
        # Right.
        canvas[ H-1, 2*W:3*W, ... ] = imgs[TOP][    ::-1, -1, ... ]
        canvas[ 2*H, 2*W:3*W, ... ] = imgs[BOTTOM][    :, -1, ... ]
        # Bottom.
        canvas[ 2*H:3*H, 2*W, ... ] = imgs[RIGHT][ -1,    :, ... ]
        canvas[ 2*H:3*H, W-1, ... ] = imgs[LEFT][  -1, ::-1, ... ]
        # Left.
        canvas[ H-1, 0:W, ... ] = imgs[TOP][       :, 0, ...]
        canvas[ 2*H, 0:W, ... ] = imgs[BOTTOM][ ::-1, 0, ...]
        # Top.
        canvas[ 0:H, W-1, ... ] = imgs[LEFT][  0,    :, ...]
        canvas[ 0:H, 2*W, ... ] = imgs[RIGHT][ 0, ::-1, ...]
        # Back.
        canvas[ H-1, 3*W:4*W, ... ] = imgs[TOP][     0, ::-1, ... ]
        canvas[ 2*H, 3*W:4*W, ... ] = imgs[BOTTOM][ -1, ::-1, ... ]

        return canvas

    def __repr__(self):
        s = f'''fov = {self.fov}
shape = {self.shape}
a = {self.a}
s = 
{self.s}
invs = 
{self.invS}
p = {self.p}
flagCuda = {self.flag_cuda}
'''
        return s

    def __call__(self, imgs, interpolation='linear', invalid_pixel_value=127):
        '''
        Arguments:
        imgs (list of arrays): The five images in the order of front, right, bottom, left, and top.
        interpolation (str): The interpolation method, could be linear or nearest.
        invalid_pixel_value (int): The value of the invalid pixel. For RGB images, it is normally 127. For depth, it is -1.
        
        Returns:
        The generated fisheye image.
        '''

        global FRONT, RIGHT, BOTTOM, LEFT, TOP, BACK

        # Make the image cross.
        img_cross = self.make_image_cross( imgs )
        # cv2.imwrite('img_cross.png', img_cross)

        # Get the original shape of the input images.
        img_shape = np.array(imgs[FRONT].shape[:2], dtype=np.float32).reshape((2, 1))

        # The 3D coordinates of the hyper-surface.
        xyz, valid_mask = self.get_xyz()

        # Get the sample locations.
        if ( self.flag_cuda ):
            m, offsets = sample_coor_cuda( xyz, valid_mask )
        else:
            m, offsets = sample_coor(xyz, valid_mask)

        m = m * ( img_shape - 1 ) + offsets * img_shape

        mx = m[0, :].reshape(self.shape)
        my = m[1, :].reshape(self.shape)

        # Get the interpolation method.
        interp_method = INTER_MAP[interpolation]

        # Sample.
        sampled = cv2.remap( 
            img_cross, 
            mx, my, 
            interpolation=interp_method )

        # Apply gray color on invalid coordinates.
        invalid = np.logical_not(valid_mask).reshape(self.shape)
        sampled[invalid, ...] = invalid_pixel_value

        return sampled, valid_mask

    def compute_mean_samping_diff(self, img_shape):
        img_shape = np.array(img_shape, dtype=np.float32).reshape((2, 1))

        # The 3D coordinates of the hyper-surface.
        xyz, valid_mask = self.get_xyz()

        # Get the sample locations.
        if ( self.flag_cuda ):
            m, offsets = sample_coor_cuda( xyz, valid_mask )
        else:
            m, offsets = sample_coor(xyz, valid_mask)

        m = m * ( img_shape - 1 ) + offsets * img_shape

        mx = m[0, :].reshape(self.shape)
        my = m[1, :].reshape(self.shape)
        valid_mask = valid_mask.reshape(self.shape)

        d = self.compute_8_way_sample_msr_diff( np.stack( (mx, my), axis=-1 ), valid_mask )
        return d, valid_mask
    