'''
Author: Yorai Shaoul (yorai@cmu.edu)
A class that resamples pinhole images from a calibrated base fisheye image.
This class optimizes for sampling multipe pinhole images from a single fisheye image.
'''

import copy
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Torch imports.
from torch.utils.data import DataLoader
import torch

# Image resampling imports.
import sys
sys.path.append('..')
from image_resampling.mvs_utils.camera_models import DoubleSphere, ShapeStruct
from image_resampling.mvs_utils.ftensor import FTensor, f_eye
from .ocv_torch import ( ocv_2_torch, torch_2_ocv )

# Constants.
IDENTITY_ROT = f_eye(3, f0='raw', f1='fisheye', rotation=True, dtype=torch.float32)

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


class FishAsBase():
    """
    A class for resampling pinhole images from fisheye images.
    """
    
    def __init__(self, fish_camera_model, pinx_camera_model, Rs_pin_in_fish=torch.tensor(np.array([np.eye(3)])), cached_pin_shape=(300, 300)):
        """_summary_

        Args:
            fish_camera_model (Fisheye): The camera model of the origin fisheye camera.
            pin_camera_model (Pinhole): The camera model of all the resampled pinhole cameras
            Rs_pin_in_fish (torch.tensor, optional): Batch of rotation matrices. Pinholes in the fisheye frame (x-right, y-down, z-out). Defaults to IDENTITY_ROT. (B,3,3)
            cached_pin_shape (tuple, optional): _description_. Defaults to (300, 300). TODO(yoraish): the cache should include the rotation matrices too?
        """
        print("Creating a new FishAsBase object.")
        # Number of pinholes.
        self.B = Rs_pin_in_fish.shape[0]

        # Note: in the following comments and descriptions, B is the batch size.
        # The device on which computations will be carried out.
        self._device = 'cpu'

        # Camera models.
        self.pinx_camera_model = pinx_camera_model
        self.fish_camera_model = fish_camera_model
        self.pin_shape = self.pinx_camera_model.ss.shape

        # Grid. The instruction for torch.grid_sample. 
        # Tensor of shape (B, H, W, 2). H,W are the pinhole image shape. Each u,v in H,W will be filled with the value of the fisheye at the pixel specified by the last dimension of the input (size 2).
        self.grid = torch.zeros((self.B, self.pin_shape[0], self.pin_shape[1], 2), dtype=torch.float32, device=self.device)

        # Rotations of pinholes (in the fish frame). They are all batched in a torch tensor.
        self.Rs_pin_in_fish = Rs_pin_in_fish.to(dtype=torch.float32)

        # The 3D coordinates of the pinhole pixels. Dummy value now.
        # Shape (B, 3, H*W).
        self.xyz = torch.zeros((self.B, 3,1), dtype=torch.float32, device=self.device)
        self.xyz_T = torch.zeros((self.B, 1,3), dtype=torch.float32, device=self.device)

        # A data stucture relating (by index) the xyz points to the (u,v) pixels those came from.
        # Shape is (B, 2, H*W).
        self.uv_of_xyz_ix = torch.zeros((self.B, 2, 1), dtype=torch.float32, device=self.device)

        # Valid mask on the pinhole images. 
        # Shape (B, H*W).
        self.valid_mask = torch.ones((self.B, self.pinx_camera_model.ss.shape[0], self.pinx_camera_model.ss.shape[1]))

        # The 3D coordinates of the hyper-surfaces. Each pinhole image is a 'batch' in this tensor, and it has a collection of 3d points associated with it.
        pins_xyz, pins_valid_mask = self.get_xyz(back_shift_pixel=True) # TODO(yorai): Backshift??
        self.xyz = pins_xyz
        self.valid_mask = pins_valid_mask.view((self.B , *self.pin_shape)) # May need to be cast to bool. 
        
        # Make a copy of xyz. Make it become Nx3.
        self.xyz_T = self.xyz.permute((0, 2,1)).contiguous()
        

    @property
    def align_corners(self):
        return False

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self.xyz = self.xyz.to(device=self.device)
        self.valid_mask = self.valid_mask.to(device=self.device)
        self.xyz_T = self.xyz_T.to(device=self.device)
        self.grid = self.grid.to(device=self.device)

        self._device = device
        self.pinx_camera_model.device = device
        self.fish_camera_model.device = device
        self.Rs_pin_in_fish = self.Rs_pin_in_fish.to(device=device)


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
        '''
        From fisheye_resampler. This could be helpful for meshgrid_pixels, and also for get_xyz, and also for create_grid.
        Grid - the one used for grid_sample, should be renamed to something more descript,
        
        pin_in_fish = self.tm.get_transform(pinx, "fish") # Child, parent.

        # Unpack the intrinsics.
        h, w = data["intrinsics"].height, data["intrinsics"].width
        fx, fy = data["intrinsics"].fx, data["intrinsics"].fy
        cx, cy = data["intrinsics"].cx, data["intrinsics"].cy
        # Generate 3D points.
        x = np.arange(w)
        y = np.arange(h)

        xy_grid = np.vstack(np.meshgrid(x, y, indexing="xy"))
        xy = xy_grid.reshape((2,-1))

        # xy1 = np.vstack((x[np.newaxis, :], y[np.newaxis, :], np.ones((1,x.shape[0]))))
        xy1 = np.vstack((xy, np.ones(xy.shape[-1])))

        # Convert to camera-frame (metric).
        inv_intrinsics = np.array([[fy,  0, -cx * fy],
                                [ 0, fx, -cy * fx],
                                [ 0,  0,  fx * fy] ]) /(fx * fy)
        XYZ = inv_intrinsics.dot(xy1)

        # Translate and rotate these inverse projected points according to the transformation pinx_in_fish.
        # Physically, since the inverse projected points are scale ambiguous, we can only get real reprojectoion into the fisheye camera upon only rotations.
        XYZ1 = np.vstack((XYZ,np.ones(XYZ.shape[-1]))) # TODO(yoraish): vstack?
        XYZ1 = pin_in_fish.dot(XYZ1)
        XYZ = XYZ1[:3, :]
        XYZ_grid = XYZ.T.reshape((h, w, 3))

        # Project on image plane. (https://github.com/matsuren/dscamera/blob/master/dscamera/camera.py)
        img_pts, valid_mask = self.world_to_cam(XYZ_grid)
        # out = self._warp_img(img_fish, img_pts, valid_mask)

        img_pts = img_pts.astype(np.float32)
        # Out is the pinhole camera image size. 
        # Each pixel in the pinhole is remapped from fisheye img, via img_pts (the points on the fisheye that will go to the pinhole)
        out = cv2.remap(
        self.fish_img, img_pts[..., 0], img_pts[..., 1], cv2.INTER_LINEAR
        )
        out[~valid_mask] = 0.0
        pin_nums_to_imgs[pinx] = out.copy()

        return pin_nums_to_imgs
        '''
        

    def get_xyz(self, back_shift_pixel=False):
        '''
        Compute the ray vectors for all valid pixels in the pinhole image.
        A ray vector is represented as a unit vector.
        All ray vectors will be transformed such that their coordiantes are
        measured in the raw frame where z-forward, x-right, and y-downward.
        
        Some pixels are not going to have valid rays. There is a mask of valid
        pixels that is also returned by this function.
        '''
        # The pixel coordinates of a single pinhole image. For each pinhole, this pixel grid will be duplicated and rotated appropriately.
        xx, yy = self.mesh_grid_pixels(self.pin_shape, flag_flatten=True) # 1D.
        
        if back_shift_pixel:
            xx -= 0.5
            yy -= 0.5
        
        pixel_coor = torch.vstack( (xx, yy) ) # 2xN=H*W
        pixel_coor_uv = torch.vstack( (yy, xx) ) # 2xN=H*W
        # Remember the relationship between the xyz indices and the pixels those originated from.
        self.uv_of_xyz_ix = pixel_coor_uv.repeat((self.B, 1 , 1))
        
        xyz, valid_mask = \
            self.pinx_camera_model.pixel_2_ray(pixel_coor) # Since this is a pinhole, all pixels that were projected out from the grid are expected to be valid. Assuming that the grid was the size of the size of the pinhole and that there were no evil bugs in the code.
        
        # Change of reference frame. This also creates points for each camera. The output is (B, 3, N=H*W)
        pins_xyz = self.Rs_pin_in_fish @ xyz

        return pins_xyz, valid_mask.repeat((self.B, 1, 1)) 
        

    def create_grid(self):
        # Get the sample locations.
        # This method produces the object self.grid, which is a size (B, H, W, 2), mapping u,v in H,W of a pinhole camera to the sampling pixels in the fisheye.
        self.grid = torch.zeros((self.B, self.pin_shape[0], self.pin_shape[1], 2), dtype=torch.float32, device=self.device)

        # Project 3D points that originated from the pinholes onto the fisheye.
        # Fish uv is of shape (B, 2, N=H*W)
        fish_uv, fish_uv_mask = self.fish_camera_model.point_3d_2_pixel(self.xyz)
        '''
        if False:
            fish_uv = torch.permute(fish_uv, (0,2,1)).cpu().numpy().astype(int)
            print(fish_uv.shape)
            for b in range(self.B):
                fish_uvb = fish_uv[b]
                for (u,v), m in zip(fish_uvb, fish_uv_mask[b]):
                    print(u,v, m)
                    try:
                        imgs[u,v, :] = [255,0,0]
                        if m == False:
                            imgs[u,v, :] = [255,255,255]
                    except:
                        continue       
            cv2.imshow('im', imgs)
            cv2.waitKey(0)
        '''
        
        # So now we have three tensors, all in the same order,
        # (B, 2, H*W) where each row in a batch is a (u,v) pixel in the pinhole (origin of 3D point).
        # (B, 3, H*W) where each row in a batch is a (x,y,z) 3D point from the pinhole pixel.
        # (B, 2, H*W) where each row in a batch is a (u,v) pixel in the fisheye, corresponding to the pinhole pixel.
        # TODO(yoraish): deal with torch indices soon.
        print(self.uv_of_xyz_ix.shape)
        for b in range(self.B):
            for ix in range(self.uv_of_xyz_ix[b].shape[-1]):
                u_pin, v_pin = self.uv_of_xyz_ix[b, :, ix].to(dtype=torch.int32)
                
                # Normalize the points on the fisheye to range [-1., 1.] expected by the torch grid sample function.
                fish_h, fish_w = self.fish_camera_model.ss.shape
                fish_center = torch.tensor([fish_h/2, fish_w/2])
                self.grid[b, u_pin, v_pin] = ((fish_uv[b, :, ix] - fish_center)/fish_center)


        # if ( self.device != 'cpu' ):
        #     start_time = time.time()
        #     # Allocate m and offsets.
        #     m = torch.zeros( (self.xyz_T.shape[0], 2), dtype=torch.float32, device=self.device )
        #     offsets = m.detach().clone()
            
        #     # Call the CUDA function.
        #     self.sample_coor_cuda(
        #         block=(self.cuda_block_size, ),
        #         grid=(self.cuda_grid_size, ),
        #         args=(
        #             cupy.int32(self.xyz_T.shape[0]),
        #             self.xyz_T.data_ptr(),
        #             self.OFFSETS_TORCH.data_ptr(),
        #             m.data_ptr(),
        #             offsets.data_ptr()
        #         )
        #     )
            
        #     # Handle the valid mask.
        #     invalid_mask = torch.logical_not(self.valid_mask).view((-1,))
        #     m[invalid_mask, :] = -1 # NOTE: This might be a bug.
        #     d = cupy.cuda.Device()
        #     d.synchronize()
        #     print(f'Time for CUDA: {time.time() - start_time}s. ')
        # else:
        #     m, offsets = sample_coor(
        #         self.xyz.cpu().numpy(), 
        #         self.valid_mask.view((-1,)).cpu().numpy().astype(bool))

        # m[:, 0] = ( m[:, 0] + offsets[:, 0]) / self.image_cross_layout_device[1] * 2 - 1
        # m[:, 1] = ( m[:, 1] + offsets[:, 1]) / self.image_cross_layout_device[0] * 2 - 1
        # m = m.view( ( 1, *self.shape, 2 ) )
        
        # self.grid = m


    def __call__(self, img, interpolation='linear', invalid_pixel_value=127):
        global INTER_MAP
        # Check if the size of the pinholes has changed between now and the cache.
        # Check if the rotation matrices tensor has changed between now and the cache.
        # TODO(yoraish).
        img_np = img.copy()
        img = torch.tensor(img, dtype=torch.int32, device=self.device)

        # If change, then recompute the self.grid object.
        self.create_grid()

        # With the grid we had, or the new one that we just created if we needed, 
        self.grid = self.grid.float()
        grid  = self.grid[0:1,...]#.permute(((1,2,0,3)))#.repeat((1,1,3, 1))

        img = img.to('cuda').float()
        img = img.unsqueeze(0)
        img = img.permute((0,3,1,2))

        print("input", img.shape, "grid", grid.shape)
        sampled = F.grid_sample( 
                                img, 
                                grid, 
                                mode=INTER_MAP[interpolation], 
                                align_corners=self.align_corners )

        for vn,un in zip(self.grid[:, :, :, 0].flatten(), self.grid[:, :, :, 1].flatten()):
            # U and V are flipped since the grid operates on x,y.
            u = un*self.fish_camera_model.ss.shape[0]/2 + self.fish_camera_model.ss.shape[0]/2
            v = vn*self.fish_camera_model.ss.shape[1]/2 + self.fish_camera_model.ss.shape[1]/2
            u = int(u.item())
            v = int(v.item())
            try:
                img_np[u,v] = np.array([255,0,0])
            except:
                continue
        print(sampled)
        plt.imshow(img_np)
        plt.title("in call img")
        plt.show()

        plt.imshow(sampled[0].cpu().permute((1,2,0))/255)
        plt.title("in call sampled")
        plt.show()