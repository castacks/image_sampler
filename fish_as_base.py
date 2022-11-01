'''
Author: Yorai Shaoul (yorai@cmu.edu)
A class that resamples pinhole images from a calibrated base fisheye image.
This class optimizes for sampling multipe pinhole images from a single fisheye image. The pinholes share the same intrinsics values, including image shapes. For now.
'''

import copy
import os
import time
import cv2
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Torch imports.
from torch.utils.data import DataLoader
import torch

# Image resampling imports.
import sys

import tqdm
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
        # Shape (B, H_pin, W_pin).
        self.valid_mask = torch.ones((self.B, *self.pin_shape))

        # The 3D coordinates of the hyper-surfaces. Each pinhole image is a 'batch' in this tensor, and it has a collection of 3d points associated with it.
        # The valid mask starts out as all-valid since, for now, all we know is that points projected from the pinholes out are all valid. When the grid will b e computed for the first time, and some points projected out of the pinholes cannot be projcted back to the fisheye (out of FoV), those will be marked as invalid as well.
        pins_xyz, pins_valid_mask = self.get_xyz(back_shift_pixel=True) # TODO(yorai): Backshift??
        self.xyz = pins_xyz
        self.valid_mask = pins_valid_mask.view((self.B , *self.pin_shape))  
        
        # Make a copy of xyz. Make it become Nx3.
        self.xyz_T = self.xyz.permute((0, 2,1)).contiguous()

        # Cache.
        self.cached_pin_shape = None
        self.cached_Rs_pin_in_fish = None
        self.resample_counter = 0
        

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
        
        # So now we have three (+1 mask) tensors, all in the same order,
        # uv_of_xyz_ix (B, 2, H*W) where each column in a batch is a (u,v) pixel in the pinhole (origin of 3D point).
        # self.xyz     (B, 3, H*W) where each column in a batch is a (x,y,z) 3D point from the pinhole pixel.
        # fish_uv      (B, 2, H*W) where each column in a batch is a (u,v) pixel in the fisheye, corresponding to the pinhole pixel.
        # fish_uv_mask (B, H*W)    where each entry in a batch at an index of [pin pixel, xyz, fish pixel], is its validity.

        # Create the valid mask. For each index, find its pinhole pixel location, and assign its validity.
        # Do not redeclare the valid mask here, as it may already have some invalid pixels. If you would want to, however, you'd use: self.valid_mask = torch.ones((self.B, *self.pin_shape)).

        print("Starting mask.")
        startmask = time.time()
        ixs = torch.arange(self.uv_of_xyz_ix.shape[-1])# .repeat((self.B, 1))
        bs = torch.arange(self.B)
        self.valid_mask = self.valid_mask.to(dtype=torch.int64, device='cpu')
        fish_uv_mask.to('cpu')

        # Populate the valid mask. 
        # NOTE(yoraish): this section is VERY bug-prone. A tested iterative implementation is commented-out below.
        # NOTE(yoraish): this could be sped up more by removing the for loop.
        for b in range(self.B):
            fish_uv_mask_slice = fish_uv_mask[b].to(dtype=torch.int64)
            self.valid_mask[b, self.uv_of_xyz_ix[b, 0, :].to(dtype=torch.int64), self.
            uv_of_xyz_ix[b, 1, :].to(dtype=torch.int64)] = fish_uv_mask_slice
        self.valid_mask = self.valid_mask.to(self.device)
        print("Mask took", time.time() - startmask)
        

        print("Starting create grid.")
        startgrid = time.time()
        fish_uv =fish_uv.to(device = self.device)
        # Normalize the points on the fisheye to range [-1., 1.] expected by the torch grid sample function.
        fish_h, fish_w = self.fish_camera_model.ss.shape
        fish_center = torch.tensor([fish_h/2, fish_w/2]).to(self.device)

        for b in range(self.B):
            print("Trying to divide shapes",  fish_uv[b, :, :].shape, fish_center.shape)
                
            self.grid[b, self.uv_of_xyz_ix[b, 0, :].to(dtype=torch.int64), self.
            uv_of_xyz_ix[b, 1, :].to(dtype=torch.int64), :] = ((fish_uv[b, :, :].T - fish_center)/fish_center)

        print("Grid took ", time.time() - startgrid)


        # Save the rotation matrices and size of pinholes to cache. This parameterizes the grid mapping. If either of these change, then there is a need to recompute the self.grid object.
        self.cached_Rs_pin_in_fish = self.Rs_pin_in_fish.clone()
        self.cached_pin_shape = self.pin_shape

        '''
        Iterative implementation for mask and grid creation.

        print("Starting mask.")
        startmask = time.time()
        good_mask = torch.ones(self.valid_mask.shape)
        for b in range(self.B):
            for ix in range(self.uv_of_xyz_ix[b].shape[-1]):
                u_pin, v_pin = self.uv_of_xyz_ix[b, :, ix].to(dtype=torch.int64)
                good_mask[b, u_pin, v_pin] = fish_uv_mask[b, ix]
        good_mask = good_mask.to(self.device)
        print("Mask took", time.time() - startmask)

        print("Starting create grid.")
        startgrid = time.time()
        good_grid = torch.zeros(self.grid.shape)
        for b in range(self.B):
            for ix in range(self.uv_of_xyz_ix[b].shape[-1]):
                u_pin, v_pin = self.uv_of_xyz_ix[b, :, ix].to(dtype=torch.int32)
                
                # Normalize the points on the fisheye to range [-1., 1.] expected by the torch grid sample function.
                fish_h, fish_w = self.fish_camera_model.ss.shape
                fish_center = torch.tensor([fish_h/2, fish_w/2])
                good_grid[b, u_pin, v_pin] = ((fish_uv[b, :, ix] - fish_center)/fish_center)

        print("Grid took ", time.time() - startgrid)
    
        '''


    def __call__(self, img, interpolation='linear', invalid_pixel_value= 0 ):
        global INTER_MAP
        # Check if the size of the pinholes has changed between now and the cache.
        # Check if the rotation matrices tensor has changed between now and the cache.
        if not self.is_same_as_cached_Rs_shape(self.Rs_pin_in_fish, self.pin_shape):
            # If change, then recompute the self.grid object.
            self.create_grid()

        # Convert the image to a torch tensor. Is this fast?
        img = torch.tensor(img, dtype=torch.int32, device=self.device)

        # With the grid we had, or the new one that we just created if we needed, 
        self.grid = self.grid.float()

        img = img.to('cuda').float()
        img = img.unsqueeze(0)
        img = img.permute((0,3,1,2))
        img = img.repeat((self.B,1,1,1))

        grid  = self.grid

        print("input", img.shape, "grid", grid.shape)
        startt = time.time()

        # Resample the pinholes from the fisheye. The output is of shape (B, C, H_pin, W_pin)
        sampled = F.grid_sample( 
                                img, 
                                grid, 
                                mode=INTER_MAP[interpolation], 
                                align_corners=self.align_corners )
        print("sample time", time.time() -startt)

        # Apply the valid mask. This marks pinhole pixels that were outside the fisheye FoV as invalid.
        repeated_valid_mask = self.valid_mask.unsqueeze(1).repeat((1,3,1,1))
        sampled[repeated_valid_mask == 0] = invalid_pixel_value
        return sampled, self.valid_mask


    def visualize_resampling(self, img, save_dir = "", interpolation='linear', invalid_pixel_value=0):
        sampled, valid_mask = self.__call__(img, interpolation, invalid_pixel_value)
        # fig, axes = plt.subplots(nrows = int(self.B**0.5)+1, ncols = int(self.B**0.5)+1)
        fig, axes = plt.subplots(nrows = 3, ncols = 3)
        axes = axes.flatten() 
        for b in range(self.B):
            ax = axes[b]
            ax.imshow(sampled[b].cpu().permute((1,2,0))/255)
            # ax.imshow(self.valid_mask[b].cpu())
            ax.set_title(f"pin{b}")
        
        if save_dir:
            fig.savefig(os.path.join(save_dir, f"{self.resample_counter}.png"))
        self.resample_counter += 1
        plt.show()

    def visualize_grid(self, img):
        """Visualize the pixels on the fisheye img that are resampled into pinhole images.
            The participating pixels are extracted from the self.grid object.
        Args:
            img (np.array (C,H,W)): The input fisheye image to be overlaid with the pinholes and shown.
        """
        img = img.copy()
        # Visualize the pinholes on the fish.
        print("Visualizing pinhole-resampled areas on top of the fisheye image.")
        for vn,un in tqdm.tqdm(zip(self.grid[:, :, :, 0].flatten(), self.grid[:, :, :, 1].flatten()), total = len(self.grid[:, :, :, 0].flatten())):
            # U and V are flipped since the grid operates on x,y.
            u = un*self.fish_camera_model.ss.shape[0]/2 + self.fish_camera_model.ss.shape[0]/2
            v = vn*self.fish_camera_model.ss.shape[1]/2 + self.fish_camera_model.ss.shape[1]/2
            u = int(u.item())
            v = int(v.item())
            try:
                img[u,v] = np.array([255,0,0])
            except:
                print(f"Could not mark pixel uv {u,v} in visualization function.")
                continue

        plt.imshow(img)
        plt.title("Marked fisheye image.")
        plt.show()

    def is_same_as_cached_Rs_shape(self, Rs, pin_shape):
        if type(self.cached_pin_shape) == type(None) or type(self.cached_Rs_pin_in_fish) == type(None):
            return False

        if tuple(self.cached_pin_shape) == tuple(pin_shape):
            if  torch.all(self.cached_Rs_pin_in_fish == Rs) == True:
                return True