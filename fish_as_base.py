'''
Author: Yorai Shaoul (yorai@cmu.edu)
A class that resamples pinhole images from a calibrated base fisheye image.
This class optimizes for sampling multipe pinhole images from a single fisheye image. The pinholes share the same intrinsics values, including image shapes. For now.
'''

import os
import time
from colorama import Fore, Style
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
import tqdm

from image_resampling.mvs_utils.camera_models import DoubleSphere, ShapeStruct
from image_resampling.mvs_utils.ftensor import FTensor, f_eye
from .ocv_torch import ( ocv_2_torch, torch_2_ocv )

# Constants.
IDENTITY_ROT = f_eye(3, f0='raw', f1='fisheye', rotation=True, dtype=torch.float32)

INTER_MAP = {
    'nearest': 'nearest',
    'linear': 'bilinear',
}


class FishAsBase():
    """
    A class for resampling pinhole images from fisheye images.
    """
    
    def __init__(self, fish_camera_model, pinx_camera_model, Rs_pin_in_fish=torch.tensor(np.array([np.eye(3)]))):
        """
        Args:
            fish_camera_model (Fisheye): The camera model of the origin fisheye camera.
            pin_camera_model (Pinhole): The camera model of all the resampled pinhole cameras
            Rs_pin_in_fish (torch.tensor, optional): Batch of rotation matrices. Pinholes in the fisheye frame (x-right, y-down, z-out). Defaults to the identity rotation. (B,3,3)
        """
        print("Creating a new FishAsBase object.")
        # Number of pinholes.
        self.P = Rs_pin_in_fish.shape[0]

        # Batch size.
        self.B = -1

        # Note: in the following comments and descriptions, B is the batch size.
        # The device on which computations will be carried out.
        self._device = 'cpu'

        # Camera models.
        self.pinx_camera_model = pinx_camera_model
        self.fish_camera_model = fish_camera_model
        self.pin_shape = self.pinx_camera_model.ss.shape

        # Grid. The instruction for torch.grid_sample. 
        # Tensor of shape (B, H, W, 2). H,W are the pinhole image shape. Each u,v in H,W will be filled with the value of the fisheye at the pixel specified by the last dimension of the input (size 2).
        self.grid = torch.zeros((self.P, self.pin_shape[0], self.pin_shape[1], 2), dtype=torch.float32, device=self.device)

        # Rotations of pinholes (in the fish frame). They are all batched in a torch tensor.
        self.Rs_pin_in_fish = Rs_pin_in_fish.to(dtype=torch.float32)

        # The 3D coordinates of the pinhole pixels. Dummy value now.
        # Shape (B, 3, H*W).
        self.xyz = torch.zeros((self.P, 3,1), dtype=torch.float32, device=self.device)
        self.xyz_T = torch.zeros((self.P, 1,3), dtype=torch.float32, device=self.device)

        # A data stucture relating (by index) the xyz points to the (u,v) pixels those came from.
        # Shape is (B, 2, H*W).
        self.uv_of_xyz_ix = torch.zeros((self.P, 2, 1), dtype=torch.float32, device=self.device)

        # Valid mask on the pinhole images. 
        # Shape (B, H_pin, W_pin).
        self.valid_mask = torch.ones((self.P, *self.pin_shape))

        # The 3D coordinates of the hyper-surfaces. Each pinhole image is a 'batch' in this tensor, and it has a collection of 3d points associated with it.
        # The valid mask starts out as all-valid since, for now, all we know is that points projected from the pinholes out are all valid. When the grid will b e computed for the first time, and some points projected out of the pinholes cannot be projcted back to the fisheye (out of FoV), those will be marked as invalid as well.
        pins_xyz, pins_valid_mask = self.get_xyz(back_shift_pixel=False)
        self.xyz = pins_xyz
        self.valid_mask = pins_valid_mask.view((self.P , *self.pin_shape))  
        
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
        self.uv_of_xyz_ix = pixel_coor_uv.repeat((self.P, 1 , 1))
        
        xyz, valid_mask = \
            self.pinx_camera_model.pixel_2_ray(pixel_coor) # Since this is a pinhole, all pixels that were projected out from the grid are expected to be valid. Assuming that the grid was the size of the size of the pinhole and that there were no evil bugs in the code.
        
        # Change of reference frame. This also creates points for each camera. The output is (B, 3, N=H*W)
        pins_xyz = self.Rs_pin_in_fish @ xyz

        return pins_xyz, valid_mask.repeat((self.P, 1, 1)) 
        

    def create_grid(self):
        # Get the sample locations.
        # This method produces the object self.grid, which is a size (B, H, W, 2), mapping u,v in H,W of a pinhole camera to the sampling pixels in the fisheye.
        self.grid = torch.zeros((self.P, self.pin_shape[0], self.pin_shape[1], 2), dtype=torch.float32, device=self.device)

        # Project 3D points that originated from the pinholes onto the fisheye.
        # Fish uv is of shape (B, 2, N=H*W)
        fish_uv, fish_uv_mask = self.fish_camera_model.point_3d_2_pixel(self.xyz)
        
        # So now we have three (+1 mask) tensors, all in the same order of the last dimension and the first dimension,
        # uv_of_xyz_ix (B, 2, H*W) where each column in a batch is a (u,v) pixel in the pinhole (origin of 3D point).
        # self.xyz     (B, 3, H*W) where each column in a batch is a (x,y,z) 3D point from the pinhole pixel.
        # fish_uv      (B, 2, H*W) where each column in a batch is a (u,v) pixel in the fisheye, corresponding to the pinhole pixel.
        # fish_uv_mask (B, H*W)    where each entry in a batch at an index of [pin pixel, xyz, fish pixel], is its validity.

        # Create the valid mask. For each index, find its pinhole pixel location, and assign its validity.
        # Do not redeclare the valid mask here, as it may already have some invalid pixels. If you would want to, however, you'd use: self.valid_mask = torch.ones((self.P, *self.pin_shape)).

        self.valid_mask = self.valid_mask.to(dtype=torch.int64, device='cpu')
        fish_uv_mask.to('cpu')

        # Populate the valid mask. 
        # NOTE(yoraish): this section is VERY bug-prone. A sanity-check iterative implementation is commented-out below.
        # NOTE(yoraish): this could be sped up more by removing the for loop.
        for p in range(self.P):
            fish_uv_mask_slice = fish_uv_mask[p].to(dtype=torch.int64)
            self.valid_mask[p, self.uv_of_xyz_ix[p, 0, :].to(dtype=torch.int64), self.
            uv_of_xyz_ix[p, 1, :].to(dtype=torch.int64)] = fish_uv_mask_slice
        self.valid_mask = self.valid_mask.to(self.device)

        
        # Create grid.
        fish_uv =fish_uv.to(device = self.device)
        # Normalize the points on the fisheye to range [-1., 1.] expected by the torch grid sample function.
        fish_h, fish_w = self.fish_camera_model.ss.shape
        fish_center = torch.tensor([fish_h/2, fish_w/2]).to(self.device)

        for b in range(self.P):                
            self.grid[b, self.uv_of_xyz_ix[b, 0, :].to(dtype=torch.int64), self.
            uv_of_xyz_ix[b, 1, :].to(dtype=torch.int64), :] = ((fish_uv[b, :, :].T - fish_center)/fish_center)


        # Save the rotation matrices and size of pinholes to cache. This parameterizes the grid mapping. If either of these change, then there is a need to recompute the self.grid object.
        self.cached_Rs_pin_in_fish = self.Rs_pin_in_fish.clone()
        self.cached_pin_shape = self.pin_shape

        '''
        Iterative implementation for mask and grid creation.

        print("Starting mask.")
        startmask = time.time()
        good_mask = torch.ones(self.valid_mask.shape)
        for b in range(self.P):
            for ix in range(self.uv_of_xyz_ix[b].shape[-1]):
                u_pin, v_pin = self.uv_of_xyz_ix[b, :, ix].to(dtype=torch.int64)
                good_mask[b, u_pin, v_pin] = fish_uv_mask[b, ix]
        good_mask = good_mask.to(self.device)
        print("Mask took", time.time() - startmask)

        print("Starting create grid.")
        startgrid = time.time()
        good_grid = torch.zeros(self.grid.shape)
        for b in range(self.P):
            for ix in range(self.uv_of_xyz_ix[b].shape[-1]):
                u_pin, v_pin = self.uv_of_xyz_ix[b, :, ix].to(dtype=torch.int32)
                
                # Normalize the points on the fisheye to range [-1., 1.] expected by the torch grid sample function.
                fish_h, fish_w = self.fish_camera_model.ss.shape
                fish_center = torch.tensor([fish_h/2, fish_w/2])
                good_grid[b, u_pin, v_pin] = ((fish_uv[b, :, ix] - fish_center)/fish_center)

        print("Grid took ", time.time() - startgrid)
        '''


    def __call__(self, img, interpolation='linear', invalid_pixel_value= 0 ):
        """Resample all the specified pinhole images from the input fisheye image.

        Args:
            img (np.array or torch.tensor, (B, C, H_fish, W_fish)): fisheye image.
            interpolation (str, optional): Interpolation method. Defaults to 'linear'.
            invalid_pixel_value (int, optional): value to assign to pinhole pixels that are out of the fisheye range. Defaults to 0.

        Returns:
            torch.tensor (P, C, H_pin, W_pin), torch.tensor (P, H_pin, W_pin): resampled pinholes, valid_mask on the pinholes. 
        """
        startt = time.time()

        global INTER_MAP
        # Check if the size of the pinholes has changed between now and the cache.
        # Check if the rotation matrices tensor has changed between now and the cache.
        if not self.is_same_as_cached_Rs_shape(self.Rs_pin_in_fish, self.pin_shape) or self.B != img.shape[0]:
            # If change, then recompute the self.grid object.
            self.B = img.shape[0]
            self.create_grid()

        # Convert the image to a torch tensor. If not already a tensor, then assume it is a numpy array.
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img, dtype=torch.int32, device=self.device)
        
        # With the grid we had, or the new one that we just created if we needed, 
        self.grid = self.grid.float()

        img = img.to('cuda').float()

        # Repeat the image to match the number of pinholes.
        img = img.repeat((self.P,1,1,1)) # This cannot be good.

        # Get the grid.
        grid  = self.grid

        # Resample the pinholes from the fisheye. The output is of shape (B, C, H_pin, W_pin)
        sampled = F.grid_sample( 
                                img, 
                                grid, 
                                mode=INTER_MAP[interpolation], 
                                align_corners=self.align_corners ) / 255.0

        # Apply the valid mask. This marks pinhole pixels that were outside the fisheye FoV as invalid.
        repeated_valid_mask = self.valid_mask.unsqueeze(1).repeat((1,3,1,1))
        sampled[repeated_valid_mask == 0] = invalid_pixel_value

        return sampled, self.valid_mask


    def visualize_resampling(self, img, save_dir = "", interpolation='linear', invalid_pixel_value=0, ax = None):
        """Visualize the resampled pinhole images from the input fisheye image.

        Args:
            img (ArrayLike (C, H, W)): input fisheye image. In BGR and float32 ([0,1]).
            save_dir (str, optional): path to directory for saving images. Not saved if none specified. Defaults to "".
            interpolation (str, optional): interpolation method for resampling. Defaults to 'linear'.
            invalid_pixel_value (int, optional): pixel value for pixels that cannot be resampled from the fisheye. Defaults to 0.
        """
        print(Fore.CYAN + "Visualizing resampling." + Style.RESET_ALL)

        sampled, valid_mask = self.__call__(img, interpolation, invalid_pixel_value)

        pinx_cols = 5
        pinx_rows = np.ceil(self.P/pinx_cols)


        inner = np.ones((pinx_cols, pinx_cols), dtype=str)
        inner [:,:] = 'a'

        pinx_array = np.array(np.arange((pinx_rows*pinx_cols)).reshape((-1,pinx_cols)), dtype=str)
        outer_nested_mosaic = inner.tolist() + pinx_array.tolist()
        fig = plt.figure(constrained_layout=True)
        axd = fig.subplot_mosaic(
            outer_nested_mosaic, empty_sentinel=None
        )

        for p in range(self.P):
            ax = axd[str(p*1.0)]
            pin_img = sampled[p].cpu().permute((1,2,0))
            pin_img = pin_img.numpy()
            pin_img = cv2.cvtColor(pin_img, cv2.COLOR_BGR2RGB)
            ax.imshow(pin_img)

            # Uncomment below to show the valid mask.
            # ax.imshow(self.valid_mask[p].cpu())
            ax.set_title(f"pin{p}")
        
        self.visualize_grid(img, ax = axd['a'])


        self.resample_counter += 1
        if save_dir:
            fig.savefig(os.path.join(save_dir, f"{self.resample_counter}.png"))
        else:
            plt.show()

    def visualize_grid(self, img, save_dir = "", ax = None):
        """Visualize the pixels on the fisheye img that are resampled into pinhole images.
            The participating pixels are extracted from the self.grid object.
        Args:
            img (torch.tensor (C,H,W)): The input fisheye image to be overlaid with the pinholes and shown. The images are in the range [0,255] uint8. And in BGR.
        """

        print(Fore.CYAN + "Visualizing grid.")
        print("img.shape" + str(img.shape) + Style.RESET_ALL) 
        print(Fore.CYAN + "self.img type" + str(img.dtype) + Style.RESET_ALL)

        if ax is None:
            fig = plt.figure(0)
            ax = fig.add_subplot(1,2,1)

        # Create a copy of the image to draw on.
        img = img.cpu().float().numpy() / 255.0
        img = img.transpose((1,2,0))

        # Convert the image to RGB.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Visualize the pinholes on the fish.
        # TODO(yoraish): the u's and v's here are not following the convention. Recheck that.
        print("Visualizing pinhole-resampled areas on top of the fisheye image.")
        un_to_u = self.fish_camera_model.ss.shape[1]/2
        vn_to_v = self.fish_camera_model.ss.shape[0]/2
        vns = self.grid[:, :, :, 0].flatten().cpu()
        uns = self.grid[:, :, :, 1].flatten().cpu()




        # Random choice of samples to dray.
        randixs = torch.randperm(vns.shape[0])[::2]
        vns = vns[randixs]
        uns = uns[randixs]
        for vn,un in tqdm.tqdm(zip(vns, uns), total = len(vns)):
            # u's are x's, columns. v's are y's, rows.
            u = un*un_to_u + un_to_u
            v = vn*vn_to_v + vn_to_v

            u = int(u.item())
            v = int(v.item())
            try:
                # img[v,u][0] = img[u,v] * 0.5 + 0.5 * np.array([255,0,0])
                img[u,v, :] = np.array([1.0,0.0,0.0])

            except Exception as e:
                print(f"Could not mark pixel uv {u,v} in visualization function.")
                print(e)
                continue


        ax.imshow(img)
        ax.set_title("Marked fisheye image.")

        if ax is None:
            if save_dir:
                fig.savefig(os.path.join(save_dir, f"marked_{self.resample_counter}.png"))
            else:
                plt.show()

    def is_same_as_cached_Rs_shape(self, Rs, pin_shape):
        if type(self.cached_pin_shape) == type(None) or type(self.cached_Rs_pin_in_fish) == type(None):
            return False

        if tuple(self.cached_pin_shape) == tuple(pin_shape):
            if  torch.all(self.cached_Rs_pin_in_fish == Rs) == True:
                return True