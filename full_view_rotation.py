
import numpy as np
import torch
import torch.nn.functional as F

from .planar_as_base import (INTER_MAP, input_2_torch, torch_2_output, PlanarAsBase)
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
        # import ipdb; ipdb.set_trace()
        # Get the longitude and latitude coordinates.
        self.lon_lat, invalid_mask = self.get_lon_lat()

        # Reshape.
        self.lon_lat = self.lon_lat.permute((1, 0)).view((*self.shape, 2))
        self.invalid_mask = invalid_mask.view(self.shape)
        
        # The grid.
        self.grid = torch.zeros( (1, *self.shape, 2), dtype=torch.float32, device=self.device )
        self.grid[0, :, :, 0] = self.lon_lat[:, :, 0] / ( 2 * np.pi ) * 2 - 1
        self.grid[0, :, :, 1] = self.lon_lat[:, :, 1] / np.pi * 2 - 1

    @PlanarAsBase.device.setter
    def device(self, device):
        PlanarAsBase.device.fset(self, device)
        
        self.lon_lat = self.lon_lat.to(device=device)
        self.invalid_mask = self.invalid_mask.to(device=device)
        self.grid = self.grid.to(device=device)

    def get_lon_lat(self):
        # Get the rays in xyz coordinates in the fisheye camera image frame (CIF).
        # The valid mask is saved in self.temp_valid_mask for compatibility concern.
        xyz, valid_mask = self.get_xyz()

        # The distance projected into the xz plane in the panorama frame.
        d = torch.linalg.norm( xyz[[0, 2], :], dim=0, keepdim=True )

        # Longitude and latitude.
        lon_lat = torch.zeros( (2, xyz.shape[1]), dtype=torch.float32, device=self.device )
        # lon_lat[0, :] = np.pi - np.arctan2( xyz[2, :], xyz[0, :] ) # Longitude.
        # lon_lat[1, :] = np.pi - np.arctan2( d, xyz[1, :] )         # Latitude.
        lon_lat[0, :] = np.pi - torch.atan2( xyz[2, :], xyz[0, :] ) # Longitude.
        lon_lat[1, :] = np.pi - torch.atan2( d, xyz[1, :] )         # Latitude.

        return lon_lat, torch.logical_not( valid_mask )

    def __call__(self, img, interpolation='linear'):
        # Convert to torch.Tensor.
        t, flag_uint8 = input_2_torch(img, self.device)

        # Get the shape of the input image.
        N, C = t.shape[:2]

        # # Get the sample location. 20220701.
        # sx = ( self.lon_lat[0, :] / ( 2 * np.pi ) * ( W - 1 ) ).reshape(self.shape)
        # sy = ( self.lon_lat[1, :] / np.pi * ( H - 1 ) ).reshape(self.shape)
        
        grid = self.grid.repeat( (N, 1, 1, 1) )

        # Sample.
        sampled = F.grid_sample( 
                                t, 
                                grid, 
                                mode=INTER_MAP[interpolation], 
                                align_corners=self.align_corners,
                                padding_mode='reflection')

        # Handle invalid pixels.
        sampled = sampled.view((N*C, *self.shape))
        sampled[:, self.invalid_mask] = 0
        sampled = sampled.view((N, C, *self.shape))

        return torch_2_output(sampled, flag_uint8), self.invalid_mask.cpu().numpy().astype(bool)

    def compute_mean_samping_diff(self, support_shape):
        '''
        support_shape is the shape of the support image.
        '''
        H, W = self.shape

        # # Get the sample location. 20220701.
        # sx = ( self.lon_lat[0, :] / ( 2 * np.pi ) * ( W - 1 ) ).reshape(self.shape)
        # sy = ( self.lon_lat[1, :] / np.pi * ( H - 1 ) ).reshape(self.shape)

        grid = self.grid.detach().clone()
        grid[0, :, :, 0] = ( grid[0, :, :, 0] + 1 ) / 2 * support_shape[1]
        grid[0, :, :, 1] = ( grid[0, :, :, 1] + 1 ) / 2 * support_shape[0]

        valid_mask = torch.logical_not( self.invalid_mask )

        d = self.compute_8_way_sample_msr_diff( grid, valid_mask.unsqueeze(0).unsqueeze(0) )
        
        return torch_2_output(d, flag_uint8=False), valid_mask.cpu().numpy().astype(bool)
