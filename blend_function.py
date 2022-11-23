
import cv2
import numpy as np

import torch
import kornia

class BlendBy2ndOrderGradient(object):
    def __init__(self, threshold) -> None:
        super().__init__()
        
        self.threshold = threshold
    
    def __call__(self, img):
        return self.blend_func(img)
    
class BlendBy2ndOrderGradTorch(BlendBy2ndOrderGradient):
    def __init__(self, threshold) -> None:
        super().__init__(threshold)
        
    def blend_func(self, img):
        # Compute the Laplacian.
        # img is (B, C, H, W), g is (B, C, 2, H, W).
        g = kornia.filters.spatial_gradient(img, mode='sobel', order=2)
        
        # Sum (norm) the result along the channel dimension.
        s = torch.linalg.norm( g, dim=-3, keepdim=False )
        s = torch.linalg.norm( s, dim=-3, keepdim=True )
        
        # Find the over-threshold ones.
        m = s > self.threshold
        m = m.to(torch.float32)
        
        # Add some dilation.
        # m = kornia.morphology.dilation(m, torch.ones((3, 3), device=s.device), border_type='geodesic', border_value=0.0)
        # Add some erosion.
        m = kornia.morphology.erosion(m, torch.ones((3, 3), device=s.device), border_type='geodesic', border_value=0.0)
        
        return m
    
class BlendBy2ndOrderGradOcv(object):
    def __init__(self, threshold) -> None:
        super().__init__(threshold)
        
    def blend_func(self, img):
        # Compute the Laplacian.
        # img is (H, W, ...), g is (H, W, ...).
        # g = cv2.Laplacian(img, cv2.CV_32F, borderType=cv2.BORDER_REFLECT)
        gx = cv2.Sobel(img, cv2.CV_32FC1, 2, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
        gy = cv2.Sobel(img, cv2.CV_32FC1, 0, 2, ksize=3, borderType=cv2.BORDER_REFLECT)
        
        # Sum (norm) the results.
        s = np.sqrt( gx**2 + gy**2 )
        
        # Find the over-threshold ones.
        m = s > self.threshold
        m = m.astype(np.float32)
        
        # Add some dilation.
        # m = cv2.dilate( m, np.ones((3, 3)), borderType=cv2.BORDER_CONSTANT, borderValue=0.0 )
        # Add some erosion.
        m = cv2.erode(m, np.ones((3, 3)), borderType=cv2.BORDER_CONSTANT, borderValue=0.0)
        
        return m