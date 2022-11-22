
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
    
class BlendBy2ndOrderGradTorch(object):
    def __init__(self, threshold) -> None:
        super().__init__(threshold)
        
        self.laplacian = kornia.filters.Laplacian(kernel_size=3, border_type='reflect', normalized=True)
        
    def blend_func(self, img):
        # Compute the Laplacian.
        # img is (B, C, H, W), g is (B, C, H, W).
        g = self.laplacian(img)
        
        # Sum (norm) the result along the channel dimension.
        s = torch.linalg.norm( g, dim=-3, keepdim=True  )
        
        # Add some dilation.
        d = kornia.morphology.dilation(s, torch.ones(3, 3), border_type='geodesic', border_value=0.0)
        
        # Find the over-threshold ones.
        m = d > self.threshold
        
        return m.to(dtype=torch.float32)
    
class BlendBy2ndOrderGradOcv(object):
    def __init__(self, threshold) -> None:
        super().__init__(threshold)
        
    def blend_func(self, img):
        # Compute the Laplacian.
        # img is (H, W, C), g is (H, W, C).
        g = cv2.Laplacian(img, cv2.CV_32F, borderType=cv2.BORDER_REFLECT)
        
        # Sum (norm) the result along the channel dimension.
        s = np.linalg.norm( g, axis=-1, keepdims=True )
        
        # Add some dilation.
        d = cv2.dilate( s, np.ones((3, 3)), borderType=cv2.BORDER_CONSTANT, borderValue=0.0 )
        
        # Find the over-threshold ones.
        m = d > self.threshold
        
        return m.astype(np.float32)