"""
Data augmentation module for neural network training.

This module provides various data augmentation techniques for image and sequential data.
"""

import torch
import random
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any, Callable
from enum import Enum
from wand.image import Image as WandImage
from wand.api import library
import ctypes

# Define wand image processing functions
def wand_to_tensor(wand_img):
    """Convert a Wand image to a PyTorch tensor."""
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(wand_img, dtype=np.float32) / 255.0
    
    # Convert to CHW format
    if len(img_array.shape) == 2:  # Grayscale
        img_array = img_array[None, :, :]  # Add channel dimension
    else:  # RGB or RGBA
        img_array = img_array.transpose(2, 0, 1)  # HWC to CHW
        if img_array.shape[0] == 4:  # RGBA to RGB
            img_array = img_array[:3, :, :]
    
    return torch.from_numpy(img_array)

def tensor_to_wand(tensor):
    """Convert a PyTorch tensor to a Wand image."""
    # Convert to HWC format
    if tensor.dim() == 4:  # Batch of images
        tensor = tensor[0]  # Take first image in batch
    
    if tensor.dim() == 3:  # CHW format
        tensor = tensor.permute(1, 2, 0)  # CHW to HWC
    
    # Convert to numpy and scale to 0-255
    img_array = (tensor.numpy() * 255).astype(np.uint8)
    
    # Create Wand image
    return WandImage.from_array(img_array)


class AugmentationType(Enum):
    """Supported augmentation types."""
    RANDOM_CROP = 'random_crop'
    RANDOM_HORIZONTAL_FLIP = 'random_hflip'
    RANDOM_VERTICAL_FLIP = 'random_vflip'
    RANDOM_ROTATION = 'random_rotation'
    COLOR_JITTER = 'color_jitter'
    RANDOM_AFFINE = 'random_affine'
    RANDOM_PERSPECTIVE = 'random_perspective'
    GAUSSIAN_BLUR = 'gaussian_blur'
    RANDOM_ERASING = 'random_erasing'
    NORMALIZE = 'normalize'
    RANDOM_NOISE = 'random_noise'
    RANDOM_SHIFT = 'random_shift'
    RANDOM_ZOOM = 'random_zoom'
    CUTOUT = 'cutout'
    MIXUP = 'mixup'


class ImageAugmentor:
    """
    Image data augmentation class using Wand for image processing.
    
    Args:
        augmentations: List of augmentation types to apply
        img_size: Output image size (height, width)
        mean: Mean for normalization
        std: Standard deviation for normalization
        **aug_params: Additional parameters for specific augmentations
    """
    def __init__(
        self,
        augmentations: List[Union[str, AugmentationType]],
        img_size: Tuple[int, int] = (32, 32),
        mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
        std: Tuple[float, ...] = (0.5, 0.5, 0.5),
        **aug_params
    ):
        self.augmentations = [aug if isinstance(aug, AugmentationType) else AugmentationType(aug) 
                            for aug in augmentations]
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.aug_params = aug_params
        
        # Initialize augmentations
        self.transforms = self._build_transforms()
        
        # Wand augmentation parameters
        self.crop_scale = aug_params.get('crop_scale', (0.8, 1.0))
        self.hflip_prob = aug_params.get('hflip_prob', 0.5)
        self.vflip_prob = aug_params.get('vflip_prob', 0.5)
        self.rotation_prob = aug_params.get('rotation_prob', 0.5)
        self.rotation_range = aug_params.get('rotation_range', 15)
        self.color_jitter_prob = aug_params.get('color_jitter_prob', 0.5)
        self.affine_prob = aug_params.get('affine_prob', 0.5)
        self.blur_prob = aug_params.get('blur_prob', 0.5)
    
    def _build_transforms(self):
        """Build the transformation pipeline using Wand."""
        # For Wand, we don't need to build a transform pipeline
        # as we'll handle everything in the __call__ method
        return None
        
    def _get_random_crop_params(self, wand_img):
        """Get random crop parameters."""
        scale = random.uniform(self.crop_scale[0], self.crop_scale[1])
        width, height = wand_img.size
        target_width = int(width * scale)
        target_height = int(height * scale)
        
        if width == target_width and height == target_height:
            return 0, 0, height, width
            
        i = random.randint(0, height - target_height)
        j = random.randint(0, width - target_width)
        return i, j, target_height, target_width
        
    def _get_jitter_param(self, min_val=0.8, max_val=1.2):
        """Get a random jitter parameter."""
        return random.uniform(min_val, max_val)
    
    def _add_random_noise(self, img: torch.Tensor) -> torch.Tensor:
        """Add random Gaussian noise to the image."""
        if not isinstance(img, torch.Tensor):
            img = TF.to_tensor(img)
            
        noise_std = self.aug_params.get('noise_std', 0.1)
        noise = torch.randn_like(img) * noise_std
        img = img + noise
        return torch.clamp(img, 0, 1)
    
    def _apply_cutout(self, img: torch.Tensor) -> torch.Tensor:
        """Apply cutout augmentation."""
        if not isinstance(img, torch.Tensor):
            img = TF.to_tensor(img)
            
        h, w = img.shape[1:]
        length = self.aug_params.get('cutout_length', 16)
        n_holes = self.aug_params.get('n_holes', 1)
        
        for _ in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)
            
            img[:, y1:y2, x1:x2] = 0
            
        return img
    
    def __call__(self, img):
        """
        Apply augmentations to the input image using Wand.
        
        Args:
            img: Input image tensor in CHW format
            
        Returns:
            Augmented image tensor
        """
        # Convert tensor to Wand image
        wand_img = tensor_to_wand(img)
        
        # Apply augmentations
        for aug_type in self.augmentations:
            if aug_type == AugmentationType.RANDOM_CROP:
                if random.random() < self.aug_params.get('crop_prob', 0.5):
                    i, j, h, w = self._get_random_crop_params(wand_img)
                    wand_img.crop(left=j, top=i, width=w, height=h)
                    wand_img.resize(self.img_size[1], self.img_size[0])
                    
            elif aug_type == AugmentationType.RANDOM_HORIZONTAL_FLIP:
                if random.random() < self.hflip_prob:
                    wand_img.flop()
                    
            elif aug_type == AugmentationType.RANDOM_VERTICAL_FLIP:
                if random.random() < self.vflip_prob:
                    wand_img.flip()
                    
            elif aug_type == AugmentationType.RANDOM_ROTATION:
                if random.random() < self.rotation_prob:
                    angle = random.uniform(-self.rotation_range, self.rotation_range)
                    wand_img.rotate(angle)
                    
            elif aug_type == AugmentationType.COLOR_JITTER:
                if random.random() < self.color_jitter_prob:
                    # Adjust brightness, contrast, saturation, and hue
                    brightness = self._get_jitter_param(0.8, 1.2)
                    saturation = self._get_jitter_param(0.8, 1.2)
                    hue = random.uniform(-10, 10)  # -10% to +10% hue shift
                    wand_img.modulate(brightness * 100, saturation * 100, 100 + hue)
                    
            elif aug_type == AugmentationType.RANDOM_AFFINE:
                if random.random() < self.affine_prob:
                    # Apply random affine transform
                    angle = random.uniform(-10, 10)
                    translate_x = random.uniform(-0.1, 0.1) * wand_img.width
                    translate_y = random.uniform(-0.1, 0.1) * wand_img.height
                    scale = random.uniform(0.9, 1.1)
                    
                    if angle != 0:
                        wand_img.rotate(angle)
                    if scale != 1.0:
                        new_width = int(wand_img.width * scale)
                        new_height = int(wand_img.height * scale)
                        wand_img.resize(new_width, new_height)
                    if translate_x != 0 or translate_y != 0:
                        wand_img.distort('affine', (1, 0, translate_x, 0, 1, translate_y))
                    
            elif aug_type == AugmentationType.GAUSSIAN_BLUR:
                if random.random() < self.blur_prob:
                    sigma = random.uniform(0.1, 2.0)
                    wand_img.gaussian_blur(radius=0, sigma=sigma)
                    
            elif aug_type == AugmentationType.NORMALIZE:
                # Convert to tensor for normalization
                img_tensor = wand_to_tensor(wand_img)
                mean = torch.tensor(self.mean).view(-1, 1, 1)
                std = torch.tensor(self.std).view(-1, 1, 1)
                img_tensor = (img_tensor - mean) / std
                return img_tensor
                
            elif aug_type == AugmentationType.RANDOM_NOISE:
                if random.random() < self.aug_params.get('noise_prob', 0.5):
                    # Add random noise using Wand's noise function
                    wand_img.noise('gaussian', attenuate=0.5)
                    
            elif aug_type == AugmentationType.CUTOUT:
                if random.random() < self.aug_params.get('cutout_prob', 0.5):
                    # Convert to tensor, apply cutout, then back to Wand
                    img_tensor = wand_to_tensor(wand_img)
                    img_tensor = self._apply_cutout(img_tensor)
                    wand_img = tensor_to_wand(img_tensor)
        
        # Convert back to tensor
        return wand_to_tensor(wand_img)
    
    def __repr__(self) -> str:
        return f"ImageAugmentor(augmentations={[a.value for a in self.augmentations]})"


def get_default_image_augmentations(
    img_size: Tuple[int, int] = (32, 32),
    mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
    std: Tuple[float, ...] = (0.5, 0.5, 0.5)
) -> ImageAugmentor:
    """
    Get a standard set of image augmentations using Wand.
    
    Args:
        img_size: Output image size (height, width)
        mean: Mean for normalization
        std: Standard deviation for normalization
        
    Returns:
        ImageAugmentor with standard augmentations
    """
    augmentations = [
        AugmentationType.RANDOM_CROP,
        AugmentationType.RANDOM_HORIZONTAL_FLIP,
        AugmentationType.COLOR_JITTER,
        AugmentationType.NORMALIZE
    ]
    
    aug_params = {
        'crop_scale': (0.8, 1.0),
        'crop_prob': 1.0,  # Always apply crop to ensure consistent size
        'hflip_prob': 0.5,
        'color_jitter_prob': 0.5,
    }
    
    return ImageAugmentor(
        augmentations=augmentations,
        img_size=img_size,
        mean=mean,
        std=std,
        **aug_params
    )


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """
    Applies mixup augmentation to a batch of data.
    
    Args:
        x: Input batch of images [batch_size, channels, height, width]
        y: Input batch of labels [batch_size]
        alpha: Mixup alpha parameter (controls the strength of interpolation)
        
    Returns:
        Mixed inputs, targets_a, targets_b, and mixing coefficient lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """
    Applies cutmix augmentation to a batch of data.
    
    Args:
        x: Input batch of images [batch_size, channels, height, width]
        y: Input batch of labels [batch_size]
        alpha: Cutmix alpha parameter (controls the strength of interpolation)
        
    Returns:
        Mixed inputs, targets_a, targets_b, and mixing coefficient lambda
    """
    if alpha <= 0:
        return x, y, 1.0, y
        
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size).to(x.device)
    
    # Generate random bounding box
    lam = np.random.beta(alpha, alpha)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    
    # Center coordinates
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    # Bounding box coordinates
    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    
    # Apply cutmix
    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    
    # Adjust lambda to account for the actual area cut
    lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam
