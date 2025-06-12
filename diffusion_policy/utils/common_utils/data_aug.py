import torch
import torch.nn as nn


class RandomShiftsAug:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = nn.functional.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return nn.functional.grid_sample(x, grid, padding_mode="zeros", align_corners=False)

import torch
import random
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image


class ImageAugmentation:
    # Defaults
    max_translate = 10

    brightness = 0.3
    contrast = 0.3
    saturation = 0.3
    hue = 0.3

    degrees = 5

    def __init__(
        self,
        random_translate=False,
        color_jitter=False,
        random_rotate=False,
        random_color_cutout=False,
        gaussian_blur=False,
    ):
        self.augmentations = []

        if random_translate:
            self.augmentations.append(self.random_translate)

        if color_jitter:
            self.augmentations.append(self.random_color_jitter)

        if random_rotate:
            self.augmentations.append(self.random_rotation)
            
        if random_color_cutout:
            self.augmentations.append(self.random_color_cutout)
            
        if gaussian_blur:
            self.augmentations.append(self.random_gaussian_blur)
            
    def random_translate(self, images):
        """
        Apply random translation to a batch of images.
        
        Args:
            images: torch.Tensor of shape [B, C, H, W] with values in range 0-255
        
        Returns:
            Translated images as torch.Tensor with values in range 0-255
        """
        # Ensure images is a 4D tensor [B, C, H, W]
        if len(images.shape) == 3:  # [C, H, W]
            images = images.unsqueeze(0)
            
        device = images.device
        B, C, H, W = images.shape
        translated_images = torch.zeros_like(images)
        
        # Generate random translations for each image in the batch
        translations_height = torch.randint(-self.max_translate, self.max_translate + 1, (B,), device=device)
        translations_width = torch.randint(-self.max_translate, self.max_translate + 1, (B,), device=device)
        
        for i in range(B):
            translation_height = translations_height[i].item()
            translation_width = translations_width[i].item()
            
            # Calculate the indices for zero-padded array
            start_height = max(translation_height, 0)
            end_height = H + min(translation_height, 0)
            start_width = max(translation_width, 0)
            end_width = W + min(translation_width, 0)

            # Calculate the indices for the original image
            start_height_orig = -min(translation_height, 0)
            end_height_orig = H - max(translation_height, 0)
            start_width_orig = -min(translation_width, 0)
            end_width_orig = W - max(translation_width, 0)

            # Place the original image in the translated position
            translated_images[i, :, start_height:end_height, start_width:end_width] = images[
                i, :, start_height_orig:end_height_orig, start_width_orig:end_width_orig
            ]
            
        return translated_images

    def random_color_jitter(self, images):
        """
        Apply random color jittering to a batch of images.
        
        Args:
            images: torch.Tensor of shape [B, C, H, W] with values in range 0-255
        
        Returns:
            Color jittered images as torch.Tensor with values in range 0-255
        """
        # Ensure images is a 4D tensor [B, C, H, W]
        if len(images.shape) == 3:  # [C, H, W]
            images = images.unsqueeze(0)
            
        device = images.device
        batch_size = images.shape[0]
        
        # Use torchvision directly on tensors when possible
        # First create a copy to avoid modifying the original
        result = images.clone()
        
        for i in range(batch_size):
            img = result[i].float() / 255.0  # Normalize to 0-1 for torchvision functions
            
            # Apply color jitter directly on the tensor
            if random.random() > 0.5:
                brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
                img = F.adjust_brightness(img, brightness_factor)
            
            if random.random() > 0.5:
                contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
                img = F.adjust_contrast(img, contrast_factor)
                
            if random.random() > 0.5:
                saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
                img = F.adjust_saturation(img, saturation_factor)
                
            if random.random() > 0.5:
                hue_factor = random.uniform(-self.hue, self.hue)
                img = F.adjust_hue(img, hue_factor)
                
            # Convert back to 0-255 range and preserve data type
            img = img.clamp(0, 1) * 255
            result[i] = img
            
        return result.to(device)

    def random_rotation(self, images):
        """
        Apply random rotation to a batch of images.
        
        Args:
            images: torch.Tensor of shape [B, C, H, W] with values in range 0-255
        
        Returns:
            Rotated images as torch.Tensor with values in range 0-255
        """
        # Ensure images is a 4D tensor [B, C, H, W]
        if len(images.shape) == 3:  # [C, H, W]
            images = images.unsqueeze(0)
            
        device = images.device
        batch_size = images.shape[0]
        result = []
        
        # Process each image in the batch
        for i in range(batch_size):
            # Apply rotation directly on GPU
            img = images[i].float() / 255.0  # Normalize to 0-1 for torchvision functions
            angle = random.uniform(-self.degrees, self.degrees)
            rotated_img = F.rotate(img, angle)
            
            # Convert back to 0-255 range
            rotated_img = rotated_img.clamp(0, 1) * 255
            result.append(rotated_img)
            
        # Stack all processed images back into a batch
        return torch.stack(result).to(device)

    def random_color_cutout(self, images, p=0.5, scale=(0.01, 0.04), ratio=(0.4, 1.7)):
        """
        Apply random color cutout to a batch of images.
        
        Args:
            images: torch.Tensor of shape [B, C, H, W] with values in range 0-255
            p: Probability of applying cutout
            scale: Range of area scale to cut out
            ratio: Range of aspect ratio for cutout
        
        Returns:
            Images with cutout applied as torch.Tensor with values in range 0-255
        """
        # Ensure images is a 4D tensor [B, C, H, W]
        if len(images.shape) == 3:  # [C, H, W]
            images = images.unsqueeze(0)
        
        device = images.device
        batch_size = images.shape[0]
        result = images.clone()
        
        for i in range(batch_size):
            if random.random() < p:
                _, c, h, w = result[i:i+1].shape
                area = h * w
                
                for _ in range(10):  # Try up to 10 times to find valid cutout parameters
                    target_area = random.uniform(scale[0], scale[1]) * area
                    aspect_ratio = random.uniform(ratio[0], ratio[1])
                    
                    cut_w = int((target_area * aspect_ratio) ** 0.5)
                    cut_h = int((target_area / aspect_ratio) ** 0.5)
                    
                    if cut_h <= h and cut_w <= w:
                        top = random.randint(0, h - cut_h)
                        left = random.randint(0, w - cut_w)
                        result[i, :, top:top+cut_h, left:left+cut_w] = 0
                        break
                
        return result.to(device)
    
    def random_gaussian_blur(self, images, p=0.5, kernel_size=3, sigma_range=(0.1, 1.0)):
        """
        Apply random Gaussian blur to a batch of images with 50% probability.
        
        Args:
            images: torch.Tensor of shape [B, C, H, W] or [C, H, W] with values in range 0-255
            p: Probability of applying Gaussian blur (default: 0.5)
            kernel_size: Size of the Gaussian kernel (default: 3)
            sigma_range: Range of sigma values for Gaussian kernel (default: (0.1, 1.0))
        
        Returns:
            Blurred images as torch.Tensor with values in range 0-255
        """
        # Ensure images is a 4D tensor [B, C, H, W]
        if len(images.shape) == 3:  # [C, H, W]
            images = images.unsqueeze(0)
            
        device = images.device
        batch_size = images.shape[0]
        result = images.clone()
        
        for i in range(batch_size):
            if random.random() < p:
                img = result[i].float() / 255.0  # Normalize to 0-1 for torchvision functions
                
                # Generate random sigma value within the specified range
                sigma = random.uniform(sigma_range[0], sigma_range[1])
                
                # Apply Gaussian blur
                blurred_img = F.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)
                
                # Convert back to 0-255 range
                blurred_img = blurred_img.clamp(0, 1) * 255
                result[i] = blurred_img
        
        return result.to(device)

    def __call__(self, images):
        """
        Apply all selected augmentations to 50% of the images in the batch in a batch-wise manner.
        
        Args:
            images: torch.Tensor of shape [B, C, H, W] or [C, H, W] with values in range 0-255
        
        Returns:
            Augmented images as torch.Tensor with values in range 0-255
        """
        # Validate input type
        if not isinstance(images, torch.Tensor):
            raise ValueError("Only torch.Tensor inputs are supported")
            
        # Ensure images is a 4D tensor [B, C, H, W]
        if len(images.shape) == 3:  # [C, H, W]
            images = images.unsqueeze(0)
            
        # Capture original device
        device = images.device
        batch_size = images.shape[0]
        
        # Create a mask for which images to augment (approximately 50%)
        augment_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        num_to_augment = max(1, batch_size // 2)  # At least 1 image
        indices_to_augment = torch.randperm(batch_size)[:num_to_augment]
        augment_mask[indices_to_augment] = True
        
        # Create a copy of the input images
        result = images.clone()
        
        # Get the subset of images to augment
        images_to_augment = images[augment_mask]
        
        # Apply each augmentation sequentially to the selected subset
        augmented_images = images_to_augment
        for aug in self.augmentations:
            augmented_images = aug(augmented_images)
            # Ensure we're still on the correct device after each augmentation
            if augmented_images.device != device:
                augmented_images = augmented_images.to(device)
        
        # Place the augmented images back into the result tensor
        result[augment_mask] = augmented_images
        
        return result