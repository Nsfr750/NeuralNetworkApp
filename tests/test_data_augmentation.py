"""
Tests for data augmentation pipelines.
"""

import unittest
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset

class TestDataAugmentation(unittest.TestCase):
    """Test cases for data augmentation pipelines."""
    
    def setUp(self):
        """Set up test data."""
        # Create a simple dataset
        self.x = torch.randn(10, 3, 32, 32)  # 10 samples, 3 channels, 32x32
        self.y = torch.randint(0, 10, (10,))  # 10 class labels
        self.dataset = TensorDataset(self.x, self.y)
    
    def test_basic_augmentations(self):
        """Test basic data augmentations."""
        # Define augmentation pipeline
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),
            transforms.ToTensor(),
        ])
        
        # Create a DataLoader with augmentation
        class AugmentedDataset(TensorDataset):
            def __init__(self, x, y, transform=None):
                super().__init__(x, y)
                self.transform = transform
            
            def __getitem__(self, idx):
                x, y = self.x[idx], self.y[idx]
                if self.transform:
                    x = self.transform(x)
                return x, y
        
        aug_dataset = AugmentedDataset(self.x, self.y, transform=transform)
        dataloader = DataLoader(aug_dataset, batch_size=2, shuffle=True)
        
        # Test that we can iterate through the dataloader
        for batch_x, batch_y in dataloader:
            self.assertEqual(batch_x.shape[0], 2)  # Batch size of 2
            self.assertEqual(batch_x.shape[1:], (3, 32, 32))  # Correct shape
            self.assertEqual(batch_y.shape, (2,))  # Correct label shape
            break  # Just test one batch
    
    def test_normalization(self):
        """Test data normalization."""
        # Define mean and std for normalization
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        # Apply normalization
        normalized = torch.stack([transform(img) for img in self.x])
        
        # Verify mean and std are approximately correct
        for c in range(3):
            channel_mean = normalized[:, c, :, :].mean()
            channel_std = normalized[:, c, :, :].std()
            self.assertAlmostEqual(channel_mean.item(), 0, places=1)
            self.assertAlmostEqual(channel_std.item(), 1, places=1)
    
    def test_custom_augmentations(self):
        """Test custom augmentation functions."""
        # Define a custom augmentation
        def random_erasing(img, p=0.5, sl=0.02, sh=0.4, r1=0.3):
            if torch.rand(1) > p:
                return img
            
            _, h, w = img.shape
            area = h * w
            
            for _ in range(100):  # Try at most 100 times
                target_area = torch.empty(1).uniform_(sl, sh).item() * area
                aspect_ratio = torch.empty(1).uniform_(r1, 1/r1).item()
                
                h_er = int(round((target_area * aspect_ratio) ** 0.5))
                w_er = int(round((target_area / aspect_ratio) ** 0.5))
                
                if w_er < w and h_er < h:
                    x1 = torch.randint(0, h - h_er, (1,)).item()
                    y1 = torch.randint(0, w - w_er, (1,)).item()
                    
                    img[0, x1:x1+h_er, y1:y1+w_er] = 0
                    img[1, x1:x1+h_er, y1:y1+w_er] = 0
                    img[2, x1:x1+h_er, y1:y1+w_er] = 0
                    break
            
            return img
        
        # Test the custom augmentation
        img = torch.randn(3, 32, 32)
        augmented_img = random_erasing(img.clone())
        
        # Verify something changed (with high probability)
        self.assertFalse(torch.allclose(img, augmented_img))


if __name__ == "__main__":
    unittest.main()
