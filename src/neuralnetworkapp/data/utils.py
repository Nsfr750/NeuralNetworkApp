"""
Data loading and preprocessing utilities for neural networks.

This module provides functions for loading, preprocessing, and augmenting data
for training and evaluating neural networks.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets, transforms
from typing import Tuple, List, Dict, Any, Optional, Union, Callable
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

from neuralnetworkapp.data.augmentation import ImageAugmentor, AugmentationType


class TabularDataset(Dataset):
    """
    A PyTorch Dataset for tabular data.
    
    Args:
        features: Input features as a numpy array or pandas DataFrame
        targets: Target values as a numpy array or pandas Series
        feature_columns: List of feature column names (if features is a DataFrame)
        target_column: Name of the target column (if targets is a DataFrame/Series)
        transform: Optional transform to be applied to the features
        target_transform: Optional transform to be applied to the targets
    """
    def __init__(
        self,
        features: Union[np.ndarray, pd.DataFrame],
        targets: Optional[Union[np.ndarray, pd.Series]] = None,
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        # Handle features
        if isinstance(features, pd.DataFrame):
            if feature_columns is not None:
                self.features = features[feature_columns].values
            else:
                self.features = features.values
        else:
            self.features = features
        
        # Handle targets
        self.has_targets = targets is not None
        if self.has_targets:
            if isinstance(targets, (pd.Series, pd.DataFrame)):
                if target_column is not None and isinstance(targets, pd.DataFrame):
                    self.targets = targets[target_column].values
                else:
                    self.targets = targets.values
            else:
                self.targets = targets
        else:
            self.targets = np.zeros(len(self.features))  # Dummy targets
        
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features[idx]
        target = self.targets[idx] if self.has_targets else torch.tensor(-1)  # Dummy target
        
        # Convert to tensor if not already
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.long if self.has_targets else torch.float32)
        
        # Apply transforms
        if self.transform:
            features = self.transform(features)
        
        if self.target_transform and self.has_targets:
            target = self.target_transform(target)
        
        return features, target


class ImageDataset(Dataset):
    """
    A PyTorch Dataset for image data.
    
    Args:
        root_dir: Root directory containing the dataset
        transform: Optional transform to be applied to the images
        target_transform: Optional transform to be applied to the targets
        is_test: If True, the dataset doesn't have targets (for inference)
    """
    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_test: bool = False
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.is_test = is_test
        
        # Get list of image files
        self.image_files = sorted([f for f in self.root_dir.glob('**/*') 
                                 if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
        
        # Get class labels if not test set
        if not self.is_test:
            self.classes = sorted(list(set(f.parent.name for f in self.image_files)))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            self.targets = [self.class_to_idx[f.parent.name] for f in self.image_files]
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_files[idx]
        
        # Load image
        img = self._load_image(img_path)
        
        # Get target if not test set
        if self.is_test:
            target = torch.tensor(-1)  # Dummy target for test set
        else:
            target = self.targets[idx]
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target, dtype=torch.long)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        if self.target_transform and not self.is_test:
            target = self.target_transform(target)
        
        return img, target
    
    def _load_image(self, path: Union[str, Path]) -> torch.Tensor:
        """Load an image from disk using Wand."""
        from wand.image import Image as WandImage
        import numpy as np
        
        with WandImage(filename=str(path)) as img:
            # Convert to RGB if not already
            if img.colorspace not in ('rgb', 'srgb'):
                img.transform_colorspace('rgb')
            
            # Convert to numpy array and normalize to [0, 1]
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Convert to CHW format
            if img_array.ndim == 2:  # Grayscale
                img_array = img_array[None, :, :]  # Add channel dimension
            else:  # RGB or RGBA
                img_array = img_array.transpose(2, 0, 1)  # HWC to CHW
                if img_array.shape[0] == 4:  # RGBA to RGB
                    img_array = img_array[:3, :, :]
            
            return torch.from_numpy(img_array)


def get_tabular_dataloaders(
    features: Union[np.ndarray, pd.DataFrame],
    targets: Optional[Union[np.ndarray, pd.Series]] = None,
    feature_columns: Optional[List[str]] = None,
    target_column: Optional[str] = None,
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    batch_size: int = 32,
    random_state: int = 42,
    normalize: bool = True,
    scale: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
    **dataloader_kwargs
) -> Tuple[DataLoader, ...]:
    """
    Create train, validation, and test dataloaders for tabular data.
    
    Args:
        features: Input features as a numpy array or pandas DataFrame
        targets: Target values as a numpy array or pandas Series
        feature_columns: List of feature column names (if features is a DataFrame)
        target_column: Name of the target column (if targets is a DataFrame/Series)
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation (None for no validation set)
        batch_size: Batch size for the dataloaders
        random_state: Random seed for reproducibility
        normalize: Whether to normalize features to have zero mean and unit variance
        scale: Whether to scale features to [0, 1] range
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        **dataloader_kwargs: Additional arguments to pass to DataLoader
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader) or (train_loader, test_loader)
        if val_size is None
    """
    # Convert features and targets to numpy arrays if they are DataFrames/Series
    if isinstance(features, pd.DataFrame):
        if feature_columns is not None:
            features = features[feature_columns].values
        else:
            features = features.values
    
    if targets is not None and isinstance(targets, (pd.Series, pd.DataFrame)):
        if target_column is not None and isinstance(targets, pd.DataFrame):
            targets = targets[target_column].values
        else:
            targets = targets.values
    
    # Split data into train and test sets
    if test_size > 0:
        if targets is not None:
            # Use stratified split for classification tasks
            if len(np.unique(targets)) < 10:  # Arbitrary threshold for classification
                train_idx, test_idx = train_test_split(
                    np.arange(len(features)),
                    test_size=test_size,
                    random_state=random_state,
                    stratify=targets
                )
            else:
                train_idx, test_idx = train_test_split(
                    np.arange(len(features)),
                    test_size=test_size,
                    random_state=random_state
                )
        else:
            train_idx, test_idx = train_test_split(
                np.arange(len(features)),
                test_size=test_size,
                random_state=random_state
            )
    else:
        train_idx = np.arange(len(features))
        test_idx = np.array([])
    
    # Further split training data into train and validation sets
    if val_size is not None and val_size > 0:
        if targets is not None and len(np.unique(targets[train_idx])) < 10:
            train_idx, val_idx = train_test_split(
                train_idx,
                test_size=val_size / (1 - test_size + 1e-10),  # Adjust for test split
                random_state=random_state,
                stratify=targets[train_idx] if targets is not None else None
            )
        else:
            train_idx, val_idx = train_test_split(
                train_idx,
                test_size=val_size / (1 - test_size + 1e-10),  # Adjust for test split
                random_state=random_state
            )
    else:
        val_idx = np.array([])
    
    # Create datasets
    datasets = {}
    
    # Fit scalers on training data
    if normalize or scale:
        if normalize:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features[train_idx])
        elif scale:
            scaler = MinMaxScaler()
            features_scaled = scaler.fit_transform(features[train_idx])
        
        # Apply scaling to all splits
        if len(train_idx) > 0:
            features_train = features_scaled
        if len(val_idx) > 0:
            features_val = scaler.transform(features[val_idx])
        if len(test_idx) > 0:
            features_test = scaler.transform(features[test_idx])
    else:
        if len(train_idx) > 0:
            features_train = features[train_idx]
        if len(val_idx) > 0:
            features_val = features[val_idx]
        if len(test_idx) > 0:
            features_test = features[test_idx]
    
    # Create datasets
    if len(train_idx) > 0:
        datasets['train'] = TabularDataset(
            features_train,
            targets[train_idx] if targets is not None else None
        )
    
    if len(val_idx) > 0:
        datasets['val'] = TabularDataset(
            features_val,
            targets[val_idx] if targets is not None else None
        )
    
    if len(test_idx) > 0:
        datasets['test'] = TabularDataset(
            features_test,
            targets[test_idx] if targets is not None else None
        )
    
    # Create dataloaders
    dataloaders = {}
    
    if 'train' in datasets:
        dataloaders['train'] = DataLoader(
            datasets['train'],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **dataloader_kwargs
        )
    
    if 'val' in datasets:
        dataloaders['val'] = DataLoader(
            datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            **dataloader_kwargs
        )
    
    if 'test' in datasets:
        dataloaders['test'] = DataLoader(
            datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            **dataloader_kwargs
        )
    
    return tuple(dataloaders.values())


def get_image_dataloaders(
    data_dir: Union[str, Path],
    image_size: Tuple[int, int] = (32, 32),
    batch_size: int = 32,
    num_workers: int = 4,
    augment: bool = True,
    augment_config: Optional[Dict[str, Any]] = None,
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    random_state: int = 42,
    **dataloader_kwargs
) -> Tuple[DataLoader, ...]:
    """
    Create train, validation, and test dataloaders for image data.
    
    Args:
        data_dir: Root directory containing the dataset (should have subdirectories for each class)
        image_size: Size to resize images to (height, width)
        batch_size: Batch size for the dataloaders
        num_workers: Number of worker processes for data loading
        augment: Whether to apply data augmentation to the training set
        augment_config: Configuration for data augmentation
        normalize: Whether to normalize images
        mean: Mean for normalization (per channel)
        std: Standard deviation for normalization (per channel)
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation (None for no validation set)
        random_state: Random seed for reproducibility
        **dataloader_kwargs: Additional arguments to pass to DataLoader
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader) or (train_loader, test_loader)
        if val_size is None
    """
    data_dir = Path(data_dir)
    
    # Define transforms
    train_transforms = []
    test_transforms = []
    
    # Resize images
    train_transforms.append(transforms.Resize(image_size))
    test_transforms.append(transforms.Resize(image_size))
    
    # Data augmentation for training
    if augment:
        if augment_config is None:
            augment_config = {
                'random_crop': True,
                'random_hflip': True,
                'color_jitter': True,
                'random_rotation': 15,
                'normalize': normalize,
                'mean': mean,
                'std': std
            }
        
        # Create augmentations
        aug_list = []
        
        if augment_config.get('random_crop', False):
            padding = augment_config.get('padding', 4)
            train_transforms.append(transforms.RandomCrop(
                image_size, 
                padding=padding,
                padding_mode='reflect'
            ))
        
        if augment_config.get('random_hflip', False):
            p = augment_config.get('hflip_prob', 0.5)
            train_transforms.append(transforms.RandomHorizontalFlip(p=p))
        
        if augment_config.get('color_jitter', False):
            brightness = augment_config.get('brightness', 0.2)
            contrast = augment_config.get('contrast', 0.2)
            saturation = augment_config.get('saturation', 0.2)
            hue = augment_config.get('hue', 0.1)
            train_transforms.append(transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue
            ))
        
        if 'random_rotation' in augment_config:
            degrees = augment_config['random_rotation']
            train_transforms.append(transforms.RandomRotation(degrees))
    
    # Convert to tensor
    train_transforms.append(transforms.ToTensor())
    test_transforms.append(transforms.ToTensor())
    
    # Normalize
    if normalize:
        mean = augment_config.get('mean', mean) if augment_config else mean
        std = augment_config.get('std', std) if augment_config else std
        
        train_transforms.append(transforms.Normalize(mean, std))
        test_transforms.append(transforms.Normalize(mean, std))
    
    # Compose transforms
    train_transform = transforms.Compose(train_transforms)
    test_transform = transforms.Compose(test_transforms)
    
    # Load dataset
    full_dataset = datasets.ImageFolder(root=str(data_dir), transform=train_transform)
    
    # Split into train, val, test
    if test_size > 0 or (val_size is not None and val_size > 0):
        # First split into train+val and test
        if test_size > 0:
            train_val_idx, test_idx = train_test_split(
                range(len(full_dataset)),
                test_size=test_size,
                random_state=random_state,
                stratify=full_dataset.targets
            )
        else:
            train_val_idx = range(len(full_dataset))
            test_idx = []
        
        # Then split train into train and val
        if val_size is not None and val_size > 0:
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_size / (1 - test_size + 1e-10),  # Adjust for test split
                random_state=random_state,
                stratify=[full_dataset.targets[i] for i in train_val_idx]
            )
        else:
            train_idx = train_val_idx
            val_idx = []
        
        # Create subsets
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx) if val_idx else None
        test_dataset = Subset(full_dataset, test_idx) if test_idx else None
        
        # For test set, use test transform
        if test_dataset is not None:
            test_dataset = _apply_transform_to_subset(test_dataset, test_transform)
    else:
        train_dataset = full_dataset
        val_dataset = None
        test_dataset = None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        **dataloader_kwargs
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            **dataloader_kwargs
        )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            **dataloader_kwargs
        )
    
    # Return appropriate tuple based on what's available
    if val_loader is not None and test_loader is not None:
        return train_loader, val_loader, test_loader
    elif test_loader is not None:
        return train_loader, test_loader
    elif val_loader is not None:
        return train_loader, val_loader
    else:
        return (train_loader,)


def _apply_transform_to_subset(subset: Subset, transform: Callable) -> Subset:
    """Apply a transform to all elements in a Subset."""
    class TransformedSubset(Subset):
        def __getitem__(self, idx):
            x, y = super().__getitem__(idx)
            return transform(x), y
    
    return TransformedSubset(subset.dataset, subset.indices)


def get_cross_validation_splits(
    dataset: Dataset,
    n_splits: int = 5,
    random_state: int = 42,
    stratified: bool = True
) -> List[Tuple[Subset, Subset]]:
    """
    Generate cross-validation splits for a dataset.
    
    Args:
        dataset: PyTorch dataset
        n_splits: Number of folds
        random_state: Random seed for reproducibility
        stratified: Whether to use stratified k-fold for classification tasks
        
    Returns:
        List of (train_indices, val_indices) tuples
    """
    # Check if we should use stratified k-fold
    if stratified and hasattr(dataset, 'targets'):
        targets = dataset.targets
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )
        splits = list(skf.split(np.zeros(len(targets)), targets))
    else:
        kf = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )
        splits = list(kf.split(range(len(dataset))))
    
    # Convert to Subsets
    cv_splits = []
    for train_idx, val_idx in splits:
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        cv_splits.append((train_subset, val_subset))
    
    return cv_splits


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for a dataset.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        **kwargs: Additional arguments to pass to DataLoader
        
    Returns:
        DataLoader for the dataset
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )
