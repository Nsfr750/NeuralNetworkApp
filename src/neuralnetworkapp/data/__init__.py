import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
from typing import Tuple, Dict, List, Optional, Union, Callable
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

class TabularDataset(Dataset):
    """
    A PyTorch Dataset for tabular data.
    """
    def __init__(
        self, 
        X: np.ndarray, 
        y: np.ndarray = None, 
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples,)
            transform: Optional transform to apply to the features
            target_transform: Optional transform to apply to the targets
        """
        self.X = X
        self.y = y
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        
        if self.y is not None:
            y = self.y[idx]
            if self.target_transform:
                y = self.target_transform(y)
            return x, y
        return x


def load_tabular_data(
    file_path: str,
    target_column: str = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    normalize: bool = True,
    standardize: bool = False,
    **kwargs
) -> Tuple[Dict[str, Dataset], Dict]:
    """
    Load tabular data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        target_column: Name of the target column
        test_size: Fraction of data to use for testing
        val_size: Fraction of training data to use for validation
        random_state: Random seed for reproducibility
        normalize: Whether to normalize features to [0, 1]
        standardize: Whether to standardize features to mean=0, std=1
        **kwargs: Additional arguments to pass to pd.read_csv()
        
    Returns:
        A tuple containing:
            - Dictionary of datasets {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
            - Dictionary of metadata about the dataset
    """
    # Load data
    df = pd.read_csv(file_path, **kwargs)
    
    # Separate features and target
    if target_column is not None:
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values
        
        # Encode target if it's not numeric
        if not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y = le.fit_transform(y)
            num_classes = len(le.classes_)
        else:
            num_classes = len(np.unique(y))
    else:
        X = df.values
        y = None
        num_classes = 0
    
    # Split into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y is not None else None
    )
    
    # Further split training data into train and validation
    val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size to be relative to train_val size
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=val_size_adjusted, 
        random_state=random_state,
        stratize=y_train_val if y_train_val is not None else None
    )
    
    # Create feature scalers
    feature_scaler = None
    if normalize or standardize:
        if standardize:
            feature_scaler = StandardScaler()
        else:
            feature_scaler = MinMaxScaler()
        
        X_train = feature_scaler.fit_transform(X_train)
        X_val = feature_scaler.transform(X_val)
        X_test = feature_scaler.transform(X_test)
    
    # Create datasets
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    test_dataset = TabularDataset(X_test, y_test)
    
    # Create metadata
    metadata = {
        'num_features': X.shape[1],
        'num_classes': num_classes,
        'feature_names': df.columns.tolist() if hasattr(df, 'columns') else None,
        'class_names': le.classes_.tolist() if 'le' in locals() else None,
        'feature_scaler': feature_scaler
    }
    
    return {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}, metadata


def create_data_loaders(
    datasets: Dict[str, Dataset],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> Dict[str, DataLoader]:
    """
    Create data loaders from datasets.
    
    Args:
        datasets: Dictionary of datasets
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of subprocesses to use for data loading
        **kwargs: Additional arguments to pass to DataLoader
        
    Returns:
        Dictionary of data loaders
    """
    loaders = {}
    
    for split, dataset in datasets.items():
        is_train = split == 'train'
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and is_train,  # Only shuffle training data
            num_workers=num_workers,
            **kwargs
        )
    
    return loaders


def load_image_data(
    data_dir: str,
    image_size: Tuple[int, int] = (32, 32),
    test_size: float = 0.2,
    val_size: float = 0.1,
    batch_size: int = 32,
    num_workers: int = 0,
    **kwargs
) -> Tuple[Dict[str, DataLoader], Dict]:
    """
    Load image data from a directory.
    
    The directory structure should be:
    data_dir/
        class1/
            img1.jpg
            img2.jpg
            ...
        class2/
            img1.jpg
            img2.jpg
            ...
        ...
    
    Args:
        data_dir: Path to the data directory
        image_size: Size to resize images to (height, width)
        test_size: Fraction of data to use for testing
        val_size: Fraction of training data to use for validation
        batch_size: Batch size
        num_workers: Number of subprocesses to use for data loading
        **kwargs: Additional arguments to pass to ImageFolder
        
    Returns:
        A tuple containing:
            - Dictionary of data loaders {'train': train_loader, 'val': val_loader, 'test': test_loader}
            - Dictionary of metadata about the dataset
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform, **kwargs)
    
    # Split into train, validation, and test sets
    train_val_size = int((1 - test_size) * len(full_dataset))
    test_size = len(full_dataset) - train_val_size
    train_val_dataset, test_dataset = random_split(
        full_dataset, [train_val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Further split training data into train and validation
    val_size = int(val_size * len(train_val_dataset))
    train_size = len(train_val_dataset) - val_size
    train_dataset, val_dataset = random_split(
        train_val_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    
    # Create metadata
    metadata = {
        'num_classes': len(full_dataset.classes),
        'class_names': full_dataset.classes,
        'class_to_idx': full_dataset.class_to_idx,
        'image_size': image_size,
        'num_channels': 3  # RGB
    }
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }, metadata
