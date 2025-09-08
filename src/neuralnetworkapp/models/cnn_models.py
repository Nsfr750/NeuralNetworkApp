"""
Convolutional Neural Network Models

This module provides building blocks and pre-defined architectures for CNNs,
including support for batch normalization and dropout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Tuple, Dict, Any


class ConvBlock(nn.Module):
    """
    A convolutional block with optional batch normalization, activation, and dropout.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        batch_norm: bool = True,
        activation: str = 'relu',
        dropout: float = 0.0,
        pool: Optional[Union[int, Tuple[int, int]]] = None,
        pool_type: str = 'max'
    ):
        """
        Initialize the convolutional block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution
            padding: Zero-padding added to both sides of the input
            dilation: Spacing between kernel elements
            groups: Number of blocked connections from input to output channels
            bias: If True, adds a learnable bias to the output
            batch_norm: If True, adds batch normalization
            activation: Activation function ('relu', 'leaky_relu', 'sigmoid', 'tanh', 'none')
            dropout: Dropout probability (0 = no dropout)
            pool: If not None, applies pooling with the given kernel size
            pool_type: Type of pooling ('max' or 'avg')
        """
        super(ConvBlock, self).__init__()
        
        # Set default padding to maintain spatial dimensions if not specified
        if padding is None:
            padding = kernel_size // 2
        
        # Convolutional layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        
        # Batch normalization
        self.batch_norm = None
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)
        
        # Activation function
        self.activation = None
        if activation.lower() == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation.lower() == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        
        # Dropout
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        
        # Pooling
        self.pool = None
        if pool is not None:
            if pool_type.lower() == 'max':
                self.pool = nn.MaxPool2d(
                    kernel_size=pool if isinstance(pool, (tuple, list)) else (pool, pool)
                )
            elif pool_type.lower() == 'avg':
                self.pool = nn.AvgPool2d(
                    kernel_size=pool if isinstance(pool, (tuple, list)) else (pool, pool)
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x)
        
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        
        if self.activation is not None:
            x = self.activation(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        if self.pool is not None:
            x = self.pool(x)
        
        return x


class CNNBuilder(nn.Module):
    """
    A flexible CNN builder that creates a sequence of convolutional blocks.
    """
    def __init__(
        self,
        input_shape: Tuple[int, int, int],  # (channels, height, width)
        architecture: List[Dict[str, Any]],
        num_classes: int = 10,
        global_pool: str = 'avg',  # 'avg', 'max', or 'flatten'
        dropout_fc: float = 0.5,
        use_batch_norm_fc: bool = True
    ):
        """
        Initialize the CNN builder.
        
        Args:
            input_shape: Input shape (channels, height, width)
            architecture: List of layer configurations
            num_classes: Number of output classes
            global_pool: Global pooling type ('avg', 'max', or 'flatten')
            dropout_fc: Dropout probability for fully connected layers
            use_batch_norm_fc: Whether to use batch norm in fully connected layers
        """
        super(CNNBuilder, self).__init__()
        
        self.input_shape = input_shape
        self.architecture = architecture
        self.global_pool = global_pool.lower()
        
        # Build the feature extractor
        layers = []
        in_channels = input_shape[0]
        
        for layer_config in architecture:
            if layer_config.get('type', 'conv') == 'conv':
                # Add a convolutional block
                layer_config['in_channels'] = in_channels
                layer = ConvBlock(**layer_config)
                layers.append(layer)
                in_channels = layer_config['out_channels']
            elif layer_config['type'] == 'flatten':
                # Add a flatten layer
                layers.append(nn.Flatten())
            elif layer_config['type'] == 'linear':
                # Add a fully connected layer
                layers.append(nn.Linear(
                    in_features=layer_config['in_features'],
                    out_features=layer_config['out_features']
                ))
                if use_batch_norm_fc and layer_config.get('batch_norm', True):
                    layers.append(nn.BatchNorm1d(layer_config['out_features']))
                if layer_config.get('activation', 'relu') == 'relu':
                    layers.append(nn.ReLU(inplace=True))
                if layer_config.get('dropout', 0) > 0:
                    layers.append(nn.Dropout(p=layer_config['dropout']))
        
        self.features = nn.Sequential(*layers)
        
        # Calculate the size of the feature maps after the convolutional layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.features(dummy_input)
            if isinstance(dummy_output, tuple):
                dummy_output = dummy_output[0]
            self.feature_size = dummy_output.numel() // dummy_output.size(0)
        
        # Global pooling
        if self.global_pool in ['avg', 'max']:
            self.pool = nn.AdaptiveAvgPool2d(1) if self.global_pool == 'avg' else nn.AdaptiveMaxPool2d(1)
            self.feature_size = in_channels
        
        # Fully connected layers
        self.classifier = nn.Sequential()
        
        if dropout_fc > 0:
            self.classifier.add_module('dropout', nn.Dropout(p=dropout_fc))
        
        self.classifier.add_module('fc', nn.Linear(self.feature_size, num_classes))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.features(x)
        
        if self.global_pool in ['avg', 'max']:
            x = self.pool(x)
            x = x.view(x.size(0), -1)
        
        x = self.classifier(x)
        return x


def create_cnn(
    model_name: str,
    input_shape: Tuple[int, int, int],
    num_classes: int = 10,
    pretrained: bool = False,
    **kwargs
) -> nn.Module:
    """
    Create a CNN model by name.
    
    Args:
        model_name: Name of the model ('simple_cnn', 'vgg_like', 'resnet_like')
        input_shape: Input shape (channels, height, width)
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights (not implemented yet)
        **kwargs: Additional arguments for the model
        
    Returns:
        A PyTorch model
    """
    if model_name == 'simple_cnn':
        architecture = [
            # Block 1
            {'out_channels': 32, 'kernel_size': 3, 'batch_norm': True, 'activation': 'relu', 'pool': 2, 'dropout': 0.2},
            # Block 2
            {'out_channels': 64, 'kernel_size': 3, 'batch_norm': True, 'activation': 'relu', 'pool': 2, 'dropout': 0.3},
            # Block 3
            {'out_channels': 128, 'kernel_size': 3, 'batch_norm': True, 'activation': 'relu', 'pool': 2, 'dropout': 0.4},
            # Flatten
            {'type': 'flatten'}
        ]
        return CNNBuilder(input_shape, architecture, num_classes=num_classes, **kwargs)
    
    elif model_name == 'vgg_like':
        architecture = [
            # Block 1
            {'out_channels': 64, 'kernel_size': 3, 'batch_norm': True, 'activation': 'relu'},
            {'out_channels': 64, 'kernel_size': 3, 'batch_norm': True, 'activation': 'relu', 'pool': 2, 'dropout': 0.2},
            # Block 2
            {'out_channels': 128, 'kernel_size': 3, 'batch_norm': True, 'activation': 'relu'},
            {'out_channels': 128, 'kernel_size': 3, 'batch_norm': True, 'activation': 'relu', 'pool': 2, 'dropout': 0.3},
            # Block 3
            {'out_channels': 256, 'kernel_size': 3, 'batch_norm': True, 'activation': 'relu'},
            {'out_channels': 256, 'kernel_size': 3, 'batch_norm': True, 'activation': 'relu', 'pool': 2, 'dropout': 0.4},
            # Flatten
            {'type': 'flatten'}
        ]
        return CNNBuilder(input_shape, architecture, num_classes=num_classes, **kwargs)
    
    elif model_name == 'resnet_like':
        # This is a simplified ResNet-like architecture
        architecture = [
            # Initial conv
            {'out_channels': 64, 'kernel_size': 7, 'stride': 2, 'padding': 3, 'batch_norm': True, 'activation': 'relu', 'pool': 3, 'pool_type': 'max'},
            
            # Residual blocks would be added here in a full implementation
            
            # Global average pooling
            {'type': 'flatten'}
        ]
        return CNNBuilder(input_shape, architecture, num_classes=num_classes, **kwargs)
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")
