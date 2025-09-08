import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple, Union, Type, Callable
from enum import Enum
import math

class ModelType(Enum):
    """Supported model types."""
    MLP = 'mlp'
    CNN = 'cnn'
    RNN = 'rnn'
    LSTM = 'lstm'
    GRU = 'gru'

class ActivationType(Enum):
    """Supported activation functions."""
    RELU = 'relu'
    LEAKY_RELU = 'leaky_relu'
    ELU = 'elu'
    SELU = 'selu'
    GELU = 'gelu'
    SIGMOID = 'sigmoid'
    TANH = 'tanh'
    SWISH = 'swish'
    MISH = 'mish'
    NONE = 'none'

def get_activation(activation: Union[str, ActivationType]) -> nn.Module:
    """Get activation function by name or enum."""
    if isinstance(activation, str):
        activation = ActivationType(activation.lower())
    
    activations = {
        ActivationType.RELU: nn.ReLU(),
        ActivationType.LEAKY_RELU: nn.LeakyReLU(),
        ActivationType.ELU: nn.ELU(),
        ActivationType.SELU: nn.SELU(),
        ActivationType.GELU: nn.GELU(),
        ActivationType.SIGMOID: nn.Sigmoid(),
        ActivationType.TANH: nn.Tanh(),
        ActivationType.SWISH: nn.SiLU(),
        ActivationType.MISH: nn.Mish(),
        ActivationType.NONE: nn.Identity()
    }
    return activations.get(activation, nn.ReLU())

class MLP(nn.Module):
    """
    A configurable Multi-Layer Perceptron (MLP) network.
    
    Args:
        input_size: Number of input features
        hidden_sizes: List of hidden layer sizes
        output_size: Number of output classes
        activation: Activation function to use (default: 'relu')
        dropout: Dropout probability (None to disable)
        batch_norm: Whether to use batch normalization
        use_skip: Whether to use skip connections
    """
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation: Union[str, ActivationType] = ActivationType.RELU,
        dropout: Optional[float] = None,
        batch_norm: bool = False,
        use_skip: bool = False
    ):
        super(MLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_name = activation.value if isinstance(activation, ActivationType) else activation
        self.dropout_p = dropout
        self.batch_norm = batch_norm
        self.use_skip = use_skip
        
        # Set activation function
        self.activation = get_activation(activation)
        
        # Create layers
        self.layers = nn.ModuleList()
        
        # Input layer
        prev_size = input_size
        
        # Hidden layers
        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            self.layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization if enabled
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_size))
                
            # Activation
            self.layers.append(self.activation)
            
            # Dropout if enabled
            if dropout is not None and dropout > 0:
                self.layers.append(nn.Dropout(dropout))
                
            # Skip connection requires matching dimensions
            if use_skip and i > 0 and hidden_sizes[i-1] == hidden_size:
                self.layers.append(nn.Identity())  # Placeholder for skip connection
                
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Flatten input if it's not already flat
        x_flat = x.view(x.size(0), -1)
        x = x_flat
        
        # Store previous layer output for skip connections
        prev_output = None
        
        # Pass through hidden layers
        for i, layer in enumerate(self.layers):
            # Handle skip connections
            if self.use_skip and isinstance(layer, nn.Identity) and prev_output is not None:
                x = x + prev_output  # Skip connection
                prev_output = x
                continue
                
            x = layer(x)
            
            # Store output for potential skip connection
            if isinstance(layer, nn.Linear):
                prev_output = x
        
        # Output layer (no activation)
        x = self.output_layer(x)
        return x
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'model_type': ModelType.MLP.value,
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'activation': self.activation_name,
            'dropout': self.dropout_p,
            'batch_norm': self.batch_norm,
            'use_skip': self.use_skip
        }

# For backward compatibility
NeuralNetwork = MLP

@classmethod
def from_config(cls, config: Dict[str, Any]) -> 'MLP':
    """Create model from configuration."""
    return cls(**config)


class CNN(nn.Module):
    """
    A configurable Convolutional Neural Network (CNN).
    
    Args:
        input_shape: Input shape (channels, height, width)
        conv_layers: List of tuples (out_channels, kernel_size, stride, padding, pool_size)
        dense_sizes: List of dense layer sizes
        output_size: Number of output classes
        activation: Activation function to use (default: 'relu')
        dropout: Dropout probability (None to disable)
        batch_norm: Whether to use batch normalization
        global_pool: Whether to use global average pooling before dense layers
    """
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        conv_layers: List[Tuple[int, int, int, int, int]],
        dense_sizes: List[int],
        output_size: int,
        activation: Union[str, ActivationType] = ActivationType.RELU,
        dropout: Optional[float] = None,
        batch_norm: bool = False,
        global_pool: bool = True
    ):
        super(CNN, self).__init__()
        
        self.input_shape = input_shape
        self.conv_layers = conv_layers
        self.dense_sizes = dense_sizes
        self.output_size = output_size
        self.activation_name = activation.value if isinstance(activation, ActivationType) else activation
        self.dropout_p = dropout
        self.batch_norm = batch_norm
        self.global_pool = global_pool
        
        # Set activation function
        self.activation = get_activation(activation)
        
        # Create convolutional layers
        self.conv_blocks = nn.ModuleList()
        in_channels = input_shape[0]
        
        for out_channels, kernel_size, stride, padding, pool_size in conv_layers:
            # Convolutional block
            block = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            ]
            
            # Batch normalization
            if batch_norm:
                block.append(nn.BatchNorm2d(out_channels))
            
            # Activation
            block.append(self.activation)
            
            # Max pooling if specified
            if pool_size > 1:
                block.append(nn.MaxPool2d(pool_size))
            
            # Dropout if enabled
            if dropout is not None and dropout > 0:
                block.append(nn.Dropout2d(dropout))
            
            self.conv_blocks.append(nn.Sequential(*block))
            in_channels = out_channels
        
        # Calculate the size of the flattened features after conv layers
        self._init_dense_layers()
    
    def _init_dense_layers(self):
        """Initialize dense layers based on the output shape of conv layers."""
        # Forward pass to get the output shape
        with torch.no_grad():
            x = torch.zeros(1, *self.input_shape)
            for block in self.conv_blocks:
                x = block(x)
            flattened_size = x.view(1, -1).shape[1]
        
        # Create dense layers
        self.dense_layers = nn.ModuleList()
        prev_size = flattened_size
        
        for size in self.dense_sizes:
            self.dense_layers.append(nn.Linear(prev_size, size))
            
            if self.batch_norm:
                self.dense_layers.append(nn.BatchNorm1d(size))
            
            self.dense_layers.append(self.activation)
            
            if self.dropout_p is not None and self.dropout_p > 0:
                self.dense_layers.append(nn.Dropout(self.dropout_p))
            
            prev_size = size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, self.output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Ensure input has the correct shape
        if x.dim() == 3:  # Add channel dimension if missing
            x = x.unsqueeze(1)
        
        # Convolutional layers
        for block in self.conv_blocks:
            x = block(x)
        
        # Global average pooling if enabled
        if self.global_pool and x.dim() > 2:
            x = F.adaptive_avg_pool2d(x, (1, 1))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense layers
        for layer in self.dense_layers:
            x = layer(x)
        
        # Output layer
        x = self.output_layer(x)
        return x
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'model_type': ModelType.CNN.value,
            'input_shape': self.input_shape,
            'conv_layers': self.conv_layers,
            'dense_sizes': self.dense_sizes,
            'output_size': self.output_size,
            'activation': self.activation_name,
            'dropout': self.dropout_p,
            'batch_norm': self.batch_norm,
            'global_pool': self.global_pool
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CNN':
        """Create model from configuration."""
        return cls(**config)


class RNNBase(nn.Module):
    """Base class for RNN, LSTM, and GRU networks."""
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        rnn_type: str = 'rnn',
        bidirectional: bool = False,
        dropout: float = 0.0,
        batch_first: bool = True,
        dense_sizes: Optional[List[int]] = None,
        activation: Union[str, ActivationType] = ActivationType.TANH,
        batch_norm: bool = False
    ):
        super(RNNBase, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.dense_sizes = dense_sizes or []
        self.activation_name = activation.value if isinstance(activation, ActivationType) else activation
        self.batch_norm = batch_norm
        
        # Set activation function
        self.activation = get_activation(activation)
        
        # RNN layer
        rnn_class = {
            'rnn': nn.RNN,
            'lstm': nn.LSTM,
            'gru': nn.GRU
        }.get(self.rnn_type, nn.RNN)
        
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Calculate the size after RNN
        rnn_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Dense layers
        self.dense_layers = nn.ModuleList()
        prev_size = rnn_output_size
        
        for size in self.dense_sizes:
            self.dense_layers.append(nn.Linear(prev_size, size))
            
            if batch_norm:
                self.dense_layers.append(nn.BatchNorm1d(size))
            
            self.dense_layers.append(self.activation)
            
            if dropout > 0:
                self.dense_layers.append(nn.Dropout(dropout))
            
            prev_size = size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # RNN expects (batch, seq_len, input_size) when batch_first=True
        if x.dim() == 2:  # Add sequence length dimension if missing
            x = x.unsqueeze(1)
        
        # RNN layer
        rnn_out, _ = self.rnn(x)
        
        # Use the last time step's output
        if self.batch_first:
            x = rnn_out[:, -1, :]
        else:
            x = rnn_out[-1, :, :]
        
        # Dense layers
        for layer in self.dense_layers:
            x = layer(x)
        
        # Output layer
        x = self.output_layer(x)
        return x
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'model_type': self.rnn_type,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'bidirectional': self.bidirectional,
            'dropout': self.rnn.dropout if self.num_layers > 1 else 0.0,
            'batch_first': self.batch_first,
            'dense_sizes': self.dense_sizes,
            'activation': self.activation_name,
            'batch_norm': self.batch_norm
        }


class LSTMNetwork(RNNBase):
    """LSTM network with configurable architecture."""
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        bidirectional: bool = False,
        dropout: float = 0.0,
        batch_first: bool = True,
        dense_sizes: Optional[List[int]] = None,
        activation: Union[str, ActivationType] = ActivationType.TANH,
        batch_norm: bool = False
    ):
        super(LSTMNetwork, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            rnn_type='lstm',
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=batch_first,
            dense_sizes=dense_sizes,
            activation=activation,
            batch_norm=batch_norm
        )


class GRUNetwork(RNNBase):
    """GRU network with configurable architecture."""
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        bidirectional: bool = False,
        dropout: float = 0.0,
        batch_first: bool = True,
        dense_sizes: Optional[List[int]] = None,
        activation: Union[str, ActivationType] = ActivationType.TANH,
        batch_norm: bool = False
    ):
        super(GRUNetwork, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            rnn_type='gru',
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=batch_first,
            dense_sizes=dense_sizes,
            activation=activation,
            batch_norm=batch_norm
        )


def create_model(
    model_type: Union[str, ModelType] = ModelType.MLP,
    input_size: Optional[Union[int, Tuple[int, ...]]] = None,
    hidden_sizes: Optional[List[int]] = None,
    output_size: Optional[int] = None,
    activation: Union[str, ActivationType] = ActivationType.RELU,
    dropout: Optional[float] = None,
    batch_norm: bool = False,
    **kwargs
) -> nn.Module:
    """
    Helper function to create a neural network model.
    
    Args:
        model_type: Type of model to create ('mlp', 'cnn', 'rnn', 'lstm', 'gru')
        input_size: For MLP: number of input features
                    For CNN: input shape (channels, height, width)
                    For RNN/LSTM/GRU: input feature size
        hidden_sizes: List of hidden layer sizes (for MLP) or dense layer sizes (for CNN/RNN)
        output_size: Number of output classes
        activation: Activation function to use
        dropout: Dropout probability (None to disable)
        batch_norm: Whether to use batch normalization
        **kwargs: Additional model-specific arguments
            
    Returns:
        Configured PyTorch model
        
    Examples:
        # MLP
        model = create_model(
            model_type='mlp',
            input_size=784,
            hidden_sizes=[128, 64],
            output_size=10,
            activation='relu',
            dropout=0.2,
            batch_norm=True
        )
        
        # CNN
        model = create_model(
            model_type='cnn',
            input_shape=(3, 32, 32),
            conv_layers=[
                (32, 3, 1, 1, 2),  # out_channels, kernel_size, stride, padding, pool_size
                (64, 3, 1, 1, 2),
            ],
            dense_sizes=[128],
            output_size=10
        )
        
        # LSTM
        model = create_model(
            model_type='lstm',
            input_size=100,
            hidden_size=128,
            num_layers=2,
            output_size=10,
            bidirectional=True,
            dense_sizes=[64]
        )
    """
    if isinstance(model_type, str):
        model_type = ModelType(model_type.lower())
    
    if hidden_sizes is None:
        hidden_sizes = []
    
    if output_size is None:
        raise ValueError("output_size must be specified")
    
    if model_type == ModelType.MLP:
        if input_size is None:
            raise ValueError("input_size must be specified for MLP")
        return MLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
            **kwargs
        )
    elif model_type == ModelType.CNN:
        if 'input_shape' not in kwargs:
            raise ValueError("input_shape must be specified for CNN")
        # Remove dense_sizes from kwargs if it exists to prevent duplicate argument
        kwargs.pop('dense_sizes', None)
        return CNN(
            input_shape=kwargs.pop('input_shape'),
            conv_layers=kwargs.pop('conv_layers', []),
            dense_sizes=hidden_sizes or [],
            output_size=output_size,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
            **kwargs
        )
    elif model_type in (ModelType.RNN, ModelType.LSTM, ModelType.GRU):
        if input_size is None:
            raise ValueError("input_size must be specified for RNN/LSTM/GRU")
        
        model_class = {
            ModelType.RNN: RNNBase,
            ModelType.LSTM: LSTMNetwork,
            ModelType.GRU: GRUNetwork
        }[model_type]
        
        return model_class(
            input_size=input_size,
            hidden_size=kwargs.pop('hidden_size', 128),
            num_layers=kwargs.pop('num_layers', 1),
            output_size=output_size,
            bidirectional=kwargs.pop('bidirectional', False),
            dropout=dropout or 0.0,
            dense_sizes=hidden_sizes,
            activation=activation,
            batch_norm=batch_norm,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")