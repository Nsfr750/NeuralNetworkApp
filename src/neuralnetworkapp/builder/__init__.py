"""
Neural Network Architecture Builder

This module provides an intuitive interface for building and visualizing
neural network architectures with real-time preview.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import os

# For visualization
from torchviz import make_dot
from graphviz import Digraph
import matplotlib.pyplot as plt
from IPython.display import display, HTML


class LayerType(Enum):
    """Supported layer types for the network builder."""
    CONV2D = auto()
    LINEAR = auto()
    BATCHNORM2D = auto()
    DROPOUT = auto()
    MAXPOOL2D = auto()
    AVGPOOL2D = auto()
    ADAPTIVEAVGPOOL2D = auto()
    FLATTEN = auto()
    RELU = auto()
    LEAKYRELU = auto()
    SIGMOID = auto()
    TANH = auto()
    SOFTMAX = auto()
    IDENTITY = auto()
    RESIDUAL = auto()


@dataclass
class LayerConfig:
    """Configuration for a neural network layer."""
    layer_type: LayerType
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert layer config to a dictionary."""
        return {
            'layer_type': self.layer_type.name,
            'name': self.name,
            'params': self.params,
            'input_shape': list(self.input_shape) if self.input_shape else None,
            'output_shape': list(self.output_shape) if self.output_shape else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LayerConfig':
        """Create a LayerConfig from a dictionary."""
        return cls(
            layer_type=LayerType[data['layer_type']],
            name=data['name'],
            params=data.get('params', {}),
            input_shape=tuple(data['input_shape']) if 'input_shape' in data else None,
            output_shape=tuple(data['output_shape']) if 'output_shape' in data else None
        )


class NetworkBuilder:
    """
    An intuitive interface for building neural network architectures.
    
    Features:
    - Visualize network architecture in real-time
    - Interactive layer addition/removal
    - Automatic shape inference
    - Export to PyTorch model
    - Import/export architecture as JSON
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (3, 224, 224)):
        """
        Initialize the network builder.
        
        Args:
            input_shape: Shape of the input tensor (channels, height, width)
        """
        self.input_shape = input_shape
        self.layers: List[LayerConfig] = []
        self._layer_counter = 0
        self._output_shapes: Dict[str, Tuple[int, ...]] = {}
        self._model: Optional[nn.Module] = None
        
        # For visualization
        self._graph = None
        self._last_visualization = None
    
    def add_layer(self, layer_type: LayerType, **params) -> 'NetworkBuilder':
        """
        Add a layer to the network.
        
        Args:
            layer_type: Type of layer to add
            **params: Parameters for the layer
            
        Returns:
            self for method chaining
        """
        # Generate a unique name for the layer
        layer_name = f"{layer_type.name.lower()}_{self._layer_counter}"
        self._layer_counter += 1
        
        # Create layer config
        layer = LayerConfig(
            layer_type=layer_type,
            name=layer_name,
            params=params
        )
        
        # Calculate input shape (shape after previous layer)
        if not self.layers:
            layer.input_shape = self.input_shape
        else:
            layer.input_shape = self.layers[-1].output_shape
        
        # Calculate output shape
        layer.output_shape = self._calculate_output_shape(layer)
        
        self.layers.append(layer)
        self._output_shapes[layer_name] = layer.output_shape
        
        return self
    
    def _calculate_output_shape(self, layer: LayerConfig) -> Tuple[int, ...]:
        """Calculate the output shape of a layer."""
        if layer.input_shape is None:
            raise ValueError("Input shape must be specified for the first layer")
        
        if layer.layer_type == LayerType.CONV2D:
            # Extract parameters with defaults
            out_channels = layer.params.get('out_channels', 1)
            kernel_size = layer.params.get('kernel_size', 3)
            stride = layer.params.get('stride', 1)
            padding = layer.params.get('padding', 0)
            dilation = layer.params.get('dilation', 1)
            
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            if isinstance(stride, int):
                stride = (stride, stride)
            if isinstance(padding, int):
                padding = (padding, padding)
            if isinstance(dilation, int):
                dilation = (dilation, dilation)
            
            # Calculate output dimensions
            def conv_output_size(input_size, kernel, stride, pad, dil):
                return (input_size + 2 * pad - dil * (kernel - 1) - 1) // stride + 1
            
            h_out = conv_output_size(
                layer.input_shape[1], kernel_size[0], stride[0], padding[0], dilation[0]
            )
            w_out = conv_output_size(
                layer.input_shape[2], kernel_size[1], stride[1], padding[1], dilation[1]
            )
            
            return (out_channels, h_out, w_out)
            
        elif layer.layer_type == LayerType.LINEAR:
            out_features = layer.params.get('out_features', 1)
            return (out_features,)
            
        elif layer.layer_type in (LayerType.BATCHNORM2D, LayerType.DROPOUT):
            # Shape-preserving layers
            return layer.input_shape
            
        elif layer.layer_type == LayerType.MAXPOOL2D:
            kernel_size = layer.params.get('kernel_size', 2)
            stride = layer.params.get('stride', kernel_size)
            padding = layer.params.get('padding', 0)
            dilation = layer.params.get('dilation', 1)
            
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            if isinstance(stride, int):
                stride = (stride, stride)
            if isinstance(padding, int):
                padding = (padding, padding)
            if isinstance(dilation, int):
                dilation = (dilation, dilation)
            
            def pool_output_size(input_size, kernel, stride, pad, dil):
                return (input_size + 2 * pad - dil * (kernel - 1) - 1) // stride + 1
            
            h_out = pool_output_size(
                layer.input_shape[1], kernel_size[0], stride[0], padding[0], dilation[0]
            )
            w_out = pool_output_size(
                layer.input_shape[2], kernel_size[1], stride[1], padding[1], dilation[1]
            )
            
            return (layer.input_shape[0], h_out, w_out)
            
        elif layer.layer_type == LayerType.ADAPTIVEAVGPOOL2D:
            output_size = layer.params.get('output_size', (1, 1))
            if isinstance(output_size, int):
                output_size = (output_size, output_size)
            return (layer.input_shape[0],) + output_size
            
        elif layer.layer_type == LayerType.FLATTEN:
            # Flatten all dimensions except batch
            return (np.prod(layer.input_shape),)
            
        elif layer.layer_type in (LayerType.RELU, LayerType.LEAKYRELU, 
                               LayerType.SIGMOID, LayerType.TANH, 
                               LayerType.SOFTMAX, LayerType.IDENTITY):
            # Shape-preserving activation functions
            return layer.input_shape
            
        elif layer.layer_type == LayerType.RESIDUAL:
            # For residual connections, output shape is the same as input
            return layer.input_shape
            
        else:
            raise ValueError(f"Unsupported layer type: {layer.layer_type}")
    
    def remove_layer(self, layer_name: str) -> 'NetworkBuilder':
        """
        Remove a layer from the network.
        
        Args:
            layer_name: Name of the layer to remove
            
        Returns:
            self for method chaining
        """
        self.layers = [layer for layer in self.layers if layer.name != layer_name]
        self._update_output_shapes()
        return self
    
    def _update_output_shapes(self):
        """Update output shapes after modifying the network."""
        self._output_shapes = {}
        input_shape = self.input_shape
        
        for layer in self.layers:
            layer.input_shape = input_shape
            layer.output_shape = self._calculate_output_shape(layer)
            self._output_shapes[layer.name] = layer.output_shape
            input_shape = layer.output_shape
    
    def build(self) -> nn.Module:
        """
        Build a PyTorch model from the current architecture.
        
        Returns:
            A PyTorch model
        """
        layers = []
        
        for layer in self.layers:
            if layer.layer_type == LayerType.CONV2D:
                layers.append(nn.Conv2d(
                    in_channels=layer.input_shape[0],
                    out_channels=layer.params['out_channels'],
                    kernel_size=layer.params.get('kernel_size', 3),
                    stride=layer.params.get('stride', 1),
                    padding=layer.params.get('padding', 0),
                    dilation=layer.params.get('dilation', 1),
                    groups=layer.params.get('groups', 1),
                    bias=layer.params.get('bias', True)
                ))
                
            elif layer.layer_type == LayerType.LINEAR:
                layers.append(nn.Linear(
                    in_features=layer.input_shape[0],
                    out_features=layer.params['out_features'],
                    bias=layer.params.get('bias', True)
                ))
                
            elif layer.layer_type == LayerType.BATCHNORM2D:
                layers.append(nn.BatchNorm2d(
                    num_features=layer.input_shape[0],
                    eps=layer.params.get('eps', 1e-5),
                    momentum=layer.params.get('momentum', 0.1),
                    affine=layer.params.get('affine', True)
                ))
                
            elif layer.layer_type == LayerType.DROPOUT:
                layers.append(nn.Dropout2d(
                    p=layer.params.get('p', 0.5),
                    inplace=layer.params.get('inplace', False)
                ))
                
            elif layer.layer_type == LayerType.MAXPOOL2D:
                layers.append(nn.MaxPool2d(
                    kernel_size=layer.params.get('kernel_size', 2),
                    stride=layer.params.get('stride', None),
                    padding=layer.params.get('padding', 0),
                    dilation=layer.params.get('dilation', 1),
                    return_indices=layer.params.get('return_indices', False),
                    ceil_mode=layer.params.get('ceil_mode', False)
                ))
                
            elif layer.layer_type == LayerType.AVGPOOL2D:
                layers.append(nn.AvgPool2d(
                    kernel_size=layer.params.get('kernel_size', 2),
                    stride=layer.params.get('stride', None),
                    padding=layer.params.get('padding', 0),
                    ceil_mode=layer.params.get('ceil_mode', False),
                    count_include_pad=layer.params.get('count_include_pad', True),
                    divisor_override=layer.params.get('divisor_override', None)
                ))
                
            elif layer.layer_type == LayerType.ADAPTIVEAVGPOOL2D:
                output_size = layer.params.get('output_size', (1, 1))
                layers.append(nn.AdaptiveAvgPool2d(output_size))
                
            elif layer.layer_type == LayerType.FLATTEN:
                layers.append(nn.Flatten(
                    start_dim=layer.params.get('start_dim', 1),
                    end_dim=layer.params.get('end_dim', -1)
                ))
                
            elif layer.layer_type == LayerType.RELU:
                layers.append(nn.ReLU(
                    inplace=layer.params.get('inplace', False)
                ))
                
            elif layer.layer_type == LayerType.LEAKYRELU:
                layers.append(nn.LeakyReLU(
                    negative_slope=layer.params.get('negative_slope', 0.01),
                    inplace=layer.params.get('inplace', False)
                ))
                
            elif layer.layer_type == LayerType.SIGMOID:
                layers.append(nn.Sigmoid())
                
            elif layer.layer_type == LayerType.TANH:
                layers.append(nn.Tanh())
                
            elif layer.layer_type == LayerType.SOFTMAX:
                layers.append(nn.Softmax(
                    dim=layer.params.get('dim', 1)
                ))
                
            elif layer.layer_type == LayerType.IDENTITY:
                layers.append(nn.Identity())
                
            elif layer.layer_type == LayerType.RESIDUAL:
                # For residual connections, we need to implement a custom module
                class ResidualBlock(nn.Module):
                    def __init__(self, sub_layers):
                        super().__init__()
                        self.sub_layers = nn.Sequential(*sub_layers)
                        
                    def forward(self, x):
                        return x + self.sub_layers(x)
                
                # Get the sub-layers for the residual block
                sub_layers = []
                for sub_layer in layer.params.get('layers', []):
                    sub_layers.append(self._build_layer(sub_layer))
                
                layers.append(ResidualBlock(sub_layers))
        
        self._model = nn.Sequential(*layers)
        return self._model
    
    def _build_layer(self, layer_config: Dict) -> nn.Module:
        """Build a single layer from its config."""
        # This is a simplified version of the build method for internal use
        # in building residual blocks
        layer_type = LayerType[layer_config['layer_type']]
        params = layer_config.get('params', {})
        
        if layer_type == LayerType.CONV2D:
            return nn.Conv2d(**params)
        elif layer_type == LayerType.LINEAR:
            return nn.Linear(**params)
        elif layer_type == LayerType.BATCHNORM2D:
            return nn.BatchNorm2d(**params)
        elif layer_type == LayerType.DROPOUT:
            return nn.Dropout2d(**params)
        elif layer_type == LayerType.MAXPOOL2D:
            return nn.MaxPool2d(**params)
        elif layer_type == LayerType.AVGPOOL2D:
            return nn.AvgPool2d(**params)
        elif layer_type == LayerType.ADAPTIVEAVGPOOL2D:
            return nn.AdaptiveAvgPool2d(**params)
        elif layer_type == LayerType.FLATTEN:
            return nn.Flatten(**params)
        elif layer_type == LayerType.RELU:
            return nn.ReLU(**params)
        elif layer_type == LayerType.LEAKYRELU:
            return nn.LeakyReLU(**params)
        elif layer_type == LayerType.SIGMOID:
            return nn.Sigmoid()
        elif layer_type == LayerType.TANH:
            return nn.Tanh()
        elif layer_type == LayerType.SOFTMAX:
            return nn.Softmax(**params)
        elif layer_type == LayerType.IDENTITY:
            return nn.Identity()
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")
    
    def visualize(self, filename: Optional[str] = None, format: str = 'png', 
                  show_shapes: bool = True, show_params: bool = False) -> None:
        """
        Visualize the network architecture.
        
        Args:
            filename: If provided, save the visualization to a file
            format: Image format (png, pdf, svg, etc.)
            show_shapes: If True, show input/output shapes
            show_params: If True, show layer parameters
        """
        if not self.layers:
            print("No layers to visualize")
            return
        
        # Create a new graph
        self._graph = Digraph(
            format=format,
            graph_attr={'rankdir': 'TB'},  # Top to bottom layout
            node_attr={'style': 'filled', 'fillcolor': 'lightblue', 'shape': 'box'}
        )
        
        # Add input node
        input_label = f"Input\n{self.input_shape}" if show_shapes else "Input"
        self._graph.node('input', label=input_label, shape='ellipse', fillcolor='lightgreen')
        
        # Add layers
        prev_node = 'input'
        for i, layer in enumerate(self.layers):
            # Create node label
            label = layer.name
            
            if show_shapes and layer.input_shape is not None:
                label += f"\nIn: {layer.input_shape}"
            if show_shapes and layer.output_shape is not None:
                label += f"\nOut: {layer.output_shape}"
                
            if show_params and layer.params:
                params_str = ", ".join(f"{k}={v}" for k, v in layer.params.items())
                label += f"\n[{params_str}]"
            
            # Add node
            self._graph.node(layer.name, label=label)
            
            # Add edge from previous node
            self._graph.edge(prev_node, layer.name)
            prev_node = layer.name
        
        # Add output node
        output_shape = self.layers[-1].output_shape if self.layers else self.input_shape
        output_label = f"Output\n{output_shape}" if show_shapes and output_shape else "Output"
        self._graph.node('output', label=output_label, shape='ellipse', fillcolor='lightcoral')
        if self.layers:
            self._graph.edge(prev_node, 'output')
        
        # Save or display the graph
        if filename:
            self._graph.render(filename, cleanup=True, format=format)
            print(f"Visualization saved to {filename}.{format}")
        else:
            # In Jupyter notebook, display the graph
            try:
                from IPython.display import display
                display(self._graph)
            except ImportError:
                print("Display requires IPython. To save the visualization, provide a filename.")
    
    def summary(self) -> None:
        """Print a summary of the network architecture."""
        print(f"Network Summary (input shape: {self.input_shape})")
        print("-" * 80)
        print(f"{'Layer (type)':<30} {'Output Shape':<30} {'Param #'}")
        print("=" * 80)
        
        total_params = 0
        
        # Input layer
        print(f"{'Input':<30} {str(self.input_shape):<30} 0")
        
        # Hidden layers
        for layer in self.layers:
            # Calculate number of parameters
            if layer.layer_type == LayerType.CONV2D:
                # (out_channels * in_channels * kernel_size[0] * kernel_size[1]) + out_channels (bias)
                in_channels = layer.input_shape[0]
                out_channels = layer.params['out_channels']
                kernel_size = layer.params.get('kernel_size', 3)
                if isinstance(kernel_size, int):
                    kernel_size = (kernel_size, kernel_size)
                bias = 1 if layer.params.get('bias', True) else 0
                params = (in_channels * out_channels * kernel_size[0] * kernel_size[1]) + (out_channels * bias)
                
            elif layer.layer_type == LayerType.LINEAR:
                # (in_features * out_features) + out_features (bias)
                in_features = layer.input_shape[0]
                out_features = layer.params['out_features']
                bias = 1 if layer.params.get('bias', True) else 0
                params = (in_features * out_features) + (out_features * bias)
                
            elif layer.layer_type == LayerType.BATCHNORM2D:
                # 2 * num_features (weight and bias)
                num_features = layer.input_shape[0]
                params = 2 * num_features
                
            else:
                # Other layers don't have parameters
                params = 0
            
            total_params += params
            
            # Print layer info
            layer_info = f"{layer.name} ({layer.layer_type.name})"
            output_shape = str(layer.output_shape) if layer.output_shape else ""
            print(f"{layer_info:<30} {output_shape:<30} {params:,}")
        
        print("=" * 80)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {total_params:,}")
        print(f"Non-trainable params: 0")
        print("-" * 80)
    
    def to_json(self, filename: Optional[str] = None) -> str:
        """
        Convert the network architecture to JSON.
        
        Args:
            filename: If provided, save the JSON to a file
            
        Returns:
            JSON string representation of the network
        """
        data = {
            'input_shape': self.input_shape,
            'layers': [layer.to_dict() for layer in self.layers]
        }
        
        json_str = json.dumps(data, indent=2)
        
        if filename:
            with open(filename, 'w') as f:
                f.write(json_str)
            print(f"Network configuration saved to {filename}")
        
        return json_str
    
    @classmethod
    def from_json(cls, json_str: str) -> 'NetworkBuilder':
        """
        Create a NetworkBuilder from a JSON string.
        
        Args:
            json_str: JSON string representation of the network
            
        Returns:
            A new NetworkBuilder instance
        """
        data = json.loads(json_str)
        builder = cls(input_shape=tuple(data['input_shape']))
        
        for layer_data in data['layers']:
            builder.layers.append(LayerConfig.from_dict(layer_data))
        
        builder._update_output_shapes()
        return builder
    
    def __str__(self) -> str:
        """Return a string representation of the network."""
        lines = [f"NetworkBuilder(input_shape={self.input_shape})"]
        
        for layer in self.layers:
            params = ", ".join(f"{k}={v}" for k, v in layer.params.items())
            lines.append(f"  {layer.name}: {layer.layer_type.name}({params})")
            
            if layer.input_shape is not None and layer.output_shape is not None:
                lines.append(f"       {' ' * len(layer.name)}  {layer.input_shape} -> {layer.output_shape}")
        
        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    # Create a simple CNN
    builder = NetworkBuilder(input_shape=(3, 32, 32))
    
    builder.add_layer(LayerType.CONV2D, out_channels=32, kernel_size=3, padding=1)\
            .add_layer(LayerType.RELU)\
            .add_layer(LayerType.MAXPOOL2D, kernel_size=2, stride=2)\
            .add_layer(LayerType.CONV2D, out_channels=64, kernel_size=3, padding=1)\
            .add_layer(LayerType.RELU)\
            .add_layer(LayerType.MAXPOOL2D, kernel_size=2, stride=2)\
            .add_layer(LayerType.FLATTEN)\
            .add_layer(LayerType.LINEAR, out_features=128)\
            .add_layer(LayerType.RELU)\
            .add_layer(LayerType.LINEAR, out_features=10)
    
    # Print network summary
    print("\nNetwork Summary:")
    print("-" * 80)
    builder.summary()
    
    # Visualize the network
    print("\nGenerating visualization...")
    builder.visualize("network_architecture", format="png", show_shapes=True)
    
    # Build the PyTorch model
    model = builder.build()
    print("\nPyTorch model built successfully!")
    print(model)
