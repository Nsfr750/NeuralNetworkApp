"""
Tests for model architectures and network builder functionality.
"""

import unittest
import torch
import numpy as np
from neuralnetworkapp.builder import NetworkBuilder, LayerType

class TestNetworkArchitectures(unittest.TestCase):
    """Test cases for network architectures."""
    
    def test_simple_cnn_creation(self):
        """Test creating a simple CNN architecture."""
        builder = NetworkBuilder(input_shape=(3, 32, 32))
        builder.add_layer(LayerType.CONV2D, out_channels=32, kernel_size=3, padding=1)\
               .add_layer(LayerType.RELU)\
               .add_layer(LayerType.MAXPOOL2D, kernel_size=2, stride=2)\
               .add_layer(LayerType.FLATTEN)\
               .add_layer(LayerType.LINEAR, out_features=10)
        
        model = builder.build()
        
        # Test model forward pass
        x = torch.randn(1, 3, 32, 32)
        output = model(x)
        self.assertEqual(output.shape, (1, 10))
    
    def test_residual_connection(self):
        """Test creating a model with residual connections."""
        builder = NetworkBuilder(input_shape=(3, 32, 32))
        
        # Add initial layers
        builder.add_layer(LayerType.CONV2D, out_channels=32, kernel_size=3, padding=1)
        
        # Add a residual block
        builder.add_layer(LayerType.RESIDUAL, layers=[
            {'layer_type': 'CONV2D', 'params': {'out_channels': 32, 'kernel_size': 3, 'padding': 1}},
            {'layer_type': 'RELU'},
            {'layer_type': 'CONV2D', 'params': {'out_channels': 32, 'kernel_size': 3, 'padding': 1}}
        ])
        
        # Add final layers
        builder.add_layer(LayerType.FLATTEN)
        builder.add_layer(LayerType.LINEAR, out_features=10)
        
        model = builder.build()
        
        # Test model forward pass
        x = torch.randn(1, 3, 32, 32)
        output = model(x)
        self.assertEqual(output.shape, (1, 10))
    
    def test_model_summary(self):
        """Test the model summary functionality."""
        builder = NetworkBuilder(input_shape=(3, 32, 32))
        builder.add_layer(LayerType.CONV2D, out_channels=32, kernel_size=3, padding=1)\
               .add_layer(LayerType.RELU)\
               .add_layer(LayerType.FLATTEN)\
               .add_layer(LayerType.LINEAR, out_features=10)
        
        # Just verify summary doesn't raise an exception
        builder.summary()
    
    def test_export_import(self):
        """Test exporting and importing model configuration."""
        # Create a network
        builder = NetworkBuilder(input_shape=(3, 32, 32))
        builder.add_layer(LayerType.CONV2D, out_channels=32, kernel_size=3, padding=1)\
               .add_layer(LayerType.RELU)\
               .add_layer(LayerType.FLATTEN)\
               .add_layer(LayerType.LINEAR, out_features=10)
        
        # Export to JSON
        json_str = builder.to_json()
        
        # Import from JSON
        from neuralnetworkapp.builder import NetworkBuilder as NB
        new_builder = NB.from_json(json_str)
        
        # Verify the imported model works
        model = new_builder.build()
        x = torch.randn(1, 3, 32, 32)
        output = model(x)
        self.assertEqual(output.shape, (1, 10))


if __name__ == "__main__":
    unittest.main()
