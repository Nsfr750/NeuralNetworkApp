"""
Tests for the NetworkBuilder class.
"""

import unittest
import torch
import numpy as np
from src.network_builder import NetworkBuilder, LayerType


class TestNetworkBuilder(unittest.TestCase):
    """Test cases for the NetworkBuilder class."""

    def setUp(self):
        """Set up test fixtures."""
        self.input_shape = (3, 32, 32)
        self.batch_size = 2
        self.dummy_input = torch.randn(self.batch_size, *self.input_shape)

    def test_conv2d_layer(self):
        """Test adding a Conv2D layer."""
        builder = NetworkBuilder(self.input_shape)
        builder.add_layer(
            LayerType.CONV2D,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Test layer properties
        self.assertEqual(len(builder.layers), 1)
        layer = builder.layers[0]
        self.assertEqual(layer.layer_type, LayerType.CONV2D)
        self.assertEqual(layer.params['out_channels'], 16)
        self.assertEqual(layer.output_shape, (16, 32, 32))  # Same spatial dims due to padding
        
        # Test building the model
        model = builder.build()
        output = model(self.dummy_input)
        self.assertEqual(output.shape, (self.batch_size, 16, 32, 32))

    def test_linear_layer(self):
        """Test adding a Linear layer."""
        # First add a flatten layer
        builder = NetworkBuilder(self.input_shape)
        builder.add_layer(LayerType.FLATTEN)
        builder.add_layer(LayerType.LINEAR, out_features=128)
        
        # Test layer properties
        self.assertEqual(len(builder.layers), 2)
        linear_layer = builder.layers[1]
        self.assertEqual(linear_layer.layer_type, LayerType.LINEAR)
        self.assertEqual(linear_layer.params['out_features'], 128)
        
        # Test building the model
        model = builder.build()
        output = model(self.dummy_input)
        self.assertEqual(output.shape, (self.batch_size, 128))

    def test_batch_norm(self):
        """Test adding a BatchNorm2d layer."""
        builder = NetworkBuilder(self.input_shape)
        builder.add_layer(
            LayerType.CONV2D,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        builder.add_layer(LayerType.BATCHNORM2D)
        
        # Test layer properties
        self.assertEqual(len(builder.layers), 2)
        self.assertEqual(builder.layers[1].layer_type, LayerType.BATCHNORM2D)
        self.assertEqual(builder.layers[1].output_shape, (16, 32, 32))
        
        # Test building the model
        model = builder.build()
        output = model(self.dummy_input)
        self.assertEqual(output.shape, (self.batch_size, 16, 32, 32))

    def test_activation_functions(self):
        """Test adding activation functions."""
        activations = [
            (LayerType.RELU, torch.nn.ReLU),
            (LayerType.LEAKYRELU, torch.nn.LeakyReLU),
            (LayerType.SIGMOID, torch.nn.Sigmoid),
            (LayerType.TANH, torch.nn.Tanh),
            (LayerType.SOFTMAX, torch.nn.Softmax)
        ]
        
        for act_type, act_class in activations:
            with self.subTest(activation=act_type):
                builder = NetworkBuilder(self.input_shape)
                builder.add_layer(act_type)
                
                # Test layer properties
                self.assertEqual(len(builder.layers), 1)
                self.assertEqual(builder.layers[0].layer_type, act_type)
                self.assertEqual(builder.layers[0].output_shape, self.input_shape)
                
                # Test building the model
                model = builder.build()
                self.assertIsInstance(model[0], act_class)
                output = model(self.dummy_input)
                self.assertEqual(output.shape, (self.batch_size, *self.input_shape))

    def test_pooling_layers(self):
        """Test adding pooling layers."""
        pool_types = [
            (LayerType.MAXPOOL2D, torch.nn.MaxPool2d),
            (LayerType.AVGPOOL2D, torch.nn.AvgPool2d),
        ]
        
        for pool_type, pool_class in pool_types:
            with self.subTest(pool_type=pool_type):
                builder = NetworkBuilder(self.input_shape)
                builder.add_layer(
                    pool_type,
                    kernel_size=2,
                    stride=2
                )
                
                # Test layer properties
                self.assertEqual(len(builder.layers), 1)
                self.assertEqual(builder.layers[0].layer_type, pool_type)
                self.assertEqual(builder.layers[0].output_shape, (3, 16, 16))
                
                # Test building the model
                model = builder.build()
                self.assertIsInstance(model[0], pool_class)
                output = model(self.dummy_input)
                self.assertEqual(output.shape, (self.batch_size, 3, 16, 16))

    def test_adaptive_pooling(self):
        """Test adding adaptive pooling layers."""
        builder = NetworkBuilder(self.input_shape)
        builder.add_layer(
            LayerType.ADAPTIVEAVGPOOL2D,
            output_size=(8, 8)
        )
        
        # Test layer properties
        self.assertEqual(len(builder.layers), 1)
        self.assertEqual(builder.layers[0].layer_type, LayerType.ADAPTIVEAVGPOOL2D)
        self.assertEqual(builder.layers[0].output_shape, (3, 8, 8))
        
        # Test building the model
        model = builder.build()
        output = model(self.dummy_input)
        self.assertEqual(output.shape, (self.batch_size, 3, 8, 8))

    def test_residual_block(self):
        """Test adding a residual block."""
        builder = NetworkBuilder((64, 32, 32))  # 64 channels input
        
        # Add a residual block with two conv layers
        builder.add_layer(
            LayerType.RESIDUAL,
            layers=[
                {
                    'layer_type': 'CONV2D',
                    'params': {
                        'out_channels': 64,
                        'kernel_size': 3,
                        'padding': 1
                    }
                },
                {
                    'layer_type': 'RELU',
                    'params': {}
                },
                {
                    'layer_type': 'CONV2D',
                    'params': {
                        'out_channels': 64,
                        'kernel_size': 3,
                        'padding': 1
                    }
                }
            ]
        )
        
        # Test layer properties
        self.assertEqual(len(builder.layers), 1)
        self.assertEqual(builder.layers[0].layer_type, LayerType.RESIDUAL)
        self.assertEqual(builder.layers[0].output_shape, (64, 32, 32))
        
        # Test building the model
        model = builder.build()
        dummy_input = torch.randn(2, 64, 32, 32)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, 64, 32, 32))

    def test_serialization(self):
        """Test serializing and deserializing a model."""
        # Build a simple model
        builder = NetworkBuilder(self.input_shape)
        builder.add_layer(
            LayerType.CONV2D,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        builder.add_layer(LayerType.RELU)
        builder.add_layer(LayerType.MAXPOOL2D, kernel_size=2, stride=2)
        builder.add_layer(LayerType.FLATTEN)
        builder.add_layer(LayerType.LINEAR, out_features=10)
        
        # Serialize to JSON
        json_str = builder.to_json()
        
        # Deserialize
        new_builder = NetworkBuilder.from_json(json_str)
        
        # Check if the deserialized builder is the same
        self.assertEqual(len(builder.layers), len(new_builder.layers))
        for layer1, layer2 in zip(builder.layers, new_builder.layers):
            self.assertEqual(layer1.layer_type, layer2.layer_type)
            self.assertEqual(layer1.params, layer2.params)
            self.assertEqual(layer1.input_shape, layer2.input_shape)
            self.assertEqual(layer1.output_shape, layer2.output_shape)
        
        # Check if the models produce the same output
        model1 = builder.build()
        model2 = new_builder.build()
        
        # Set the same random seed for both models
        torch.manual_seed(42)
        output1 = model1(self.dummy_input)
        torch.manual_seed(42)
        output2 = model2(self.dummy_input)
        
        self.assertTrue(torch.allclose(output1, output2))

    def test_model_summary(self):
        """Test the model summary method."""
        import io
        import sys
        
        # Redirect stdout to capture the summary
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        # Build a simple model
        builder = NetworkBuilder(self.input_shape)
        builder.add_layer(
            LayerType.CONV2D,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        builder.add_layer(LayerType.RELU)
        builder.add_layer(LayerType.FLATTEN)
        builder.add_layer(LayerType.LINEAR, out_features=10)
        
        # Generate summary
        builder.summary()
        
        # Get the output
        sys.stdout = sys.__stdout__
        summary = captured_output.getvalue()
        
        # Check if the summary contains expected information
        self.assertIn("Network Summary", summary)
        self.assertIn("Input", summary)
        self.assertIn("conv2d_0", summary)
        self.assertIn("relu_1", summary)
        self.assertIn("flatten_2", summary)
        self.assertIn("linear_3", summary)
        self.assertIn("Total params", summary)


if __name__ == '__main__':
    unittest.main()
