"""
Tests for CNN models module.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np

# Import the modules to test
from neuralnetworkapp.models.cnn_modelsimport ConvBlock, CNNBuilder, create_cnn


class TestCNNModels(unittest.TestCase):
    """Test cases for CNN models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 4
        self.input_channels = 3
        self.image_size = 32
        self.num_classes = 10
        self.dummy_input = torch.randn(
            self.batch_size, 
            self.input_channels, 
            self.image_size, 
            self.image_size
        ).to(self.device)
    
    def test_conv_block_forward_pass(self):
        """Test forward pass through ConvBlock."""
        # Test with batch norm and ReLU
        conv_block = ConvBlock(
            in_channels=self.input_channels,
            out_channels=32,
            kernel_size=3,
            batch_norm=True,
            activation='relu',
            dropout=0.2,
            pool=2
        ).to(self.device)
        
        output = conv_block(self.dummy_input)
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], 32)  # out_channels
        self.assertEqual(output.shape[2], self.image_size // 2)  # After pooling
        self.assertEqual(output.shape[3], self.image_size // 2)  # After pooling
        
        # Test without batch norm and with max pooling
        conv_block = ConvBlock(
            in_channels=self.input_channels,
            out_channels=64,
            kernel_size=5,
            batch_norm=False,
            activation='leaky_relu',
            pool=2,
            pool_type='max'
        ).to(self.device)
        
        output = conv_block(self.dummy_input)
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], 64)
    
    def test_cnn_builder_forward_pass(self):
        """Test forward pass through CNNBuilder."""
        architecture = [
            # Block 1
            {
                'out_channels': 32,
                'kernel_size': 3,
                'batch_norm': True,
                'activation': 'relu',
                'pool': 2,
                'dropout': 0.2
            },
            # Block 2
            {
                'out_channels': 64,
                'kernel_size': 3,
                'batch_norm': True,
                'activation': 'relu',
                'pool': 2,
                'dropout': 0.3
            },
            # Flatten
            {'type': 'flatten'}
        ]
        
        cnn = CNNBuilder(
            input_shape=(self.input_channels, self.image_size, self.image_size),
            architecture=architecture,
            num_classes=self.num_classes,
            global_pool='avg',
            dropout_fc=0.5
        ).to(self.device)
        
        # Test forward pass
        output = cnn(self.dummy_input)
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], self.num_classes)
    
    def test_create_cnn(self):
        """Test create_cnn function with different architectures."""
        # Test simple_cnn
        model = create_cnn(
            model_name='simple_cnn',
            input_shape=(self.input_channels, self.image_size, self.image_size),
            num_classes=self.num_classes
        ).to(self.device)
        
        output = model(self.dummy_input)
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], self.num_classes)
        
        # Test vgg_like
        model = create_cnn(
            model_name='vgg_like',
            input_shape=(self.input_channels, self.image_size, self.image_size),
            num_classes=self.num_classes
        ).to(self.device)
        
        output = model(self.dummy_input)
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], self.num_classes)
    
    def test_model_parameters(self):
        """Test that model parameters are being updated during training."""
        model = create_cnn(
            model_name='simple_cnn',
            input_shape=(self.input_channels, self.image_size, self.image_size),
            num_classes=self.num_classes
        ).to(self.device)
        
        # Get initial parameters
        initial_params = [p.detach().clone() for p in model.parameters() if p.requires_grad]
        
        # Create a dummy loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Forward pass
        output = model(self.dummy_input)
        target = torch.randint(0, self.num_classes, (self.batch_size,)).to(self.device)
        loss = criterion(output, target)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check if parameters have been updated
        params_have_changed = False
        for init_param, param in zip(initial_params, model.parameters()):
            if not torch.equal(init_param, param):
                params_have_changed = True
                break
        
        self.assertTrue(params_have_changed, "Model parameters were not updated during training")


if __name__ == '__main__':
    unittest.main()
