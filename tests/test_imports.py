#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify package imports and basic functionality.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

def test_imports():
    """Test importing main package components."""
    print("Testing imports...")
    
    try:
        import neuralnetworkapp
        from neuralnetworkapp import (
            Trainer,
            load_tabular_data, create_data_loaders, TabularDataset,
            save_model, load_model, plot_training_history,
            save_config, load_config, count_parameters, set_seed
        )
        from neuralnetworkapp.models import create_model
        print("✅ All imports successful!")
        print(f"NeuralNetworkApp version: {neuralnetworkapp.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """Test creating a simple neural network model."""
    print("\nTesting model creation...")
    try:
        from neuralnetworkapp.models import create_model
        
        # Function to count trainable parameters
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
            
        # Create a simple MLP model
        model = create_model(
            model_type='mlp',
            input_size=784,
            hidden_sizes=[128, 64],
            output_size=10,
            activation='relu',
            dropout=0.2,
            batch_norm=True
        )
        print("✅ Model created successfully!")
        print(f"Number of parameters: {count_parameters(model):,}")
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("NeuralNetworkApp - Package Test")
    print("=" * 50)
    
    # Run tests
    import_success = test_imports()
    model_success = test_model_creation()
    
    print("\nTest Summary:")
    print(f"- Imports: {'✅' if import_success else '❌'}")
    print(f"- Model Creation: {'✅' if model_success else '❌'}")
    
    if import_success and model_success:
        print("\n🎉 All tests passed! The package is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
