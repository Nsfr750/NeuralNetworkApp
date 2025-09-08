"""
Tests for transfer learning and model export functionality.
"""

import os
import unittest
import tempfile
import torch
import torch.nn as nn

# Import the modules to test
from neuralnetworkapp.transferimport (
    get_pretrained_model,
    fine_tune_model,
    export_to_onnx,
    export_to_tflite
)


class TestTransferLearning(unittest.TestCase):
    """Test cases for transfer learning and model export."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 4
        self.input_size = (3, 224, 224)
        self.num_classes = 10
        
        # Create a dummy dataset
        self.dummy_input = torch.randn(
            self.batch_size, 
            *self.input_size
        ).to(self.device)
        
        self.dummy_target = torch.randint(
            0, self.num_classes, (self.batch_size,)
        ).to(self.device)
        
        # Create a simple DataLoader
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, x, y):
                self.x = x
                self.y = y
                
            def __len__(self):
                return len(self.x)
                
            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]
        
        dataset = DummyDataset(
            self.dummy_input.cpu(),
            self.dummy_target.cpu()
        )
        
        self.dummy_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_get_pretrained_model(self):
        """Test loading pre-trained models."""
        # Test with different model architectures
        model_names = ['resnet18', 'mobilenet_v2', 'efficientnet_b0']
        
        for model_name in model_names:
            with self.subTest(model_name=model_name):
                # Test with frozen features
                model = get_pretrained_model(
                    model_name=model_name,
                    num_classes=self.num_classes,
                    pretrained=False,  # Don't download weights for testing
                    freeze_features=True
                ).to(self.device)
                
                # Check model output shape
                output = model(self.dummy_input)
                self.assertEqual(output.shape, (self.batch_size, self.num_classes))
                
                # Check that feature extractor is frozen
                for name, param in model.named_parameters():
                    if 'fc' not in name and 'classifier' not in name:
                        self.assertFalse(param.requires_grad, 
                                      f"Parameter {name} should be frozen")
                
                # Check that classifier is trainable
                classifier_params = []
                if hasattr(model, 'fc'):
                    classifier_params.extend(model.fc.parameters())
                if hasattr(model, 'classifier'):
                    if isinstance(model.classifier, nn.Sequential):
                        classifier_params.extend(model.classifier.parameters())
                    else:
                        classifier_params.extend([model.classifier])
                
                for param in classifier_params:
                    self.assertTrue(param.requires_grad, 
                                  "Classifier parameters should be trainable")
    
    def test_fine_tune_model(self):
        """Test fine-tuning a pre-trained model."""
        # Create a simple model for testing
        model = get_pretrained_model(
            model_name='resnet18',
            num_classes=self.num_classes,
            pretrained=False,  # Don't download weights for testing
            freeze_features=True
        ).to(self.device)
        
        # Get initial parameters
        initial_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                initial_params[name] = param.detach().clone()
        
        # Fine-tune the model
        history = fine_tune_model(
            model=model,
            train_loader=self.dummy_loader,
            val_loader=self.dummy_loader,
            num_epochs=2,
            device=self.device,
            freeze_epochs=1,
            unfreeze_layers=['layer4', 'fc']
        )
        
        # Check that training history was recorded
        self.assertIn('train_loss', history)
        self.assertIn('val_loss', history)
        self.assertEqual(len(history['train_loss']), 2)
        
        # Check that parameters were updated
        params_updated = False
        for name, param in model.named_parameters():
            if param.requires_grad and name in initial_params:
                if not torch.equal(param, initial_params[name]):
                    params_updated = True
                    break
        
        self.assertTrue(params_updated, "Model parameters were not updated during training")
    
    def test_export_to_onnx(self):
        """Test exporting a model to ONNX format."""
        # Create a simple model
        model = get_pretrained_model(
            model_name='resnet18',
            num_classes=self.num_classes,
            pretrained=False
        ).to(self.device).eval()
        
        # Export to ONNX
        onnx_path = os.path.join(self.output_dir, 'test_model.onnx')
        export_to_onnx(
            model=model,
            output_path=onnx_path,
            input_size=self.input_size,
            input_names=['input'],
            output_names=['output']
        )
        
        # Check that the file was created
        self.assertTrue(os.path.exists(onnx_path), "ONNX file was not created")
        
        # Verify the ONNX model can be loaded (if onnx is available)
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
        except ImportError:
            print("ONNX not installed, skipping model verification")
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_export_to_tflite(self):
        """Test exporting a model to TensorFlow Lite format."""
        # This test is marked as expected to fail if TensorFlow is not installed
        try:
            import tensorflow as tf
            from onnx_tf.backend import prepare
        except ImportError:
            self.skipTest("TensorFlow and ONNX-TF are required for TFLite export tests")
        
        # Create a simple model
        model = get_pretrained_model(
            model_name='resnet18',
            num_classes=self.num_classes,
            pretrained=False
        ).to(self.device).eval()
        
        # Export to TFLite (quantized)
        tflite_path = os.path.join(self.output_dir, 'test_model.tflite')
        export_to_tflite(
            model=model,
            output_path=tflite_path,
            input_size=self.input_size,
            quantize=True
        )
        
        # Check that the file was created
        self.assertTrue(os.path.exists(tflite_path), "TFLite file was not created")
        
        # Verify the TFLite model can be loaded
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Check input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        self.assertEqual(len(input_details), 1, "Should have 1 input")
        self.assertEqual(len(output_details), 1, "Should have 1 output")
        self.assertEqual(
            tuple(input_details[0]['shape']), 
            (1, *self.input_size),
            "Input shape mismatch"
        )
        self.assertEqual(
            tuple(output_details[0]['shape']), 
            (1, self.num_classes),
            "Output shape mismatch"
        )


if __name__ == '__main__':
    unittest.main()
