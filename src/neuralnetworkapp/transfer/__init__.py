"""
Transfer Learning and Model Export

This module provides utilities for transfer learning with pre-trained models
and exporting models to different formats (ONNX, TensorFlow Lite).
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Available pre-trained models from torchvision
PRETRAINED_MODELS = {
    # Classification models
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
    'vgg11': models.vgg11,
    'vgg13': models.vgg13,
    'vgg16': models.vgg16,
    'vgg19': models.vgg19,
    'vgg11_bn': models.vgg11_bn,
    'vgg13_bn': models.vgg13_bn,
    'vgg16_bn': models.vgg16_bn,
    'vgg19_bn': models.vgg19_bn,
    'densenet121': models.densenet121,
    'densenet169': models.densenet169,
    'densenet201': models.densenet201,
    'mobilenet_v2': models.mobilenet_v2,
    'mobilenet_v3_small': models.mobilenet_v3_small,
    'mobilenet_v3_large': models.mobilenet_v3_large,
    'efficientnet_b0': models.efficientnet_b0,
    'efficientnet_b1': models.efficientnet_b1,
    'efficientnet_b2': models.efficientnet_b2,
    'efficientnet_b3': models.efficientnet_b3,
    'efficientnet_b4': models.efficientnet_b4,
    'efficientnet_b5': models.efficientnet_b5,
    'efficientnet_b6': models.efficientnet_b6,
    'efficientnet_b7': models.efficientnet_b7,
    'regnet_y_400mf': models.regnet_y_400mf,
    'regnet_y_800mf': models.regnet_y_800mf,
    'regnet_y_1_6gf': models.regnet_y_1_6gf,
    'regnet_y_3_2gf': models.regnet_y_3_2gf,
    'regnet_y_8gf': models.regnet_y_8gf,
    'regnet_y_16gf': models.regnet_y_16gf,
    'regnet_x_400mf': models.regnet_x_400mf,
    'regnet_x_800mf': models.regnet_x_800mf,
    'regnet_x_1_6gf': models.regnet_x_1_6gf,
    'regnet_x_3_2gf': models.regnet_x_3_2gf,
    'regnet_x_8gf': models.regnet_x_8gf,
    'regnet_x_16gf': models.regnet_x_16gf,
}


def get_pretrained_model(
    model_name: str,
    num_classes: int = 1000,
    pretrained: bool = True,
    freeze_features: bool = False,
    **kwargs
) -> nn.Module:
    """
    Load a pre-trained model from torchvision models.
    
    Args:
        model_name: Name of the pre-trained model
        num_classes: Number of output classes
        pretrained: If True, returns a model pre-trained on ImageNet
        freeze_features: If True, freeze all layers except the final classifier
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        A PyTorch model with the specified number of output classes
    """
    if model_name not in PRETRAINED_MODELS:
        raise ValueError(f"Model '{model_name}' is not supported. "
                         f"Available models: {list(PRETRAINED_MODELS.keys())}")
    
    # Load pre-trained model
    model_func = PRETRAINED_MODELS[model_name]
    model = model_func(pretrained=pretrained, **kwargs)
    
    # Get the input size for this model
    input_size = 224  # Default for most models
    if hasattr(model, 'input_size'):
        input_size = model.input_size[1]  # type: ignore
    
    # Get the number of features in the last layer
    if hasattr(model, 'classifier') and model.classifier is not None:
        # Models like VGG, DenseNet
        if isinstance(model.classifier, nn.Sequential):
            num_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_features, num_classes)
        else:
            num_features = model.classifier.in_features
            model.classifier = nn.Linear(num_features, num_classes)
    elif hasattr(model, 'fc') and model.fc is not None:
        # Models like ResNet, ResNeXt
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif hasattr(model, 'last_linear') and model.last_linear is not None:
        # Models like MobileNetV2
        num_features = model.last_linear.in_features
        model.last_linear = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"Could not find classifier layer in model {model_name}")
    
    # Freeze feature extraction layers if requested
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze the classifier/fully connected layer
        if hasattr(model, 'classifier') and model.classifier is not None:
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(model, 'fc') and model.fc is not None:
            for param in model.fc.parameters():
                param.requires_grad = True
    
    # Add model metadata
    model.model_name = model_name  # type: ignore
    model.input_size = (3, input_size, input_size)  # type: ignore
    
    return model


def fine_tune_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    num_epochs: int = 10,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    freeze_epochs: int = 0,
    unfreeze_layers: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, List[float]]:
    """
    Fine-tune a pre-trained model.
    
    Args:
        model: Pre-trained model to fine-tune
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        device: Device to train on
        freeze_epochs: Number of epochs to keep feature extractor frozen
        unfreeze_layers: List of layer names to unfreeze after freeze_epochs
        **kwargs: Additional arguments for the Trainer
        
    Returns:
        Training history
    """
    from neuralnetworkapp.training import Trainer  # Import here to avoid circular imports
    
    # Set up default criterion if not provided
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    # Set up default optimizer if not provided
    if optimizer is None:
        # Only optimize parameters that require gradients
        params_to_update = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params_to_update, lr=0.001)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        **kwargs
    )
    
    # Train with frozen features first
    if freeze_epochs > 0:
        print(f"Training with frozen feature extractor for {freeze_epochs} epochs...")
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=freeze_epochs,
            verbose=1
        )
        
        # Unfreeze specified layers or all layers
        if unfreeze_layers is not None:
            # Unfreeze specific layers
            for name, param in model.named_parameters():
                if any(layer in name for layer in unfreeze_layers):
                    param.requires_grad = True
        else:
            # Unfreeze all layers
            for param in model.parameters():
                param.requires_grad = True
        
        # Update optimizer to include newly unfrozen parameters
        params_to_update = [p for p in model.parameters() if p.requires_grad]
        trainer.optimizer = torch.optim.Adam(params_to_update, lr=0.0001)
        
        print("Unfrozen layers. Continuing training...")
    
    # Continue training with unfrozen layers
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=num_epochs - (freeze_epochs if freeze_epochs > 0 else 0),
        verbose=1
    )
    
    return history


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_size: Tuple[int, int, int] = (3, 224, 224),
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset_version: int = 11,
    **kwargs
) -> None:
    """
    Export a PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to export
        output_path: Path to save the ONNX model
        input_size: Input tensor size (channels, height, width)
        input_names: Names for the input tensors
        output_names: Names for the output tensors
        dynamic_axes: Dynamic axes for variable length inputs/outputs
        opset_version: ONNX opset version to use
        **kwargs: Additional arguments for torch.onnx.export
    """
    # Set default names if not provided
    if input_names is None:
        input_names = ['input']
    if output_names is None:
        output_names = ['output']
    
    # Set default dynamic axes if not provided
    if dynamic_axes is None:
        dynamic_axes = {
            'input': {0: 'batch_size'},  # Variable length batch size
            'output': {0: 'batch_size'}
        }
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_size, device=next(model.parameters()).device)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Export the model
    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        **kwargs
    )
    
    print(f"Model exported to {output_path}")


def export_to_tflite(
    model: nn.Module,
    output_path: str,
    input_size: Tuple[int, int, int] = (3, 224, 224),
    quantize: bool = False,
    **kwargs
) -> None:
    """
    Export a PyTorch model to TensorFlow Lite format.
    
    Note: This requires the model to be exported to ONNX first, then converted to TFLite.
    
    Args:
        model: PyTorch model to export
        output_path: Path to save the TFLite model
        input_size: Input tensor size (channels, height, width)
        quantize: If True, apply post-training quantization
        **kwargs: Additional arguments for ONNX export
    """
    try:
        import onnx
        import tensorflow as tf
        from onnx_tf.backend import prepare
    except ImportError:
        raise ImportError(
            "To export to TFLite, please install the following packages:\n"
            "pip install onnx onnx-tf tensorflow\n"
            "Note: TensorFlow 2.x is required for ONNX-TF conversion."
        )
    
    # First, export to ONNX
    onnx_path = os.path.splitext(output_path)[0] + '.onnx'
    export_to_onnx(
        model=model,
        output_path=onnx_path,
        input_size=input_size,
        **kwargs
    )
    
    # Convert ONNX to TensorFlow
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    
    # Export to TensorFlow SavedModel
    saved_model_dir = os.path.splitext(output_path)[0] + '_saved_model'
    tf_rep.export_graph(saved_model_dir)
    
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    
    if quantize:
        # Apply post-training quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model exported to {output_path}")
    
    # Clean up temporary files
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
    
    if os.path.exists(saved_model_dir):
        import shutil
        shutil.rmtree(saved_model_dir)
