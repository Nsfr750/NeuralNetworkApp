"""
Transfer Learning Example

This script demonstrates how to use transfer learning with pre-trained models
and export them to ONNX and TensorFlow Lite formats.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR

# Import our transfer learning utilities
from neuralnetworkapp.transferimport get_pretrained_model, fine_tune_model, export_to_onnx, export_to_tflite
from neuralnetworkapp.trainingimport Trainer

# Set random seed for reproducibility
torch.manual_seed(42)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_custom_dataset(data_dir: str, batch_size: int = 32, input_size: int = 224):
    """
    Load a custom dataset from a directory structure like:
    data/
        train/
            class1/
            class2/
            ...
        val/
            class1/
            class2/
            ...
    
    Args:
        data_dir: Root directory of the dataset
        batch_size: Batch size for data loaders
        input_size: Size to resize images to
        
    Returns:
        Tuple of (train_loader, val_loader, num_classes, class_names)
    """
    # Data augmentation and normalization for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Just normalization for validation
    val_transform = transforms.Compose([
        transforms.Resize(input_size + 32),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    
    val_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Get class names
    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    print(f"Found {len(train_dataset)} training images in {num_classes} classes")
    print(f"Found {len(val_dataset)} validation images")
    print(f"Class names: {class_names}")
    
    return train_loader, val_loader, num_classes, class_names

def main():
    # Configuration
    data_dir = 'data/flowers'  # Update this to your dataset path
    model_name = 'resnet18'    # Try different models: 'resnet18', 'efficientnet_b0', 'mobilenet_v2', etc.
    num_epochs = 15
    batch_size = 32
    learning_rate = 0.001
    freeze_epochs = 5  # Number of epochs to keep feature extractor frozen
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    train_loader, val_loader, num_classes, class_names = load_custom_dataset(
        data_dir=data_dir,
        batch_size=batch_size,
        input_size=224  # Standard size for most pre-trained models
    )
    
    # Load pre-trained model
    print(f"Loading pre-trained {model_name}...")
    model = get_pretrained_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=True,
        freeze_features=True  # Freeze all layers initially
    ).to(device)
    
    # Print model summary
    print("\nModel architecture:")
    print(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Only optimize parameters that require gradients
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Fine-tune the model
    print("\nStarting training...")
    history = fine_tune_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device,
        freeze_epochs=freeze_epochs,
        unfreeze_layers=['layer4', 'fc'],  # Unfreeze last layer group and classifier
        metrics=['accuracy'],
        checkpoint_dir='checkpoints',
        checkpoint_freq=1
    )
    
    # Export the model
    print("\nExporting models...")
    
    # Create dummy input for export
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Export to ONNX
    onnx_path = f'output/{model_name}_flowers.onnx'
    export_to_onnx(
        model=model,
        output_path=onnx_path,
        input_size=(3, 224, 224),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Export to TensorFlow Lite (quantized)
    tflite_path = f'output/{model_name}_flowers_quant.tflite'
    try:
        export_to_tflite(
            model=model,
            output_path=tflite_path,
            input_size=(3, 224, 224),
            quantize=True,
            input_names=['input'],
            output_names=['output']
        )
    except Exception as e:
        print(f"Error exporting to TFLite: {e}")
        print("Make sure you have TensorFlow and ONNX-TF installed.")
    
    print("\nDone!")
    print(f"- ONNX model saved to {onnx_path}")
    print(f"- TensorFlow Lite model saved to {tflite_path}")
    print("\nTo use the models:")
    print(f"- ONNX: Use with ONNX Runtime or convert to other formats")
    print(f"- TFLite: Deploy on mobile devices with TensorFlow Lite")

if __name__ == '__main__':
    main()
