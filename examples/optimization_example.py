"""
Optimization and Loss Function Example

This script demonstrates how to use the enhanced optimization and loss functionality
in the Neural Network Application.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# Import our custom modules
from neuralnetworkapp.trainingimport Trainer
from neuralnetworkapp.optimizationimport get_optimizer, get_scheduler
from neuralnetworkapp.lossesimport get_loss_function, CompositeLoss

# Set random seed for reproducibility
torch.manual_seed(42)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create a simple neural network for demonstration
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(7 * 7 * 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
val_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

def train_with_config(optimizer_name, loss_fn_name, scheduler_name=None, scheduler_params=None):
    print(f"\n{'='*50}")
    print(f"Training with: Optimizer={optimizer_name}, Loss={loss_fn_name}, Scheduler={scheduler_name}")
    print(f"{'='*50}")
    
    # Create model
    model = SimpleCNN().to(device)
    
    # Create optimizer with custom parameters if needed
    optimizer = get_optimizer(
        name=optimizer_name,
        model_params=model.parameters(),
        custom_params={
            'lr': 0.001,
            'weight_decay': 1e-4
        }
    )
    
    # Create loss function
    if isinstance(loss_fn_name, str):
        criterion = get_loss_function(loss_fn_name, device=device)
    elif isinstance(loss_fn_name, dict):
        # For composite loss
        criterion = CompositeLoss(loss_fn_name, device=device)
    else:
        raise ValueError("Invalid loss function configuration")
    
    # Create scheduler if specified
    scheduler = None
    if scheduler_name:
        scheduler = get_scheduler(
            name=scheduler_name,
            optimizer=optimizer,
            custom_params=scheduler_params or {}
        )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        metrics=['accuracy'],
        use_amp=True  # Enable mixed precision training if available
    )
    
    # Train the model
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        verbose=1
    )
    
    return history

# Example 1: Basic training with Adam optimizer and CrossEntropy loss
train_with_config(
    optimizer_name='adam',
    loss_fn_name='cross_entropy'
)

# Example 2: Training with SGD with momentum and StepLR scheduler
train_with_config(
    optimizer_name='sgd',
    loss_fn_name='cross_entropy',
    scheduler_name='steplr',
    scheduler_params={'step_size': 2, 'gamma': 0.5}
)

# Example 3: Training with RMSprop and composite loss
composite_loss = {
    'cross_entropy': {'reduction': 'mean'},
    'l1': {'reduction': 'mean'}
}

train_with_config(
    optimizer_name='rmsprop',
    loss_fn_name=composite_loss,
    scheduler_name='reduce_on_plateau',
    scheduler_params={'mode': 'min', 'factor': 0.5, 'patience': 2}
)

# Example 4: Training with different optimizers and loss functions
optimizers = ['adam', 'sgd', 'rmsprop', 'adagrad']
loss_fns = ['cross_entropy', 'focal_loss', 'label_smoothing']

for opt in optimizers:
    for loss_fn in loss_fns:
        try:
            train_with_config(
                optimizer_name=opt,
                loss_fn_name=loss_fn,
                scheduler_name='cosine',
                scheduler_params={'T_max': 5, 'eta_min': 1e-6}
            )
        except Exception as e:
            print(f"Error with {opt} and {loss_fn}: {str(e)}")
            continue
