# Project Structure

This document outlines the directory structure of the NeuralNetworkApp project.

```text
NeuralNetworkApp/
│
├── .github/                            # GitHub specific files
│   └── workflows/                      # GitHub Actions workflows
│       ├── tests.yml                   # CI/CD pipeline configuration
│       └── deploy.yml                  # Deployment workflow
│
├── assets/                             # Static assets (images, etc.)
│
├── checkpoints/                        # Saved model checkpoints
│   └── README.md                       # Checkpoint management
│
├── cli/                                # Command line interface
│   ├── __init__.py
│   └── cli_menu.py                     # Main CLI interface
│
├── data/                               # Datasets and data processing
│   ├── MNIST/                          # MNIST dataset
│   │   └── raw/                        # Raw dataset files
│   └── README.md                       # Data documentation
│
├── docs/                               # Documentation
│   ├── api/                            # API reference
│   ├── examples/                       # Example implementations
│   ├── guides/                         # Comprehensive guides
│   ├── images/                         # Documentation images
│   ├── Directory_Structure.md          # This file
│   └── README.md                       # Main documentation
│
├── examples/                           # Example scripts
│   ├── cifar10_training.py             # CIFAR-10 training example
│   ├── cnn_example.py                  # CNN implementation example
│   ├── network_builder_demo.py         # Network builder demo
│   ├── optimization_example.py         # Optimization techniques
│   └── README.md                       # Examples documentation
│
├── logs/                               # Training logs and metrics
│
├── src/                                # Source code
│   ├── network_builder/                # Network builder module
│   │   ├── __init__.py
│   │   └── network_builder.py          # Core network builder
│   │
│   └── neuralnetworkapp/               # Main package
│       ├── builder/                    # Model building utilities
│       ├── data/                       # Data loading and processing
│       ├── gui/                        # Graphical user interface
│       ├── layers/                     # Custom layers
│       ├── losses/                     # Loss functions
│       ├── metrics/                    # Evaluation metrics
│       ├── models/                     # Model architectures
│       ├── training/                   # Training utilities
│       ├── utils/                      # Helper functions
│       ├── __init__.py                 # Package initialization
│       └── version.py                  # Version information
│
├── tests/                              # Unit and integration tests
│   ├── test_cnn_models.py              # CNN model tests
│   ├── test_data_augmentation.py       # Data augmentation tests
│   ├── test_imports.py                 # Import tests
│   ├── test_network_architectures.py   # Architecture tests
│   └── conftest.py                     # Test configuration
│
├── utils/                              # Utility scripts
│   └── model_utils.py                  # Model utilities
│
├── .gitignore                          # Git ignore rules
├── CHANGELOG.md                        # Release history
├── LICENSE                             # License information
├── MANIFEST.in                         # Package data files
├── README.md                           # Project overview
├── TO_DO.md                            # Development roadmap
├── __init__.py                         # Python package marker
├── main.py                             # Main application entry point
├── pyproject.toml                      # Build system configuration
├── requirements.txt                    # Core dependencies
├── requirements-visualization.txt      # Visualization dependencies
└── setup.py                            # Package installation script
```

## Key Directories

### .github/
Contains GitHub-specific configurations including:
- Workflow definitions for CI/CD
- Issue and pull request templates
- Security policies

### checkpoints/
Stores trained model checkpoints and training states.

### cli/
Command-line interface implementation for interacting with the application.

### data/
- Raw and processed datasets
- Data loading and preprocessing utilities
- Dataset documentation

### docs/
Comprehensive documentation including:
- API reference
- Usage examples
- Development guides
- Architecture decisions

### examples/
Ready-to-run example scripts demonstrating various features:
- Model training
- Network building
- Optimization techniques

### logs/
Training logs, metrics, and visualizations.

### src/
Main source code organized into logical modules:
- `network_builder/`: Core network architecture utilities
- `neuralnetworkapp/`: Main application package
  - `builder/`: Model construction utilities
  - `data/`: Data loading and processing
  - `gui/`: Graphical user interface components
  - `layers/`: Custom neural network layers
  - `losses/`: Loss function implementations
  - `metrics/`: Evaluation metrics
  - `models/`: Model architectures
  - `training/`: Training loops and utilities
  - `utils/`: Helper functions and utilities

### tests/
Unit and integration tests ensuring code quality and functionality.

## Root Files

- `CHANGELOG.md`: Version history and release notes
- `LICENSE`: Software license (GPLv3)
- `pyproject.toml`: Build system configuration
- `README.md`: Project overview and getting started guide
- `requirements*.txt`: Python dependencies
- `setup.py`: Package installation script

---
© Copyright 2025 Nsfr750 - All Rights Reserved
