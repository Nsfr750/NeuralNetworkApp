# Project Structure

This document outlines the directory structure of the NeuralNetworkApp project.

```text
NeuralNetworkApp/
│
├── .github/                            # GitHub specific files
│   ├── CODEOWNERS                      # Code ownership rules
│   └── FUNDING.yml                     # Funding configuration
│
├── assets/                             # Static assets (images, icons, etc.)
│
├── checkpoints/                        # Saved model checkpoints
│   ├── README.md                       # Checkpoint management
│   ├── custom_models/                  # Custom neural network models
│   ├── experiments/                    # Experimental models
│   └── mninst_cnn/                     # MNIST CNN models
│
├── cli/                                # Command line interface
│   └── cli_menu.py                     # Main CLI interface
│
├── data/                               # Datasets and data processing
│   ├── README.md                       # Data documentation
│   ├── MNIST/                          # MNIST dataset
│   │   ├── raw/                        # Raw dataset files
│   │   └── processed/                  # Processed dataset files
│   ├── CIFAR10/                        # CIFAR-10 dataset
│   │   ├── raw/                        # Raw dataset files
│   │   ├── processed/                  # Processed dataset files
│   │   └── README.md                   # CIFAR-10 documentation
│   ├── custom_datasets/                # User-defined datasets
│   └── temp/                           # Temporary data files
│
├── docs/                               # Documentation
│   ├── STRUCT.md                       # This file
│   ├── README.md                       # Main documentation
│   ├── api/                            # API reference
│   │   ├── data.md                     # Data API documentation
│   │   ├── models.md                   # Models API documentation
│   │   └── network_builder.md          # Network builder API documentation
│   ├── examples/                       # Example implementations
│   │   ├── index.md                    # Examples index
│   │   └── mnist_classification.md     # MNIST classification example
│   ├── guides/                         # Comprehensive guides
│   │   ├── architecture.md             # Architecture guide
│   │   ├── code_style.md               # Code style guide
│   │   ├── custom_layers.md            # Custom layers guide
│   │   ├── data_preprocessing.md       # Data preprocessing guide
│   │   ├── deployment.md               # Deployment guide
│   │   ├── development.md              # Development guide
│   │   ├── installation.md             # Installation guide
│   │   ├── optimization.md             # Optimization guide
│   │   ├── testing.md                  # Testing guide
│   │   ├── troubleshooting.md          # Troubleshooting guide
│   │   ├── usage.md                    # Usage guide
│   │   ├── visualization.md            # Visualization guide
│   │   ├── windows_deployment.md       # Windows deployment guide
│   │   └── writing_examples.md         # Writing examples guide
│   ├── images/                         # Documentation images
│   └── optimization_guide.md           # Optimization guide
│
├── examples/                           # Example scripts
│   ├── README.md                       # Examples documentation
│   ├── cifar10_training.py             # CIFAR-10 training example
│   ├── cnn_example.py                  # CNN implementation example
│   ├── network_builder_demo.py         # Network builder demo
│   ├── optimization_example.py         # Optimization techniques
│   └── transfer_learning_example.py    # Transfer learning example
│
├── logs/                               # Training logs and metrics
│
├── src/                                # Source code
│   ├── __init__.py                     # Source package initialization
│   ├── network_builder/                # Network builder module
│   │   ├── __init__.py                 # Network builder package
│   │   └── network_builder.py          # Core network builder
│   ├── neuralnetworkapp/               # Main application package
│   │   ├── __init__.py                 # Package initialization
│   │   ├── version.py                  # Version information
│   │   ├── builder/                    # Model building utilities
│   │   ├── data/                       # Data loading and processing
│   │   ├── gui/                        # Graphical user interface
│   │   ├── losses/                     # Loss function implementations
│   │   ├── metrics/                    # Evaluation metrics
│   │   ├── models/                     # Model architectures
│   │   ├── optimization/               # Optimization utilities
│   │   ├── training/                   # Training loops and utilities
│   │   ├── transfer/                   # Transfer learning utilities
│   │   ├── utils/                      # Helper functions
│   │   └── visualization/              # Visualization utilities
│   ├── ui/                             # User interface components
│   │   ├── __init__.py                 # UI package initialization
│   │   ├── about.py                    # About dialog
│   │   ├── help.py                     # Help system
│   │   ├── lang_mgr.py                 # Language manager
│   │   ├── main_window.py              # Main application window
│   │   ├── settings.py                 # Settings dialog
│   │   ├── theme_manager.py            # Theme manager
│   │   └── update_dialog.py            # Update dialog
│   ├── utils/                          # Utility modules
│   │   ├── __init__.py                 # Utils package initialization
│   │   └── updates.py                  # Update utilities
│   └── visualization.py                # Visualization utilities
│
├── tests/                              # Unit and integration tests
│   ├── test_cnn_models.py              # CNN model tests
│   ├── test_data_augmentation.py       # Data augmentation tests
│   ├── test_imports.py                 # Import tests
│   ├── test_network_architectures.py   # Architecture tests
│   ├── test_network_builder.py         # Network builder tests
│   ├── test_training_loop.py           # Training loop tests
│   ├── test_transfer_learning.py       # Transfer learning tests
│   └── test_visualization.py           # Visualization tests
│
├── utils/                              # Utility scripts
│   └── model_utils.py                  # Model utilities
│
├── .gitattributes                      # Git attributes
├── .gitignore                          # Git ignore rules
├── CHANGELOG.md                        # Release history
├── CODE_OF_CONDUCT.md                  # Code of conduct
├── CONTRIBUTING.md                     # Contribution guidelines
├── LICENSE                             # License information (GPLv3)
├── MANIFEST.in                         # Package data files
├── PREREQUISITES.md                    # Prerequisites and setup
├── README.md                           # Project overview and getting started
├── ROADMAP.md                          # Development roadmap
├── TO_DO.md                            # Development TODO list
├── __init__.py                         # Python package marker
├── example.py                          # Example usage script
├── main.py                             # Main application entry point
├── nuitka_compiler.py                  # Nuitka compilation script
├── pyproject.toml                      # Build system configuration
├── requirements.txt                    # Core dependencies
├── requirements-visualization.txt      # Visualization dependencies
├── requirements-windows.txt            # Windows-specific dependencies
└── setup.py                            # Package installation script
```

## Key Directories

### .github/
Contains GitHub-specific configurations:
- `CODEOWNERS`: Code ownership and review rules
- `FUNDING.yml`: GitHub Sponsors configuration

### assets/
Static assets including:
- Application icons
- Logo images
- Other visual resources

### checkpoints/
Organized storage for trained models:
- `custom_models/`: User-defined model architectures
- `experiments/`: Experimental and research models
- `mninst_cnn/`: MNIST-specific CNN models
- `README.md`: Comprehensive checkpoint management guide

### cli/
Command-line interface components:
- `cli_menu.py`: Main CLI menu and interaction system

### data/
Comprehensive dataset management:
- `MNIST/`: Handwritten digit dataset with raw/processed versions
- `CIFAR10/`: Image classification dataset with full documentation
- `custom_datasets/`: Framework for user-defined datasets
- `temp/`: Temporary storage for data processing
- `README.md`: Complete data management documentation

### docs/
Extensive documentation system:
- `api/`: Detailed API references for all modules
- `examples/`: Practical implementation examples
- `guides/`: 13 comprehensive guides covering all aspects
- `images/`: Supporting images for documentation
- Multiple specialized guides and references

### examples/
Practical example implementations:
- `cifar10_training.py`: Complete CIFAR-10 training pipeline
- `cnn_example.py`: Basic CNN implementation
- `network_builder_demo.py`: Dynamic network construction
- `optimization_example.py`: Various optimization techniques
- `transfer_learning_example.py`: Transfer learning implementation
- `README.md`: Comprehensive examples documentation

### logs/
Training and application logs:
- Training progress and metrics
- Error logs and debugging information
- Performance monitoring data

### src/
Well-organized source code architecture:
- `network_builder/`: Dynamic network construction utilities
- `neuralnetworkapp/`: Core application package with 11 specialized modules
- `ui/`: Complete user interface system with 8 components
- `utils/`: Shared utility functions and update system
- `visualization.py`: Standalone visualization utilities

### tests/
Comprehensive test suite:
- 8 test files covering all major components
- Model, data, training, and visualization tests
- Integration and unit tests

### utils/
Supporting utilities:
- `model_utils.py`: Model manipulation and analysis tools

## Root Files

### Configuration Files
- `pyproject.toml`: Modern Python build system configuration
- `setup.py`: Package installation and distribution
- `requirements*.txt`: Dependency management (core, visualization, Windows-specific)
- `MANIFEST.in`: Package data inclusion rules
- `.gitignore` & `.gitattributes`: Git configuration

### Documentation Files
- `README.md`: Project overview and quick start
- `CHANGELOG.md`: Version history and release notes
- `ROADMAP.md`: Future development plans
- `TO_DO.md`: Current development tasks
- `PREREQUISITES.md`: System requirements and setup
- `CONTRIBUTING.md`: Contribution guidelines
- `CODE_OF_CONDUCT.md`: Community standards

### Application Files
- `main.py`: Main application entry point (44KB)
- `example.py`: Usage examples and demonstrations
- `nuitka_compiler.py`: Standalone compilation script
- `__init__.py`: Python package initialization

### Legal Files
- `LICENSE`: GPLv3 license

---
© Copyright 2025 Nsfr750 - All Rights Reserved
