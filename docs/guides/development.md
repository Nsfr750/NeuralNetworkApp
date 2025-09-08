# Development Setup Guide

This guide will help you set up your development environment for contributing to NeuralNetworkApp.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Development Workflow](#development-workflow)
- [Code Organization](#code-organization)
- [Version Control](#version-control)
- [Debugging](#debugging)
- [Contributing](#contributing)

## Prerequisites

### System Requirements
- Python 3.8+
- Git
- pip (Python package manager)
- Virtual environment (venv, conda, or pipenv)

### Recommended Tools
- Code editor (VS Code, PyCharm, etc.)
- Git client (GitHub Desktop, GitKraken, or command line)
- Docker (for containerized development)

## Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Nsfr750/NeuralNetworkApp.git
cd NeuralNetworkApp
```

### 2. Create a Virtual Environment
#### Using venv (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Using conda
```bash
conda create -n neuralnetworkapp python=3.9
conda activate neuralnetworkapp
```

### 3. Install Dependencies
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install package in development mode
pip install -e .
```

### 4. Install Pre-commit Hooks
```bash
pre-commit install
```

## Development Workflow

### 1. Branching Strategy
- `main`: Stable, production-ready code
- `develop`: Integration branch for features
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `docs/*`: Documentation updates

### 2. Creating a New Feature
```bash
# Create and switch to a new feature branch
git checkout -b feature/your-feature-name

# Make your changes
# Run tests and linting
pytest
black .
flake8

# Stage and commit changes
git add .
git commit -m "Add your feature"

# Push to remote
git push -u origin feature/your-feature-name
```

### 3. Running Tests
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=neuralnetworkapp

# Run a specific test file
pytest tests/test_module.py

# Run a specific test function
pytest tests/test_module.py::test_function_name
```

## Code Organization

### Project Structure
```
neuralnetworkapp/
├── src/                      # Source code
│   ├── neuralnetworkapp/     # Main package
│   └── tests/                # Test files
├── examples/                 # Example scripts
├── docs/                     # Documentation
└── scripts/                  # Utility scripts
```

### Module Structure
```
neuralnetworkapp/
├── __init__.py
├── layers/           # Custom layers
├── models/           # Model architectures
├── data/             # Data loading and processing
├── training/         # Training utilities
├── utils/            # Helper functions
└── visualization/    # Visualization tools
```

## Version Control

### Commit Message Format
```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or modifying tests
- `chore`: Maintenance tasks

### Example
```
feat(training): add learning rate scheduler

Add support for cosine annealing learning rate scheduling

Closes #123
```

## Debugging

### Debugging with VSCode
1. Add breakpoints in your code
2. Press F5 to start debugging
3. Use the debug console to inspect variables

### Debugging with pdb
```python
import pdb; pdb.set_trace()  # Add this line where you want to debug
```

### Common Issues
1. **Import Errors**
   - Make sure the package is installed in development mode
   - Check PYTHONPATH
   - Restart your IDE

2. **Dependency Conflicts**
   - Use a fresh virtual environment
   - Check `pip list` for conflicting packages

## Contributing

### Pull Request Process
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Update documentation
6. Submit a pull request

### Code Review Guidelines
- Keep PRs small and focused
- Include tests for new features
- Update documentation
- Follow the code style guide

## Development Tools

### Recommended VS Code Extensions
- Python
- Pylance
- Black Formatter
- Flake8
- GitLens
- Docker

### Useful Commands
```bash
# Format code
black .

# Check code style
flake8

# Run type checking
mypy src/

# Build documentation
cd docs && make html
```

## Getting Help

- Check the [FAQ](faq.md)
- Search existing issues
- Open a new issue if needed
- Join our [Discord](https://discord.gg/ryqNeuRYjD) for support

---
© Copyright 2025 Nsfr750. All Rights Reserved.
