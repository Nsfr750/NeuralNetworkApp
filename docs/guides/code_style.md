# Code Style Guide

This document outlines the coding standards and style guidelines for the NeuralNetworkApp project.

## Table of Contents
- [General Principles](#general-principles)
- [Python Style](#python-style)
- [Documentation](#documentation)
- [Naming Conventions](#naming-conventions)
- [Type Hints](#type-hints)
- [Error Handling](#error-handling)
- [Testing](#testing)
- [Version Control](#version-control)

## General Principles

1. **Readability Counts**
   - Code should be clear and easy to understand
   - Favor readability over cleverness
   - Follow the principle of least surprise

2. **Consistency**
   - Follow existing style in the codebase
   - Be consistent within a module or function

3. **Explicit over Implicit**
   - Avoid magic numbers and strings
   - Use named constants
   - Be explicit with imports

## Python Style

### Formatting
- **Line Length**: 88 characters maximum (Black default)
- **Indentation**: 4 spaces (no tabs)
- **Blank Lines**:
  - Two blank lines between top-level functions and classes
  - One blank line between methods

### Imports
```python
# Standard library imports
import os
import sys
from typing import List, Dict

# Third-party imports
import numpy as np
import tensorflow as tf

# Local application imports
from neuralnetworkapp import layers
from neuralnetworkapp.models import Model
```

### Black Formatting
We use [Black](https://github.com/psf/black) for automatic code formatting.

```bash
# Format all Python files
black .

# Check formatting without making changes
black --check .
```

### Flake8 Linting
We use [Flake8](https://flake8.pycqa.org/) for code linting.

```bash
# Run flake8
flake8
```

## Documentation

### Docstrings
Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for docstrings.

```python
def function_with_pep484_type_annotations(param1: int, param2: str) -> bool:
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    Raises:
        ValueError: If `param1` is None.
    """
    if param1 is None:
        raise ValueError("param1 cannot be None")
    return True
```

### Inline Comments
- Use sparingly
- Explain why, not what
- Keep them up-to-date

## Naming Conventions

### Variables and Functions
- Use `snake_case` for variables and functions
- Use descriptive names
- Avoid single-letter names (except in loops)

### Classes
- Use `CamelCase`
- Be descriptive and specific

### Constants
- Use `UPPER_SNAKE_CASE`
- Define at module level

### Type Variables
- Use `CamelCase`
- Prefixed with `T` (e.g., `TensorType`)

## Type Hints

### Basic Usage
```python
from typing import List, Dict, Optional, Union, Any

def process_data(
    data: List[Dict[str, Any]],
    batch_size: int = 32,
    verbose: bool = False
) -> Dict[str, float]:
    """Process input data and return statistics."""
    # Function implementation
    pass
```

### Type Aliases
```python
from typing import List, Tuple
import numpy as np

# For complex types
ArrayLike = Union[List[float], np.ndarray]
ModelInput = Tuple[np.ndarray, np.ndarray]
```

## Error Handling

### Exceptions
- Be specific with exception types
- Include helpful error messages
- Use custom exceptions when appropriate

```python
class ModelError(Exception):
    """Base class for model-related errors."""
    pass

class TrainingError(ModelError):
    """Raised when training fails."""
    pass

def train_model():
    try:
        # Training code
        pass
    except (ValueError, tf.errors.InvalidArgumentError) as e:
        raise TrainingError(f"Training failed: {e}") from e
```

## Testing

### Test Naming
- Test files should be named `test_*.py`
- Test functions should be named `test_*`
- Test classes should be named `Test*`

### Fixtures
- Use `pytest` fixtures for test dependencies
- Keep fixtures in `conftest.py` when shared

```python
import pytest

@pytest.fixture
def sample_data():
    """Return sample data for testing."""
    return np.random.rand(10, 10)

def test_data_processing(sample_data):
    """Test data processing function."""
    result = process_data(sample_data)
    assert result is not None
```

## Version Control

### Commit Messages
Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Branch Naming
- `feature/feature-name` for new features
- `bugfix/issue-description` for bug fixes
- `docs/documentation-update` for documentation
- `refactor/component-name` for refactoring

## Pre-commit Hooks

We use pre-commit hooks to enforce code quality:

```yaml
# .pre-commit-config.yaml
repos:
-   repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
    - id: black
      language_version: python3.9

-   repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
    - id: flake8
      additional_dependencies: [flake8-docstrings]

-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.961
    hooks:
    - id: mypy
      additional_dependencies: [types-requests]
```

## IDE Configuration

### VS Code Settings
```json
{
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

---
© Copyright 2025 Nsfr750. All Rights Reserved.
