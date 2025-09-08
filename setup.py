#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages

# Read the contents of the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the version from a file
version_file = os.path.join("src", "neuralnetworkapp", "version.py")
with open(version_file, "r", encoding="utf-8") as f:
    exec(f.read())

# Find all packages under src/neuralnetworkapp
packages = find_packages(where="src")

setup(
    name="neuralnetworkapp",
    version=__version__,  # From version.py
    author="Nsfr750",
    author_email="nsfr750@yandex.com",
    description="A flexible neural network builder and trainer with visualization capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nsfr750/NeuralNetworkApp",
    project_urls={
        "Bug Tracker": "https://github.com/Nsfr750/NeuralNetworkApp/issues",
        "Documentation": "https://github.com/Nsfr750/NeuralNetworkApp#readme",
        "Source Code": "https://github.com/Nsfr750/NeuralNetworkApp",
        "Sponsor": "https://www.patreon.com/Nsfr750",
        "Donate": "https://paypal.me/3dmega",
    },
    packages=packages,
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.5",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "Pillow>=8.3.0",
        "scikit-learn>=0.24.2",
        "pandas>=1.3.0",
        "opencv-python>=4.5.0",
    ],
    extras_require={
        "all": [
            "torchviz>=0.0.2",
            "graphviz>=0.16",
            "tensorboard>=2.6.0",
            "umap-learn>=0.5.0",
            "pytest>=6.2.5",
            "pytest-cov>=2.12.0",
            "pytest-mock>=3.6.1",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.2",
            "sphinx-copybutton>=0.4.0",
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.5",
        ],
        "visualization": [
            "torchviz>=0.0.2",
            "graphviz>=0.16",
            "tensorboard>=2.6.0",
            "umap-learn>=0.5.0",
            "plotly>=5.0.0",
            "bokeh>=2.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.2",
            "sphinx-copybutton>=0.4.0",
            "nbsphinx>=0.8.0",
            "nbsphinx-link>=1.3.0",
        ],
        "tests": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.0",
            "pytest-mock>=3.6.1",
            "pytest-xdist>=2.3.0",
        ],
        "dev": [
            "black>=21.7b0",
            "isort>=5.9.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "pre-commit>=2.13.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "deep-learning",
        "neural-networks",
        "pytorch",
        "machine-learning",
        "computer-vision",
        "visualization",
        "deep-neural-networks",
        "artificial-intelligence",
        "ai",
        "model-training",
        "model-evaluation",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    # Entry points for command-line scripts
    entry_points={
        'console_scripts': [
            'neuralnetworkapp=neuralnetworkapp.cli.main:main',
        ],
    },
    # Additional metadata
    license="GPLv3",
    platforms="any",
)
