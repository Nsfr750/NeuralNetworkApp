"""
Version information for the neuralnetworkapp package.
"""

# Version as a string (PEP 440 compliant)
__version__ = "0.1.0"

# Version as a tuple for comparisons
version_info = tuple(int(part) for part in __version__.split('.') if part.isdigit())

__all__ = ['__version__', 'version_info']
