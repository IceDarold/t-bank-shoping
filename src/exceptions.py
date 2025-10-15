# src/exceptions.py

class MLFoundryError(Exception):
    """Base exception for ML Foundry."""
    pass


class ConfigurationError(MLFoundryError):
    """Raised when configuration is invalid."""
    pass


class DataLoadError(MLFoundryError):
    """Raised when data loading fails."""
    pass


class ModelLoadError(MLFoundryError):
    """Raised when model loading fails."""
    pass


class FileOperationError(MLFoundryError):
    """Raised when file operations fail."""
    pass


class ValidationError(MLFoundryError):
    """Raised when validation fails."""
    pass