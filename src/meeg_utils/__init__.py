"""meeg-utils: A Python-based MEEG processing toolkit.

This package provides utilities for processing MEG and EEG data,
including preprocessing, epoching, and feature extraction.
"""

__version__ = "0.1.0"

# Initialize logging system when package is imported
from .logger import logger, setup_logging
from .preprocessing import BatchPreprocessingPipeline, PreprocessingPipeline

__all__ = [
    "BatchPreprocessingPipeline",
    "PreprocessingPipeline",
    "logger",
    "setup_logging",
]
