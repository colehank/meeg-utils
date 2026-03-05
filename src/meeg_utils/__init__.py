"""meeg-utils: A Python-based MEEG processing toolkit.

This package provides utilities for processing MEG and EEG data,
including preprocessing, epoching, and feature extraction.
"""

__version__ = "0.1.0"

from .preprocessing import BatchPreprocessingPipeline, PreprocessingPipeline

__all__ = [
    "BatchPreprocessingPipeline",
    "PreprocessingPipeline",
]
