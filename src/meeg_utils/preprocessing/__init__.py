"""Preprocessing module for MEG/EEG data.

This module provides preprocessing pipelines for MEG and EEG data,
including filtering, bad channel detection, line noise removal,
and ICA artifact removal.
"""

from __future__ import annotations

from .batch import BatchPreprocessingPipeline

# These imports will fail until we implement the classes
# This is expected in TDD - we write tests first!
from .pipeline import PreprocessingPipeline

__all__ = [
    "BatchPreprocessingPipeline",
    "PreprocessingPipeline",
]
