"""Processing steps; use as ``from meeg_utils import steps as S``."""

from __future__ import annotations

from .channels import BadChannels, Interpolate, Reference
from .filtering import Filter, Resample
from .ica import ICA
from .line_noise import LineNoise

__all__ = [
    "ICA",
    "BadChannels",
    "Filter",
    "Interpolate",
    "LineNoise",
    "Reference",
    "Resample",
]
