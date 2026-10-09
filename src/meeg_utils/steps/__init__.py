"""Processing steps; use as ``from meeg_utils import steps as S``."""

from __future__ import annotations

from .autoreject import AutoReject
from .bridging import BridgedElectrodes
from .channels import BadChannels, Interpolate, Reference
from .epochs import Baseline, DropChannels, Epoch
from .filtering import Filter, Resample
from .head import HeadAlign
from .ica import ICA
from .line_noise import LineNoise
from .maxwell import Maxwell

__all__ = [
    "ICA",
    "AutoReject",
    "BadChannels",
    "Baseline",
    "BridgedElectrodes",
    "DropChannels",
    "Epoch",
    "Filter",
    "HeadAlign",
    "Interpolate",
    "LineNoise",
    "Maxwell",
    "Reference",
    "Resample",
]
