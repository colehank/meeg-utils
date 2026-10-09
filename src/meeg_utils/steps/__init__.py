"""Processing steps; use as ``from meeg_utils import steps as S``."""

from __future__ import annotations

from .autoreject import AutoReject
from .bridging import BridgedElectrodes
from .channels import BadChannels, Interpolate, Reference
from .compose import ByChannelType
from .denoise import ASR, SNS
from .epochs import Baseline, DropChannels, Epoch, FixedLengthEpochs
from .filtering import Filter, Resample
from .head import HeadAlign
from .hfc import HFC
from .ica import ICA
from .line_noise import LineNoise
from .maxwell import Maxwell
from .regression import Regression
from .segments import BadSegments

__all__ = [
    "ASR",
    "HFC",
    "ICA",
    "SNS",
    "AutoReject",
    "BadChannels",
    "BadSegments",
    "Baseline",
    "BridgedElectrodes",
    "ByChannelType",
    "DropChannels",
    "Epoch",
    "Filter",
    "FixedLengthEpochs",
    "HeadAlign",
    "Interpolate",
    "LineNoise",
    "Maxwell",
    "Reference",
    "Regression",
    "Resample",
]
