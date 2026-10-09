"""Toy steps used by the core tests."""

from __future__ import annotations

import warnings

import mne
import numpy as np
from mne.epochs import BaseEpochs
from mne.io import BaseRaw

from meeg_utils.core import Step


class Scale(Step):
    """Stateless step: multiply the data by a factor."""

    def __init__(self, factor: float = 2.0) -> None:
        self.factor = factor

    def _transform(self, inst):
        inst._data *= self.factor
        return inst


class Demean(Step):
    """Stateful step: subtract the per-channel means learned in fit."""

    def __init__(self, picks: tuple[str, ...] = ("eeg",)) -> None:
        self.picks = picks

    def _fit(self, inst):
        self.means_ = inst.get_data(picks=list(self.picks)).mean(axis=1, keepdims=True)
        self.qc_["max_abs_mean"] = float(np.abs(self.means_).max())

    def _transform(self, inst):
        picks = mne.pick_types(inst.info, eeg=True)
        inst._data[picks] -= self.means_
        return inst


class Recorder(Step):
    """Record calls in a class-level list (to check what ran)."""

    calls: list[str] = []  # noqa: RUF012

    def __init__(self, tag: str = "r") -> None:
        self.tag = tag

    def _fit(self, inst):
        Recorder.calls.append(f"fit:{self.tag}")

    def _transform(self, inst):
        Recorder.calls.append(f"transform:{self.tag}")
        return inst


class Warns(Step):
    """Emit a warning while fitting."""

    def _fit(self, inst):
        warnings.warn("something looks off", RuntimeWarning, stacklevel=2)

    def _transform(self, inst):
        return inst


class CTFOnly(Step):
    """A step restricted to CTF data."""

    systems = frozenset({"ctf"})

    def _transform(self, inst):
        return inst


class ToEpochs(Step):
    """Change the data type: Raw -> Epochs."""

    returns = BaseEpochs

    def __init__(self, duration: float = 1.0) -> None:
        self.duration = duration

    def _transform(self, inst):
        return mne.make_fixed_length_epochs(inst, duration=self.duration, preload=True)


class WithTuple(Step):
    """A step with a tuple parameter (serialized as a list)."""

    def __init__(self, band: tuple[float, float] = (1.0, 40.0)) -> None:
        self.band = band

    def _transform(self, inst):
        return inst


# Steps that break the contract ------------------------------------------


class DropsChannel(Step):
    """Drop a channel without declaring changes_channels."""

    def _transform(self, inst):
        return inst.drop_channels(inst.ch_names[:1])


class RebuildsRaw(Step):
    """Rebuild the Raw without first_samp (the bug found in the old code)."""

    def _transform(self, inst: BaseRaw):
        out = mne.io.RawArray(inst.get_data(), inst.info, verbose=False)
        out.set_annotations(inst.annotations)
        return out


class ModifiesInputInFit(Step):
    """Modify the input while fitting."""

    def _fit(self, inst):
        inst._data[0, 0] += 1.0

    def _transform(self, inst):
        return inst
