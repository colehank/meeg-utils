"""Homogeneous / harmonic field correction for OPM arrays."""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any, ClassVar

import mne
import numpy as np
from mne import Evoked
from mne.epochs import BaseEpochs
from mne.io import BaseRaw

from ..core import Step
from ..core.step import Inst
from ._plotting import plot_psd_comparison, psd_summary


class HFC(Step):
    """Remove environmental fields from OPM data by a homogeneous or harmonic field model.

    On-head OPM arrays have no reference sensors or fixed helmet, so the
    interference is modelled as a spatially smooth field: homogeneous
    (``order=1``; Tierney et al., 2021, NeuroImage 244:118484) or a low-order
    spherical-harmonic expansion (``order=2`` adds gradients; Tierney et al.,
    2022, NeuroImage 258:119338). The field is fitted from the sensor
    positions and orientations alone (:func:`mne.preprocessing.compute_proj_hfc`)
    and projected out of the good MEG channels.

    Parameters
    ----------
    order : int
        Order of the field model: 1 (homogeneous, 3 components), 2 (plus
        gradients, 8 components), ...
    accuracy : {"point", "normal", "accurate"}
        Coil-definition accuracy used for the model.

    Attributes
    ----------
    projs_ : list of Projection
        The projectors, fitted from the sensor geometry.
    qc_ : dict
        ``n_components`` (projected out), ``n_channels``,
        ``axes_per_location`` (median number of sensors per location; with
        one, HFC also removes brain signal and a warning is raised),
        ``power_change_db``
        (good MEG channels, after relative to before; strongly negative
        values mean much interference was removed) and ``shielding_db``
        (power reduction below 10 Hz, where environmental fields dominate).
    psd_ : dict
        PSDs before and after the last transform.

    Notes
    -----
    The projectors are applied to the data and stored as active projectors
    in ``info["projs"]``, so later rank estimates (e.g. for ICA) account
    for them; existing projectors are left as they are. Mark bad channels
    first: they are excluded from the model and left unchanged.

    Figures (:meth:`plot`): ``"psd"``, spectra before and after.
    """

    accepts: ClassVar[tuple[type, ...]] = (BaseRaw, BaseEpochs, Evoked)
    systems: ClassVar[frozenset[str] | None] = frozenset({"opm"})
    plot_kinds: ClassVar[dict[str, bool]] = {"psd": False}

    def __init__(self, order: int = 1, *, accuracy: str = "accurate") -> None:
        self.order = order
        self.accuracy = accuracy

    def _fit(self, inst: Inst) -> None:
        picks = mne.pick_types(inst.info, meg=True, ref_meg=False, exclude="bads")
        if len(picks) == 0:
            raise ValueError("HFC needs good MEG channels.")
        self.projs_ = mne.preprocessing.compute_proj_hfc(
            inst.info, order=self.order, accuracy=self.accuracy, verbose=False
        )
        names = list(self.projs_[0]["data"]["col_names"])
        vectors = np.array([p["data"]["data"][0] for p in self.projs_]).T  # (n_ch, n_comp)
        basis, _ = np.linalg.qr(vectors)
        self.ch_names_ = names
        self.projector_ = np.eye(len(names)) - basis @ basis.T
        axes = _axes_per_location(inst.info, names)
        self.qc_.update(
            n_components=len(self.projs_), n_channels=len(names), axes_per_location=axes
        )
        if axes < 2:
            warnings.warn(
                "Every OPM location has a single sensor axis. On such arrays a homogeneous "
                "field looks much like the brain's own lowest-order field, so HFC removes "
                "brain signal too (in simulations with radial sensors, most of it); dual- "
                "or triaxial sensors avoid this.",
                stacklevel=4,
            )
        if len(self.projs_) >= len(names):
            raise ValueError(
                f"order={self.order} has {len(self.projs_)} components for {len(names)} channels; "
                "lower the order."
            )

    def _transform(self, inst: Inst) -> Inst:
        missing = [ch for ch in self.ch_names_ if ch not in inst.ch_names]
        if missing:
            raise ValueError(f"The data lack channels HFC was fitted on: {missing}.")
        idx = [inst.ch_names.index(ch) for ch in self.ch_names_]
        data = inst._data
        sfreq = inst.info["sfreq"]
        before = data[..., idx, :].copy()
        after = np.einsum("ij,...jt->...it", self.projector_, before)
        data[..., idx, :] = after

        power = float(np.mean(before**2))
        freqs, psd_before = psd_summary(before, sfreq)
        _, psd_after = psd_summary(after, sfreq)
        self.psd_ = {"mag": {"freqs": freqs, "before": psd_before, "after": psd_after}}
        low = (freqs > 0) & (freqs < 10)
        if power > 0:
            self.qc_["power_change_db"] = round(10 * np.log10(np.mean(after**2) / power), 2)
        if low.any() and psd_before[:, low].sum() > 0:
            ratio = psd_after[:, low].sum() / psd_before[:, low].sum()
            self.qc_["shielding_db"] = round(float(-10 * np.log10(ratio)), 2)

        with inst.info._unlock():
            for proj in self.projs_:
                applied = deepcopy(proj)
                applied["active"] = True
                inst.info["projs"].append(applied)
        return inst

    def _plot_unavailable(self, kind: str) -> str | None:
        if kind == "psd" and not hasattr(self, "psd_"):
            return "the step has not transformed data yet (use fit_transform)"
        return None

    def _plot_psd(self, inst: Any) -> Any:
        return plot_psd_comparison(
            self.psd_, title=f"Spectra before and after HFC (order {self.order})"
        )


def _axes_per_location(info: mne.Info, names: list[str]) -> float:
    """Median number of sensors sharing a location (within 1 mm)."""
    positions = np.array([info["chs"][info.ch_names.index(ch)]["loc"][:3] for ch in names])
    _, counts = np.unique(np.round(positions * 1000), axis=0, return_counts=True)
    return float(np.median(counts))
