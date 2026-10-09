"""Repair of bridged EEG electrodes."""

from __future__ import annotations

from typing import Any, ClassVar

from mne import Evoked
from mne.epochs import BaseEpochs
from mne.io import BaseRaw

from ..core import Step
from ..core.step import Inst

LARGE_GROUP = ("raise", "bad")


class BridgedElectrodes(Step):
    """Detect bridged EEG electrodes and interpolate them.

    Detection is :class:`meeg_utils.qc.Bridging` (MNE's
    :func:`~mne.preprocessing.compute_bridged_electrodes`); repair is
    :func:`mne.preprocessing.interpolate_bridged_electrodes`, which places a
    virtual electrode at the centre of each bridged group, carrying the
    group's average signal, and interpolates the bridged electrodes from it
    and their neighbours. Channel names and order are unchanged.

    Parameters
    ----------
    lm_cutoff, epoch_threshold, l_freq, h_freq, epoch_duration : float
        Detection parameters (see :class:`meeg_utils.qc.Bridging`).
    bad_limit : int
        Groups of more bridged electrodes than this cannot be interpolated
        reliably (MNE default 4).
    large_groups : {"raise", "bad"}
        What to do with such groups: raise (default), or mark their
        electrodes bad (``info["bads"]``) and interpolate the other groups.

    Attributes
    ----------
    check_ : meeg_utils.qc.Bridging
        The detection, with its metrics and figures.
    pairs_ : list of tuple
        Bridged channel pairs (names).
    qc_ : dict
        ``pairs``, ``groups``, ``interpolated`` and ``marked_bad`` channels.

    Notes
    -----
    Needs electrode positions. Figures (:meth:`plot`): those of the
    detection, ``"topomap"`` and ``"distances"``.
    """

    accepts: ClassVar[tuple[type, ...]] = (BaseRaw, BaseEpochs, Evoked)
    systems: ClassVar[frozenset[str] | None] = None
    plot_kinds: ClassVar[dict[str, bool]] = {"topomap": False, "distances": False}

    def __init__(
        self,
        *,
        lm_cutoff: float = 16.0,
        epoch_threshold: float = 0.5,
        l_freq: float = 0.5,
        h_freq: float = 30.0,
        epoch_duration: float = 2.0,
        bad_limit: int = 4,
        large_groups: str = "raise",
    ) -> None:
        self.lm_cutoff = lm_cutoff
        self.epoch_threshold = epoch_threshold
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.epoch_duration = epoch_duration
        self.bad_limit = bad_limit
        self.large_groups = large_groups

    def _fit(self, inst: Inst) -> None:
        from ..qc import Bridging

        if self.large_groups not in LARGE_GROUP:
            raise ValueError(
                f"large_groups must be one of {LARGE_GROUP}, got {self.large_groups!r}"
            )
        if not isinstance(inst, BaseRaw):
            raise TypeError(
                "BridgedElectrodes detects bridges on Raw data; fit it on the recording."
            )
        check = Bridging(
            lm_cutoff=self.lm_cutoff,
            epoch_threshold=self.epoch_threshold,
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            epoch_duration=self.epoch_duration,
            bad_limit=self.bad_limit,
        )
        reason = check.not_applicable(inst)
        if reason is not None:
            raise ValueError(f"BridgedElectrodes {reason}.")
        self.check_ = check.compute(inst)
        self.pairs_ = [tuple(p) for p in check.metrics_["pairs"]]
        groups = check.metrics_["groups"]
        large = [g for g in groups if len(g) > self.bad_limit]
        if large and self.large_groups == "raise":
            raise ValueError(
                f"Bridged groups larger than bad_limit={self.bad_limit} cannot be interpolated "
                f"reliably: {large}. Fix the cap, or pass large_groups='bad' to mark them bad."
            )
        self.marked_bad_ = sorted(ch for g in large for ch in g)
        self.qc_.update(
            pairs=[list(p) for p in self.pairs_],
            groups=groups,
            interpolated=sorted(ch for g in groups if len(g) <= self.bad_limit for ch in g),
            marked_bad=self.marked_bad_,
        )

    def _transform(self, inst: Inst) -> Inst:
        import mne

        missing = sorted({ch for p in self.pairs_ for ch in p} - set(inst.ch_names))
        if missing:
            raise ValueError(f"Bridged channels {missing} are not in the data.")
        bad = set(self.marked_bad_)
        idx = [
            (inst.ch_names.index(a), inst.ch_names.index(b))
            for a, b in self.pairs_
            if a not in bad and b not in bad
        ]
        if idx:
            bads = list(inst.info["bads"])
            mne.preprocessing.interpolate_bridged_electrodes(inst, idx, bad_limit=self.bad_limit)
            inst.info["bads"] = bads
        if bad:
            inst.info["bads"] = sorted(set(inst.info["bads"]) | bad, key=inst.ch_names.index)
        return inst

    def _plot_topomap(self, inst: Any, **kwargs: Any) -> Any:
        return self.check_.plot("topomap", **kwargs)["topomap"]

    def _plot_distances(self, inst: Any) -> Any:
        return self.check_.plot("distances")["distances"]
