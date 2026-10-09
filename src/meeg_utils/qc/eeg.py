"""EEG-specific acquisition checks: electrode bridging and impedances."""

from __future__ import annotations

from typing import Any, ClassVar

import mne
import numpy as np
from mne.io import BaseRaw

from ._base import Check


class Bridging(Check):
    """Electrode bridging: neighbouring electrodes shorted by gel or sweat.

    Bridged electrodes record nearly the same signal, so the electrical
    distance (variance of their difference) is close to zero. Uses
    :func:`mne.preprocessing.compute_bridged_electrodes`, the local-minimum
    method of Tenke & Kayser (2001, Clin Neurophysiol 112:545) and Greischar
    et al. (2004, Clin Neurophysiol 115:710) as implemented in MNE, with
    MNE's defaults. The method looks for a local minimum in the distribution
    of small distances, so it needs enough bridged samples: a single bridged
    pair in a short recording can go undetected (the ``"distances"`` figure
    still shows it as a separate cluster near zero).

    The electrical distance is absolute (µV²), so in recordings with small
    signals, neighbouring electrodes that are not bridged can fall below
    the cutoff too (seen in HAD-MEEG, where 0.5-30 Hz EEG is about 7 µV).
    Each candidate pair is therefore confirmed by its correlation after
    average referencing, in a band above slow artifacts (15-30 Hz by
    default): a bridge is a physical short, so the two electrodes record
    nearly identical signals at every frequency and their correlation is
    close to 1, while unbridged neighbours with small signals are not, and
    broad slow artifacts (sweat, movement), which correlate distant
    electrodes at low frequencies, are excluded. In HAD-MEEG (sub-01),
    bridged rows of electrodes kept correlations of 0.98-0.99 from 15 to
    90 Hz, unbridged neighbours 0.6-0.8.

    Parameters
    ----------
    lm_cutoff : float
        Electrical-distance cutoff (µV²) below which a local minimum of the
        distance distribution indicates bridging (MNE default 16).
    epoch_threshold : float
        Fraction of 2-s epochs in which a pair must be bridged (MNE default 0.5).
    l_freq, h_freq : float
        Band-pass applied before computing distances (MNE default 0.5-30 Hz).
    epoch_duration : float
        Epoch length in seconds (MNE default 2).
    min_correlation : float | None
        Candidate pairs whose correlation (average reference,
        ``confirm_band``) is below this are not counted as bridged. There is
        no published value; 0.98 is this library's choice. ``None`` keeps
        every pair MNE finds.
    confirm_band : tuple of float
        Band of the confirming correlation (Hz).
    bad_limit : int
        A group of more bridged electrodes than this cannot be repaired by
        :func:`mne.preprocessing.interpolate_bridged_electrodes` (MNE default
        4); such a group is a ``"fail"``. Any bridge is a ``"warn"``: the
        electrodes should be fixed during the session, as bridged channels
        carry no independent information.

    Attributes
    ----------
    metrics_ : dict
        ``n_pairs``, ``pairs`` (channel-name pairs), ``groups`` (connected
        bridged electrodes) and ``largest_group``, after confirmation;
        ``n_candidates`` (pairs found by MNE) and ``unconfirmed`` (candidate
        pairs with their correlation below ``min_correlation``).

    Notes
    -----
    Figures: ``"topomap"`` (:func:`mne.viz.plot_bridged_electrodes`; needs
    electrode positions) and ``"distances"`` (distribution of electrical
    distances with the cutoff).
    """

    name: ClassVar[str] = "bridging"
    modalities: ClassVar[frozenset[str]] = frozenset({"eeg"})
    plot_kinds: ClassVar[dict[str, bool]] = {"topomap": False, "distances": False}

    def __init__(
        self,
        *,
        lm_cutoff: float = 16.0,
        epoch_threshold: float = 0.5,
        l_freq: float = 0.5,
        h_freq: float = 30.0,
        epoch_duration: float = 2.0,
        min_correlation: float | None = 0.98,
        confirm_band: tuple[float, float] = (15.0, 30.0),
        bad_limit: int = 4,
    ) -> None:
        self.lm_cutoff = lm_cutoff
        self.epoch_threshold = epoch_threshold
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.epoch_duration = epoch_duration
        self.min_correlation = min_correlation
        self.confirm_band = confirm_band
        self.bad_limit = bad_limit

    def not_applicable(self, raw: BaseRaw) -> str | None:
        """Return why the check cannot run on ``raw``, or ``None`` if it can.

        Parameters
        ----------
        raw : Raw
            The recording.

        Returns
        -------
        str | None
            The reason, or ``None``.
        """
        reason = super().not_applicable(raw)
        if reason is None and len(_live_eeg(raw)) < 2:
            reason = "needs at least two EEG channels that are not flat"
        return reason

    def _compute(self, raw: BaseRaw) -> None:
        # flat channels have zero distance to each other and are not bridges
        eeg = raw.copy().pick(_live_eeg(raw))
        bridged_idx, ed_matrix = mne.preprocessing.compute_bridged_electrodes(
            eeg,
            lm_cutoff=self.lm_cutoff,
            epoch_threshold=self.epoch_threshold,
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            epoch_duration=self.epoch_duration,
            verbose=False,
        )
        names = eeg.ch_names
        candidates = [(int(a), int(b)) for a, b in bridged_idx]
        unconfirmed: list[list[Any]] = []
        if self.min_correlation is not None and candidates:
            r = _correlations(eeg, candidates, *self.confirm_band)
            unconfirmed = [
                [names[a], names[b], round(float(ri), 3)]
                for (a, b), ri in zip(candidates, r, strict=True)
                if ri < self.min_correlation
            ]
            confirmed = [
                p for p, ri in zip(candidates, r, strict=True) if ri >= self.min_correlation
            ]
        else:
            confirmed = candidates
        pairs = sorted(tuple(sorted((names[a], names[b]))) for a, b in confirmed)
        groups = _connected_groups(pairs)
        self.bridged_idx_ = confirmed
        self.ed_matrix_ = ed_matrix.astype(np.float32)
        self.info_ = eeg.info
        self.metrics_.update(
            n_candidates=len(candidates),
            unconfirmed=unconfirmed,
            n_pairs=len(pairs),
            pairs=[list(p) for p in pairs],
            groups=groups,
            largest_group=max((len(g) for g in groups), default=0),
        )
        self._judge(
            "n_pairs",
            len(pairs),
            warn=0,
            what="bridged electrode pairs",
            detail="check the gel between " + ", ".join("-".join(p) for p in pairs[:10]),
        )
        self._judge(
            "largest_group",
            self.metrics_["largest_group"],
            fail=self.bad_limit,
            what="largest group of bridged electrodes",
            detail="too many to interpolate; the group should be marked bad",
        )

    def _plot_topomap(self, inst: Any, **kwargs: Any) -> Any:
        eeg = mne.pick_types(self.info_, eeg=True, exclude=[])
        if not any(np.isfinite(self.info_["chs"][i]["loc"][:3]).all() for i in eeg):
            raise ValueError("The bridging topomap needs electrode positions (set a montage).")
        topomap_args = {"vlim": (None, self.lm_cutoff), **kwargs.pop("topomap_args", {})}
        return mne.viz.plot_bridged_electrodes(
            self.info_,
            self.bridged_idx_,
            self.ed_matrix_,
            title=kwargs.pop("title", f"Bridged electrodes ({self.metrics_['n_pairs']} pairs)"),
            topomap_args={"show": False, **topomap_args},
        )

    def _plot_distances(self, inst: Any) -> Any:
        import matplotlib.pyplot as plt

        values = self.ed_matrix_[np.isfinite(self.ed_matrix_)]
        values = values[values > 0]
        fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
        bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 80)
        ax.hist(values, bins=bins.tolist(), color="C0", alpha=0.8)
        ax.axvline(self.lm_cutoff, color="C3", ls="--", label=f"cutoff {self.lm_cutoff:g} µV²")
        ax.set(
            xscale="log",
            xlabel="Electrical distance (µV²)",
            ylabel="Electrode pairs x epochs",
            title="Electrical distances (bridges: local minimum below the cutoff)",
        )
        ax.legend(frameon=False)
        return fig


def _correlations(
    eeg: BaseRaw, pairs: list[tuple[int, int]], l_freq: float, h_freq: float
) -> np.ndarray:
    """Correlation of each channel pair after band-pass and average reference."""
    data = eeg.get_data()
    data = mne.filter.filter_data(data, eeg.info["sfreq"], l_freq, h_freq, verbose=False)
    data = data - data.mean(axis=0, keepdims=True)
    data = data - data.mean(axis=1, keepdims=True)
    norms = np.sqrt((data**2).sum(axis=1))
    return np.array([(data[a] @ data[b]) / (norms[a] * norms[b]) for a, b in pairs])


class Impedance(Check):
    """Electrode impedances measured before or during the recording.

    Read from the BrainVision header (``raw.impedances``) or given
    explicitly (e.g. from :func:`mne.io.read_impedances_curry`). High
    impedance increases line-noise pickup and low-frequency noise (Kappenman
    & Luck, 2010, Psychophysiology 47:888).

    Parameters
    ----------
    impedances : dict | None
        Channel name mapped to impedance in kΩ; ``None`` reads
        ``raw.impedances``.
    warn_kohm : float
        Impedance above which an electrode is flagged. The default, 25 kΩ, is
        Brain Products' recommended maximum for active electrodes; use 5-10
        kΩ for passive electrodes, as the acceptable value depends on the
        amplifier's input impedance and the electrode type.

    Attributes
    ----------
    metrics_ : dict
        ``impedances`` (kΩ per channel), ``median_kohm``, ``max_kohm`` and
        ``high`` (channels above ``warn_kohm``).

    Notes
    -----
    Figures: ``"channels"`` (impedance per electrode with the threshold).
    """

    name: ClassVar[str] = "impedance"
    modalities: ClassVar[frozenset[str]] = frozenset({"eeg"})
    plot_kinds: ClassVar[dict[str, bool]] = {"channels": False}

    def __init__(self, *, impedances: dict[str, float] | None = None, warn_kohm: float = 25.0):
        self.impedances = impedances
        self.warn_kohm = warn_kohm

    def not_applicable(self, raw: BaseRaw) -> str | None:
        """Return why the check cannot run on ``raw``, or ``None`` if it can.

        Parameters
        ----------
        raw : Raw
            The recording.

        Returns
        -------
        str | None
            The reason, or ``None``.
        """
        reason = super().not_applicable(raw)
        if reason is None and self.impedances is None and not getattr(raw, "impedances", None):
            reason = "found no impedances in the recording (pass impedances=...)"
        return reason

    def _compute(self, raw: BaseRaw) -> None:
        eeg = set(mne.pick_info(raw.info, mne.pick_types(raw.info, eeg=True, exclude=[])).ch_names)
        if self.impedances is not None:
            values = {ch: float(v) for ch, v in self.impedances.items()}
        else:
            values = {ch: _to_kohm(entry) for ch, entry in raw.impedances.items()}
        values = {ch: v for ch, v in values.items() if ch in eeg and np.isfinite(v)}
        if not values:
            raise ValueError("No impedance values for the EEG channels.")
        high = sorted(
            (ch for ch, v in values.items() if v > self.warn_kohm), key=lambda ch: values[ch]
        )
        self.metrics_.update(
            impedances=values,
            median_kohm=float(np.median(list(values.values()))),
            max_kohm=float(max(values.values())),
            high=high,
        )
        self._judge(
            "n_high",
            len(high),
            warn=0,
            what=f"electrodes above {self.warn_kohm:g} kΩ",
            detail=", ".join(high[:15]),
        )

    def _plot_channels(self, inst: Any) -> Any:
        import matplotlib.pyplot as plt

        values = self.metrics_["impedances"]
        names = sorted(values, key=lambda ch: values[ch], reverse=True)
        fig, ax = plt.subplots(figsize=(max(6.0, 0.18 * len(names)), 3.5), layout="constrained")
        colors = ["C3" if values[n] > self.warn_kohm else "C0" for n in names]
        ax.bar(range(len(names)), [values[n] for n in names], color=colors)
        ax.axhline(self.warn_kohm, color="C3", ls="--", lw=1)
        ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
        ax.set(ylabel="Impedance (kΩ)", title="Electrode impedances")
        return fig


def _live_eeg(raw: BaseRaw) -> list[str]:
    """EEG channels (including marked bads) whose signal is not constant."""
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if not len(picks):
        return []
    std = raw.get_data(picks=picks).std(axis=1)
    return [raw.ch_names[p] for p, s in zip(picks, std, strict=True) if s > 0]


def _to_kohm(entry: Any) -> float:
    """Impedance from a BrainVision entry (``{"imp": value, "imp_unit": unit}``) in kΩ."""
    if not isinstance(entry, dict):
        return float(entry)
    scale = {"ohm": 1e-3, "kohm": 1.0, "mohm": 1e3}.get(str(entry.get("imp_unit", "kOhm")).lower())
    if scale is None:
        raise ValueError(f"Unknown impedance unit {entry.get('imp_unit')!r}")
    value = entry.get("imp")
    return float("nan") if value is None else float(value) * scale


def _connected_groups(pairs: list[tuple[str, str]]) -> list[list[str]]:
    """Connected components of the bridging graph, as sorted name lists."""
    parent: dict[str, str] = {}

    def root(x: str) -> str:
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x

    for a, b in pairs:
        parent[root(a)] = root(b)
    groups: dict[str, list[str]] = {}
    for x in parent:
        groups.setdefault(root(x), []).append(x)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: (-len(g), g))
