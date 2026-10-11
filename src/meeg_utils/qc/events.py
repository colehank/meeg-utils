"""Event and trigger checks."""

from __future__ import annotations

from typing import Any, ClassVar

import mne
import numpy as np
from mne.io import BaseRaw

from ._base import Check

#: Annotations that are not events.
_NOT_EVENTS = ("bad", "edge")


class Events(Check):
    """Event counts and timing, compared with what the experiment should produce.

    Events come from the annotations (excluding ``BAD``/``EDGE``
    annotations), or, if there are none, from the stimulus channel
    (:func:`mne.find_events`).

    Parameters
    ----------
    expected : dict | None
        Event name (annotation description, or trigger code as a string)
        mapped to the expected count. A different count is a ``"fail"``.
    stim_channel : str | None
        Stimulus channel; ``None`` uses MNE's default channels.
    min_interval : float | None
        Shortest plausible interval between consecutive events, in seconds;
        shorter intervals are flagged (e.g. trigger bouncing).

    Attributes
    ----------
    metrics_ : dict
        ``source`` (``"annotations"`` or the stimulus channel), ``counts``,
        ``n_events``, ``min_interval_s``, ``n_simultaneous`` (events sharing a
        sample) and ``n_in_bad`` (events inside ``BAD`` segments).

    Notes
    -----
    Figures: ``"timeline"`` (:func:`mne.viz.plot_events`).
    """

    name: ClassVar[str] = "events"
    plot_kinds: ClassVar[dict[str, bool]] = {"timeline": False}

    def __init__(
        self,
        *,
        expected: dict[str, int] | None = None,
        stim_channel: str | None = None,
        min_interval: float | None = None,
    ) -> None:
        self.expected = expected
        self.stim_channel = stim_channel
        self.min_interval = min_interval

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
        if self.expected or _event_annotations(raw) or self._stim_picks(raw):
            return None
        return "found no event annotations and no stimulus channel"

    def _stim_picks(self, raw: BaseRaw) -> list[int]:
        if self.stim_channel is not None:
            return (
                [raw.ch_names.index(self.stim_channel)] if self.stim_channel in raw.ch_names else []
            )
        return list(mne.pick_types(raw.info, meg=False, stim=True, exclude=[]))

    def _compute(self, raw: BaseRaw) -> None:
        sfreq = raw.info["sfreq"]
        if _event_annotations(raw):
            events, event_id = mne.events_from_annotations(
                raw, regexp=r"^(?![Bb][Aa][Dd]|[Ee][Dd][Gg][Ee]).*$", verbose=False
            )
            source = "annotations"
        elif self._stim_picks(raw):
            events = mne.find_events(
                raw, stim_channel=self.stim_channel, shortest_event=1, verbose=False
            )
            event_id = {str(code): int(code) for code in np.unique(events[:, 2])}
            source = self.stim_channel or "stim"
        else:
            events, event_id, source = np.zeros((0, 3), int), {}, "none"
        counts = {name: int((events[:, 2] == code).sum()) for name, code in event_id.items()}
        samples = events[:, 0]
        intervals = np.diff(samples) / sfreq
        positive = intervals[intervals > 0]
        bad_onsets, bad_ends = _bad_spans(raw)
        times = (samples - raw.first_samp) / sfreq
        in_bad = int(
            sum(
                ((times >= a) & (times < b)).sum()
                for a, b in zip(bad_onsets, bad_ends, strict=True)
            )
        )

        self.events_ = events
        self.event_id_ = event_id
        self.sfreq_ = sfreq
        self.first_samp_ = int(raw.first_samp)
        self.metrics_.update(
            source=source,
            counts=counts,
            n_events=len(events),
            min_interval_s=round(float(positive.min()), 4) if len(positive) else None,
            n_simultaneous=int((intervals == 0).sum()),
            n_in_bad=in_bad,
        )
        if self.expected:
            for name, n in self.expected.items():
                found = counts.get(str(name), 0)
                level = "ok" if found == n else "fail"
                self._note(
                    f"count_{name}", found, f"{name!r} events: {found} (expected {n})", level
                )
        else:
            self._note("n_events", len(events), f"{len(events)} events ({source})")
        self._judge(
            "n_simultaneous",
            self.metrics_["n_simultaneous"],
            warn=0,
            what="events sharing a sample with another event",
            detail="duplicated triggers or overlapping codes",
        )
        if self.min_interval is not None and len(positive):
            self._judge(
                "min_interval_s",
                self.metrics_["min_interval_s"],
                warn=self.min_interval,
                below=True,
                what="shortest interval between events",
                unit=" s",
                detail="trigger bouncing or spurious events",
            )
        self._note(
            "n_in_bad", in_bad, f"{in_bad} events inside BAD segments", "warn" if in_bad else "ok"
        )

    def _plot_timeline(self, inst: Any) -> Any:
        import matplotlib.pyplot as plt

        if len(self.events_) == 0:
            fig, ax = plt.subplots(figsize=(8, 2.5), layout="constrained")
            ax.text(0.5, 0.5, "no events", ha="center", va="center")
            ax.set_axis_off()
            return fig
        return mne.viz.plot_events(
            self.events_, sfreq=self.sfreq_, first_samp=self.first_samp_,
            event_id=self.event_id_ or None, show=False, verbose=False,
        )  # fmt: skip


def _event_annotations(raw: BaseRaw) -> bool:
    return any(not d.lower().startswith(_NOT_EVENTS) for d in raw.annotations.description)


def _bad_spans(raw: BaseRaw) -> tuple[list[float], list[float]]:
    """Onsets and ends (s, relative to the first sample) of BAD annotations."""
    annot = raw.annotations
    onsets, ends = [], []
    offset = raw.first_time if annot.orig_time is not None else 0.0
    for onset, duration, desc in zip(annot.onset, annot.duration, annot.description, strict=True):
        if desc.lower().startswith("bad"):
            onsets.append(float(onset - offset))
            ends.append(float(onset - offset + duration))
    return onsets, ends
