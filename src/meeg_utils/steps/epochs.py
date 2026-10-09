"""Epoching: selecting channels, cutting epochs around events, baseline correction."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, ClassVar

import mne
import numpy as np
from mne import Evoked
from mne.epochs import BaseEpochs
from mne.io import BaseRaw

from ..core import Step
from ..core.step import Inst

EVENT_SOURCES = ("auto", "annotations", "stim")
BASELINE_MODES = ("mean", "ratio", "logratio", "percent", "zscore", "zlogratio")
_NOT_EVENTS = ("bad", "edge")


class DropChannels(Step):
    """Remove channels, e.g. mastoid electrodes before re-referencing.

    Parameters
    ----------
    ch_names : list of str
        Channels to remove.
    on_missing : {"raise", "ignore"}
        What to do if a channel is not in the data.

    Attributes
    ----------
    qc_ : dict
        ``dropped`` and ``missing`` channels.
    """

    accepts: ClassVar[tuple[type, ...]] = (BaseRaw, BaseEpochs, Evoked)
    changes_channels: ClassVar[bool] = True

    def __init__(self, ch_names: list[str] | tuple[str, ...], *, on_missing: str = "raise") -> None:
        self.ch_names = ch_names
        self.on_missing = on_missing

    def _fit(self, inst: Inst) -> None:
        if self.on_missing not in ("raise", "ignore"):
            raise ValueError(f"on_missing must be 'raise' or 'ignore', got {self.on_missing!r}")
        names = [self.ch_names] if isinstance(self.ch_names, str) else list(self.ch_names)
        missing = [ch for ch in names if ch not in inst.ch_names]
        if missing and self.on_missing == "raise":
            raise ValueError(
                f"Channels {missing} are not in the data (on_missing='ignore' to skip)."
            )
        self.dropped_ = [ch for ch in names if ch in inst.ch_names]
        self.qc_.update(dropped=self.dropped_, missing=missing)

    def _transform(self, inst: Inst) -> Inst:
        return inst.drop_channels([ch for ch in self.dropped_ if ch in inst.ch_names])


class Epoch(Step):
    """Cut the recording into epochs around events.

    Events come from the annotations (excluding ``BAD``/``EDGE``; for BIDS
    data these are the ``trial_type`` of ``events.tsv``) or from the
    stimulus channel. The event names and codes are fixed when fitting, so
    every run transformed with the fitted step uses the same codes.

    Parameters
    ----------
    event_id : None | str | list of str | dict
        Events to epoch: ``None`` for all; names (or MNE's ``/``-separated
        tags, e.g. ``"video"`` matches ``"video/on"``); or a dict mapping
        names to codes.
    tmin, tmax : float
        Epoch start and end relative to the events, in seconds.
    baseline : tuple | None
        Baseline interval subtracted by :class:`mne.Epochs` (MNE's default
        ``(None, 0)``); ``None`` to leave it to :class:`Baseline`.
    events : {"auto", "annotations", "stim"}
        Where to find the events; ``"auto"`` uses the annotations if there
        are event annotations, otherwise the stimulus channel.
    stim_channel : str | None
        Stimulus channel (default: MNE's choice).
    reject, flat : dict | None
        Peak-to-peak rejection limits per channel type (see :class:`mne.Epochs`).
    reject_by_annotation : bool
        Drop epochs overlapping ``BAD`` annotations.
    decim : int
        Decimation factor (low-pass the data first).
    detrend : int | None
        0 to remove the mean, 1 to remove a linear trend of each epoch.
    metadata : None | "bids" | str
        ``"bids"`` adds the rows of the recording's BIDS ``events.tsv``
        (matched by onset); a path adds the rows of a TSV/CSV file with one
        row per selected event. Needs pandas.
    event_repeated : {"error", "drop", "merge"}
        What to do with several events at the same sample.

    Attributes
    ----------
    event_id_ : dict
        Event names mapped to codes.
    qc_ : dict
        ``n_events`` (per condition), ``n_kept``, ``n_dropped``,
        ``fraction_kept`` and ``drop_reasons`` (count per reason).

    Notes
    -----
    Figures (:meth:`plot`): ``"drop_log"`` (:func:`mne.viz.plot_drop_log`)
    and ``"evoked"`` (averages per condition,
    :func:`mne.viz.plot_compare_evokeds`, one figure per channel type).
    """

    accepts: ClassVar[tuple[type, ...]] = (BaseRaw,)
    returns: ClassVar[type | None] = BaseEpochs
    changes_times: ClassVar[bool] = True
    plot_kinds: ClassVar[dict[str, bool]] = {"drop_log": False, "evoked": False}

    def __init__(
        self,
        event_id: Any = None,
        tmin: float = -0.2,
        tmax: float = 0.5,
        *,
        baseline: tuple[float | None, float | None] | None = (None, 0.0),
        events: str = "auto",
        stim_channel: str | None = None,
        reject: dict[str, float] | None = None,
        flat: dict[str, float] | None = None,
        reject_by_annotation: bool = True,
        decim: int = 1,
        detrend: int | None = None,
        metadata: str | None = None,
        event_repeated: str = "error",
    ) -> None:
        self.event_id = event_id
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        self.events = events
        self.stim_channel = stim_channel
        self.reject = reject
        self.flat = flat
        self.reject_by_annotation = reject_by_annotation
        self.decim = decim
        self.detrend = detrend
        self.metadata = metadata
        self.event_repeated = event_repeated

    def _fit(self, inst: BaseRaw) -> None:
        if self.events not in EVENT_SOURCES:
            raise ValueError(f"events must be one of {EVENT_SOURCES}, got {self.events!r}")
        if self.tmin >= self.tmax:
            raise ValueError(f"tmin ({self.tmin}) must be before tmax ({self.tmax}).")
        self.source_ = self._source(inst)
        events, found = self._find_events(inst, None)
        self.event_id_ = _select_events(found, self.event_id)
        counts = Counter(events[:, 2])
        self.qc_["n_events_fit"] = {n: int(counts.get(c, 0)) for n, c in self.event_id_.items()}

    def _transform(self, inst: BaseRaw) -> BaseEpochs:
        events, _ = self._find_events(inst, self.event_id_)
        events = events[np.isin(events[:, 2], list(self.event_id_.values()))]
        if len(events) == 0:
            raise ValueError(f"No events {sorted(self.event_id_)} in the data.")
        present = {n: c for n, c in self.event_id_.items() if c in events[:, 2]}
        metadata = None if self.metadata is None else _metadata(self.metadata, inst, events)
        epochs = mne.Epochs(
            inst,
            events,
            event_id=present,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=None if self.baseline is None else tuple(self.baseline),
            reject=self.reject,
            flat=self.flat,
            reject_by_annotation=self.reject_by_annotation,
            decim=self.decim,
            detrend=self.detrend,
            metadata=metadata,
            event_repeated=self.event_repeated,
            preload=True,
            verbose=False,
        )
        reasons = Counter(
            reason for log in epochs.drop_log for reason in log if reason != "IGNORED"
        )
        n_events = Counter(events[:, 2])
        kept = len(epochs)
        self.drop_log_ = epochs.drop_log
        self.evoked_ = {name: epochs[name].average() for name in present if len(epochs[name])}
        self.qc_.update(
            n_events={n: int(n_events.get(c, 0)) for n, c in present.items()},
            n_kept=kept,
            n_dropped=len(events) - kept,
            fraction_kept=round(kept / len(events), 4),
            drop_reasons=dict(reasons),
        )
        return epochs

    def _source(self, raw: BaseRaw) -> str:
        if self.events != "auto":
            return self.events
        if any(not d.lower().startswith(_NOT_EVENTS) for d in raw.annotations.description):
            return "annotations"
        return "stim"

    def _find_events(
        self, raw: BaseRaw, event_id: dict[str, int] | None
    ) -> tuple[np.ndarray, dict[str, int]]:
        if self.source_ == "annotations":
            if event_id is None:
                events, found = mne.events_from_annotations(
                    raw, regexp=r"^(?![Bb][Aa][Dd]|[Ee][Dd][Gg][Ee]).*$", verbose=False
                )
                return np.asarray(events), dict(found)
            present = set(raw.annotations.description)
            ids = {n: c for n, c in event_id.items() if n in present}
            if not ids:
                return np.zeros((0, 3), int), {}
            events, _ = mne.events_from_annotations(raw, event_id=ids, verbose=False)
            return np.asarray(events), ids
        events = mne.find_events(
            raw, stim_channel=self.stim_channel, shortest_event=1, verbose=False
        )
        return events, {str(c): int(c) for c in np.unique(events[:, 2])}

    def _plot_drop_log(self, inst: Any) -> Any:
        if not hasattr(self, "drop_log_"):
            raise ValueError("The 'drop_log' plot needs the step to have transformed data.")
        return mne.viz.plot_drop_log(self.drop_log_, show=False)

    def _plot_evoked(self, inst: Any) -> Any:
        if not hasattr(self, "evoked_"):
            raise ValueError("The 'evoked' plot needs the step to have transformed data.")
        return mne.viz.plot_compare_evokeds(self.evoked_, combine="gfp", show=False)


class FixedLengthEpochs(Step):
    """Cut a continuous recording into consecutive epochs of equal length (e.g. resting state).

    Wraps :func:`mne.make_fixed_length_epochs`. Epochs overlapping ``BAD``
    annotations are dropped.

    Parameters
    ----------
    duration : float
        Epoch length in seconds.
    overlap : float
        Overlap between consecutive epochs in seconds.
    reject_by_annotation : bool
        Drop epochs overlapping ``BAD`` annotations.

    Attributes
    ----------
    qc_ : dict
        ``n_epochs`` (before dropping), ``n_kept``, ``fraction_kept`` and
        ``duration_kept_s``.

    Notes
    -----
    Figures (:meth:`plot`): ``"drop_log"``.
    """

    accepts: ClassVar[tuple[type, ...]] = (BaseRaw,)
    returns: ClassVar[type | None] = BaseEpochs
    changes_times: ClassVar[bool] = True
    plot_kinds: ClassVar[dict[str, bool]] = {"drop_log": False}

    def __init__(
        self, duration: float = 2.0, *, overlap: float = 0.0, reject_by_annotation: bool = True
    ) -> None:
        self.duration = duration
        self.overlap = overlap
        self.reject_by_annotation = reject_by_annotation

    def _fit(self, inst: BaseRaw) -> None:
        if not 0 <= self.overlap < self.duration:
            raise ValueError(
                f"overlap must be at least 0 and below duration ({self.duration}), "
                f"got {self.overlap}."
            )

    def _transform(self, inst: BaseRaw) -> BaseEpochs:
        epochs = mne.make_fixed_length_epochs(
            inst,
            duration=self.duration,
            overlap=self.overlap,
            reject_by_annotation=self.reject_by_annotation,
            preload=True,
            verbose=False,
        )
        n_total = len(epochs.drop_log)
        self.drop_log_ = epochs.drop_log
        self.qc_.update(
            n_epochs=n_total,
            n_kept=len(epochs),
            fraction_kept=round(len(epochs) / n_total, 4) if n_total else 0.0,
            duration_kept_s=round(len(epochs) * (self.duration - self.overlap), 3),
        )
        return epochs

    def _plot_unavailable(self, kind: str) -> str | None:
        if not hasattr(self, "drop_log_"):
            return "the step has not transformed data yet (use fit_transform)"
        return None

    def _plot_drop_log(self, inst: Any) -> Any:
        return mne.viz.plot_drop_log(self.drop_log_, show=False)


class Baseline(Step):
    """Baseline correction of epochs or evoked data.

    Parameters
    ----------
    baseline : tuple
        Baseline interval ``(start, end)`` in seconds; ``None`` means the
        start or end of the epoch.
    mode : {"mean", "ratio", "logratio", "percent", "zscore", "zlogratio"}
        ``"mean"`` subtracts the baseline mean (:meth:`mne.Epochs.apply_baseline`);
        the others are :func:`mne.baseline.rescale`.
    scale : {"epoch", "pooled"}
        For ``"zscore"``: divide by each epoch's own baseline standard
        deviation (MNE), or by the standard deviation of the baseline pooled
        over all epochs (per channel), which is far more precise for short
        baselines.
    max_std_error : float
        Per-epoch z-scoring is refused when the baseline is too short to
        estimate a standard deviation: with *n* samples its relative standard
        error is about ``1 / sqrt(2 (n - 1))`` (for Gaussian noise). The
        default, 10 %, requires at least 51 samples.

    Attributes
    ----------
    qc_ : dict
        ``mode``, ``window`` (s) and ``n_samples`` in the baseline.
    """

    accepts: ClassVar[tuple[type, ...]] = (BaseEpochs, Evoked)

    def __init__(
        self,
        baseline: tuple[float | None, float | None] = (None, 0.0),
        mode: str = "mean",
        *,
        scale: str = "epoch",
        max_std_error: float = 0.1,
    ) -> None:
        self.baseline = baseline
        self.mode = mode
        self.scale = scale
        self.max_std_error = max_std_error

    def _fit(self, inst: Inst) -> None:
        if self.mode not in BASELINE_MODES:
            raise ValueError(f"mode must be one of {BASELINE_MODES}, got {self.mode!r}")
        if self.scale not in ("epoch", "pooled"):
            raise ValueError(f"scale must be 'epoch' or 'pooled', got {self.scale!r}")
        if self.scale == "pooled" and self.mode != "zscore":
            raise ValueError("scale='pooled' only applies to mode='zscore'.")
        start, stop = _window(self.baseline, inst.times)
        n = int(((inst.times >= start) & (inst.times <= stop)).sum())
        if n < 2:
            raise ValueError(f"The baseline {self.baseline} contains {n} sample(s).")
        if self.mode in ("zscore", "zlogratio") and self.scale == "epoch":
            error = 1 / np.sqrt(2 * (n - 1))
            if error > self.max_std_error:
                raise ValueError(
                    f"The baseline has {n} samples, so each epoch's standard deviation is "
                    f"estimated with about {error:.0%} error (max_std_error="
                    f"{self.max_std_error:g}). Use mode='mean', a longer baseline, or "
                    "scale='pooled'."
                )
        self.qc_.update(mode=self.mode, window=[float(start), float(stop)], n_samples=n)

    def _transform(self, inst: Inst) -> Inst:
        baseline = tuple(self.baseline)
        if self.mode == "mean":
            return inst.apply_baseline(baseline, verbose=False)
        picks = mne.pick_types(
            inst.info, meg=True, eeg=True, eog=True, ecg=True, emg=True, seeg=True, ecog=True,
            dbs=True, ref_meg=False, exclude=[],
        )  # fmt: skip
        data = inst._data
        if self.mode == "zscore" and self.scale == "pooled":
            start, stop = _window(baseline, inst.times)
            sel = (inst.times >= start) & (inst.times <= stop)
            x = data[:, picks] if data.ndim == 3 else data[picks][np.newaxis]
            mean = x[..., sel].mean(axis=-1, keepdims=True)
            sd = (x[..., sel] - mean).std(axis=(0, 2), ddof=1, keepdims=True)
            out = (x - mean) / sd
            if data.ndim == 3:
                data[:, picks] = out
            else:
                data[picks] = out[0]
        else:
            data[..., picks, :] = mne.baseline.rescale(
                data[..., picks, :], inst.times, baseline, mode=self.mode, copy=True, verbose=False
            )
        return inst


def _window(baseline: Any, times: np.ndarray) -> tuple[float, float]:
    start, stop = baseline
    return (times[0] if start is None else start, times[-1] if stop is None else stop)


def _select_events(found: dict[str, int], event_id: Any) -> dict[str, int]:
    if not found:
        raise ValueError("No events found in the data.")
    if event_id is None:
        return dict(found)
    if isinstance(event_id, dict):
        return {str(n): int(c) for n, c in event_id.items()}
    names = [event_id] if isinstance(event_id, str) else list(event_id)
    selected = mne.event.match_event_names(found, names, on_missing="ignore")
    if not selected:
        raise ValueError(f"No events match {names}; found {sorted(found)}.")
    return {n: found[n] for n in selected}


def _metadata(spec: str, raw: BaseRaw, events: np.ndarray) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - pandas is optional
        raise ImportError("Epoch(metadata=...) needs pandas: pip install pandas") from exc
    if spec == "bids":
        from ..io.read import as_bids_path

        fname = raw.filenames[0] if raw.filenames else None
        bids_path = as_bids_path(Path(fname)) if fname is not None else None
        if bids_path is None:
            raise ValueError("metadata='bids' needs a recording inside a BIDS dataset.")
        tsv = Path(str(bids_path.copy().update(suffix="events", extension=".tsv").fpath))
        if not tsv.exists():
            raise FileNotFoundError(f"No events file {tsv}.")
        table = pd.read_csv(tsv, sep="\t", na_values="n/a")
        onsets = (events[:, 0] - raw.first_samp) / raw.info["sfreq"]
        tolerance = 1.5 / raw.info["sfreq"]
        rows = []
        for onset in onsets:
            match = np.flatnonzero(np.abs(table["onset"].to_numpy() - onset) <= tolerance)
            if len(match) == 0:
                raise ValueError(f"No row of {tsv.name} at onset {onset:.3f} s.")
            rows.append(match[0])
        return table.iloc[rows].reset_index(drop=True)
    path = Path(spec)
    table = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    if len(table) != len(events):
        raise ValueError(f"{path.name} has {len(table)} rows for {len(events)} events.")
    return table
