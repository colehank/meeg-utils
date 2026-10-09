"""Annotation of bad data segments."""

from __future__ import annotations

from typing import Any, ClassVar

import mne
import numpy as np
from mne.io import BaseRaw

from ..core import Step
from ._utils import DATA_CH_TYPES

SEGMENT_METHODS = ("amplitude", "flat", "muscle")


class BadSegments(Step):
    """Mark bad stretches of a recording with ``BAD_*`` annotations.

    The data are not changed. Later steps skip annotated segments: ICA
    fitting, epoching (``reject_by_annotation``) and most MNE functions.

    Methods (any combination):

    - ``"amplitude"``: the recording is cut into windows of ``window``
      seconds and a window is bad if the peak-to-peak amplitude of any good
      channel exceeds its type's threshold. ``reject="auto"`` (default)
      learns one threshold per channel type from the data, as autoreject's
      global threshold (Jas et al., 2017, NeuroImage 159:417), in ``fit``;
      or pass thresholds, e.g. ``{"eeg": 150e-6}``. The learned threshold
      also flags blinks and heartbeats; before ICA, which removes them,
      pass looser thresholds so that only gross artifacts are marked.
    - ``"flat"``: a window is bad if a good channel's peak-to-peak amplitude
      is below ``flat`` (e.g. ``{"eeg": 1e-6}``), e.g. amplifier saturation
      or disconnection.
    - ``"muscle"``: :func:`mne.preprocessing.annotate_muscle_zscore` (MNE
      defaults: 110-140 Hz envelope, z > 4, segments shorter than 0.1 s
      between bad ones also marked). Needs data up to 140 Hz.

    Parameters
    ----------
    methods : str | list of str
        Methods to apply.
    reject : "auto" | dict
        ``"amplitude"``: thresholds per channel type (peak-to-peak, SI units).
    flat : dict | None
        ``"flat"``: minimum peak-to-peak per channel type.
    window : float
        Window length (s) for ``"amplitude"`` and ``"flat"``.
    muscle_threshold : float
        ``"muscle"``: z-score threshold.
    muscle_freqs : tuple of float
        ``"muscle"``: band of the muscle envelope.
    random_state : int | None
        Seed for ``reject="auto"``.

    Attributes
    ----------
    reject_ : dict
        Peak-to-peak thresholds used for ``"amplitude"``.
    qc_ : dict
        Per method: ``n_segments``, ``duration_s`` and ``fraction``;
        ``fraction_bad`` (any method, existing ``BAD`` annotations included)
        and ``reject`` (thresholds).

    Notes
    -----
    Annotations are added with descriptions ``BAD_amplitude``,
    ``BAD_flat`` and ``BAD_muscle``; existing annotations are kept.
    Channels that exceed the thresholds most of the time should be marked
    bad first (:class:`BadChannels`), or they mark the whole recording.

    Figures (:meth:`plot`): ``"segments"``, the bad segments of each method
    over time with the window amplitudes and muscle z-scores.
    """

    accepts: ClassVar[tuple[type, ...]] = (BaseRaw,)
    adds_annotations: ClassVar[bool] = True
    plot_kinds: ClassVar[dict[str, bool]] = {"segments": False}

    def __init__(
        self,
        methods: str | list[str] | tuple[str, ...] = ("amplitude", "muscle"),
        *,
        reject: str | dict[str, float] = "auto",
        flat: dict[str, float] | None = None,
        window: float = 1.0,
        muscle_threshold: float = 4.0,
        muscle_freqs: tuple[float, float] = (110.0, 140.0),
        random_state: int | None = 42,
    ) -> None:
        self.methods = methods
        self.reject = reject
        self.flat = flat
        self.window = window
        self.muscle_threshold = muscle_threshold
        self.muscle_freqs = muscle_freqs
        self.random_state = random_state

    def _fit(self, inst: BaseRaw) -> None:
        methods = self._methods()
        if "flat" in methods and not self.flat:
            raise ValueError("method 'flat' needs thresholds, e.g. flat={'eeg': 1e-6}.")
        if "muscle" in methods:
            top = self.muscle_freqs[1]
            if min(inst.info["sfreq"] / 2, inst.info["lowpass"]) <= top:
                raise ValueError(
                    f"method 'muscle' needs data up to {top:g} Hz; the recording has a low-pass "
                    f"of {inst.info['lowpass']:g} Hz at {inst.info['sfreq']:g} Hz. Annotate muscle "
                    "before low-pass filtering or resampling, or drop the method."
                )
        self.reject_: dict[str, float] = {}
        if "amplitude" in methods:
            if self.reject == "auto":
                self.reject_ = _auto_reject(inst, self.window, self.random_state)
            elif isinstance(self.reject, dict):
                self.reject_ = {k: float(v) for k, v in self.reject.items()}
            else:
                raise ValueError(f"reject must be 'auto' or a dict, got {self.reject!r}.")
        self.qc_["reject"] = dict(self.reject_)

    def _transform(self, inst: BaseRaw) -> BaseRaw:
        methods = self._methods()
        sfreq = inst.info["sfreq"]
        new: list[tuple[np.ndarray, np.ndarray, str]] = []
        self.windows_: dict[str, Any] = {}
        if "amplitude" in methods or "flat" in methods:
            starts, ptp = _window_ptp(inst, self.window)
            self.windows_ = {"starts": starts.astype(np.float32), "ratio": {}}
            for method, limits, above in (
                ("amplitude", self.reject_ if "amplitude" in methods else {}, True),
                ("flat", (self.flat or {}) if "flat" in methods else {}, False),
            ):
                if method not in methods:
                    continue
                bad = np.zeros(len(starts), bool)
                for ch_type, values in ptp.items():
                    if ch_type not in limits:
                        continue
                    if above:
                        ratio = values.max(axis=0) / limits[ch_type]
                        self.windows_["ratio"][ch_type] = ratio.astype(np.float32)
                        bad |= ratio > 1
                    else:
                        bad |= values.min(axis=0) < limits[ch_type]
                onsets, durations = _merge(starts[bad], self.window)
                new.append((onsets, durations, f"BAD_{method}"))
        if "muscle" in methods:
            annot, scores = mne.preprocessing.annotate_muscle_zscore(
                inst, threshold=self.muscle_threshold, filter_freq=tuple(self.muscle_freqs),
                verbose=False,
            )  # fmt: skip
            bin_size = max(1, int(sfreq / 10))
            n = len(scores) // bin_size * bin_size
            self.muscle_scores_ = scores[:n].reshape(-1, bin_size).max(axis=1).astype(np.float32)
            onsets = np.asarray(annot.onset) - _offset(inst)  # relative to the data start
            new.append((onsets, np.asarray(annot.duration), "BAD_muscle"))

        total = inst.times[-1] + 1 / sfreq
        for onsets, durations, description in new:
            if len(onsets):
                inst.annotations.append(onsets + _offset(inst), durations, description)
            self.qc_[description.removeprefix("BAD_")] = {
                "n_segments": len(onsets),
                "duration_s": round(float(np.sum(durations)), 3),
                "fraction": round(float(np.sum(durations) / total), 4),
            }
        self.qc_["fraction_bad"] = round(_bad_fraction(inst), 4)
        self.duration_ = float(total)
        self.segments_ = {
            d.removeprefix("BAD_"): np.c_[o, o + dur].astype(np.float32) for o, dur, d in new
        }
        return inst

    def _methods(self) -> list[str]:
        methods = [self.methods] if isinstance(self.methods, str) else list(self.methods)
        unknown = sorted(set(methods) - set(SEGMENT_METHODS))
        if unknown or not methods:
            raise ValueError(f"methods must be among {SEGMENT_METHODS}, got {self.methods!r}.")
        return methods

    def _plot_unavailable(self, kind: str) -> str | None:
        if kind == "segments" and not hasattr(self, "segments_"):
            return "the step has not transformed data yet (use fit_transform)"
        return None

    def _plot_segments(self, inst: Any) -> Any:
        import matplotlib.pyplot as plt

        rows = list(self.segments_)
        has_ratio = bool(self.windows_.get("ratio"))
        has_muscle = hasattr(self, "muscle_scores_") and "muscle" in rows
        n_panels = 1 + has_ratio + has_muscle
        fig, axes = plt.subplots(
            n_panels, 1, figsize=(10, 1.2 + 1.6 * n_panels), sharex=True, squeeze=False,
            layout="constrained",
        )  # fmt: skip
        ax = axes[0, 0]
        for i, method in enumerate(rows):
            spans = self.segments_[method]
            ax.broken_barh(
                [(a, b - a) for a, b in spans], (i - 0.35, 0.7), color=f"C{3 + i}", alpha=0.8
            )
        ax.set(
            yticks=range(len(rows)),
            yticklabels=[f"{m} ({100 * self.qc_[m]['fraction']:.1f} %)" for m in rows],
            xlim=(0, self.duration_),
            ylim=(-0.6, len(rows) - 0.4),
            title=f"Bad segments: {100 * self.qc_['fraction_bad']:.1f} % of the recording",
        )
        panel = 1
        if has_ratio:
            ax = axes[panel, 0]
            starts = self.windows_["starts"]
            for ch_type, ratio in self.windows_["ratio"].items():
                ax.step(starts, ratio, where="post", lw=0.8, label=ch_type.upper())
            ax.axhline(1, color="C3", ls="--", lw=1)
            ax.set(yscale="log", ylabel="peak-to-peak /\nthreshold")
            ax.legend(frameon=False, fontsize="small", loc="upper right")
            panel += 1
        if has_muscle:
            ax = axes[panel, 0]
            times = np.arange(len(self.muscle_scores_)) * 0.1
            ax.plot(times, self.muscle_scores_, lw=0.6, color="C0")
            ax.axhline(self.muscle_threshold, color="C3", ls="--", lw=1)
            ax.set(ylabel="muscle z-score")
        axes[-1, 0].set_xlabel("Time (s, from the start of the data)")
        return fig


# ----------------------------------------------------------------------


def _window_ptp(raw: BaseRaw, window: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Peak-to-peak amplitude of good data channels in consecutive windows, per type."""
    n = round(window * raw.info["sfreq"])
    n_windows = raw.n_times // n
    if n_windows == 0:
        raise ValueError(f"The recording is shorter than one window ({window:g} s).")
    starts = np.arange(n_windows) * n / raw.info["sfreq"]
    ptp = {}
    types = raw.get_channel_types()
    good = [i for i, ch in enumerate(raw.ch_names) if ch not in raw.info["bads"]]
    for ch_type in DATA_CH_TYPES:
        picks = [i for i in good if types[i] == ch_type]
        if not picks:
            continue
        data = raw.get_data(picks)[:, : n_windows * n].reshape(len(picks), n_windows, n)
        ptp[ch_type] = data.max(axis=-1) - data.min(axis=-1)
    if not ptp:
        raise ValueError("No good MEG or EEG channels.")
    return starts, ptp


def _auto_reject(raw: BaseRaw, window: float, random_state: int | None) -> dict[str, float]:
    import autoreject

    epochs = mne.make_fixed_length_epochs(
        raw, duration=window, preload=True, reject_by_annotation=True, verbose=False
    )
    epochs.pick([t for t in DATA_CH_TYPES if t in epochs.get_channel_types(unique=True)])
    if len(epochs) < 20:
        raise ValueError(
            f"reject='auto' needs at least 20 clean windows of {window:g} s, got {len(epochs)}; "
            "pass thresholds instead, e.g. reject={'eeg': 150e-6}."
        )
    reject = autoreject.get_rejection_threshold(epochs, random_state=random_state, verbose=False)
    return {k: float(v) for k, v in reject.items() if k in DATA_CH_TYPES}


def _merge(starts: np.ndarray, length: float) -> tuple[np.ndarray, np.ndarray]:
    """Merge adjacent windows into segments."""
    if not len(starts):
        return np.empty(0), np.empty(0)
    onsets, ends = [starts[0]], [starts[0] + length]
    for start in starts[1:]:
        if start <= ends[-1] + 1e-9:
            ends[-1] = start + length
        else:
            onsets.append(start)
            ends.append(start + length)
    onsets_arr = np.asarray(onsets)
    return onsets_arr, np.asarray(ends) - onsets_arr


def _bad_fraction(raw: BaseRaw) -> float:
    """Fraction of samples inside any BAD annotation."""
    mask = np.zeros(raw.n_times, bool)
    for annot in raw.annotations:
        if not annot["description"].upper().startswith("BAD"):
            continue
        start = round((annot["onset"] - _offset(raw)) * raw.info["sfreq"])
        stop = start + round(annot["duration"] * raw.info["sfreq"])
        mask[max(start, 0) : max(stop, 0)] = True
    return float(mask.mean())


def _offset(raw: BaseRaw) -> float:
    """Annotation time of the first sample (onsets count from meas_date when it is set)."""
    return raw.first_time if raw.annotations.orig_time is not None else 0.0
