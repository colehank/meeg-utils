"""Signal-quality checks for EEG and MEG: amplitude, muscle, outliers, spectra, EOG, ECG."""

from __future__ import annotations

from typing import Any, ClassVar

import mne
import numpy as np
from mne.io import BaseRaw

from ..steps._plotting import UNITS, psd_summary
from ..steps._utils import picks_by_type
from ._base import Check


class Amplitude(Check):
    """Flat signals, clipping (amplifier saturation) and missing samples.

    - **Flat**: a channel repeats exactly the same value for at least
      ``min_flat_duration``; channels flat for more than ``bad_percent`` of
      the recording are flat channels (disconnected, broken), the rest are
      flat segments (dropouts).
    - **Clipped**: the signal stays at its own maximum or minimum for at
      least ``min_clip_samples`` samples, which is what an amplifier or
      SQUID at the end of its range produces.
    - **Missing**: NaN samples (:func:`mne.preprocessing.annotate_nan`).

    Exact repetition is used instead of an amplitude threshold, so no
    unit- or system-specific threshold is needed.

    Parameters
    ----------
    min_flat_duration : float
        Shortest run of identical samples counted as flat, in seconds. The
        default (50 ms) is far longer than chance repetitions of quantized
        noise.
    min_clip_samples : int
        Shortest run at the extreme value counted as clipping.
    bad_percent : float
        Percentage of time above which a channel is a flat channel (the
        default of :func:`mne.preprocessing.annotate_amplitude`, 5 %).

    Attributes
    ----------
    metrics_ : dict
        ``flat_channels``, ``clipped_channels`` and ``nan_channels``;
        ``flat_segments_s`` (flat time in the other channels) and, per
        channel type, the percentage of time each affected channel is flat,
        clipped or NaN.

    Notes
    -----
    Figures: ``"channels"`` (percentage of time flat / clipped / NaN for
    every affected channel).
    """

    name: ClassVar[str] = "amplitude"
    plot_kinds: ClassVar[dict[str, bool]] = {"channels": False}

    def __init__(
        self,
        *,
        min_flat_duration: float = 0.05,
        min_clip_samples: int = 3,
        bad_percent: float = 5.0,
    ) -> None:
        self.min_flat_duration = min_flat_duration
        self.min_clip_samples = min_clip_samples
        self.bad_percent = bad_percent

    def _compute(self, raw: BaseRaw) -> None:
        sfreq = raw.info["sfreq"]
        min_flat = max(2, round(self.min_flat_duration * sfreq))
        per_type: dict[str, dict[str, dict[str, float]]] = {}
        flat_channels, clipped_channels, nan_channels = [], [], []
        flat_segments = 0.0
        for ch_type, picks in picks_by_type(raw.info, exclude_bads=False).items():
            data = raw.get_data(picks=picks)
            affected: dict[str, dict[str, float]] = {}
            for name, x in zip((raw.ch_names[p] for p in picks), data, strict=True):
                nan = float(np.isnan(x).mean() * 100)
                flat, clip = _plateau_percent(x, min_flat, self.min_clip_samples)
                if flat > self.bad_percent:
                    flat_channels.append(name)
                    clip = 0.0  # a flat channel sits at its own extreme everywhere
                else:
                    flat_segments += flat / 100 * raw.times[-1]
                if clip > 0:
                    clipped_channels.append(name)
                if nan > 0:
                    nan_channels.append(name)
                if flat or clip or nan:
                    affected[name] = {
                        "flat_pct": round(flat, 3),
                        "clipped_pct": round(clip, 3),
                        "nan_pct": round(nan, 3),
                    }
            per_type[ch_type] = affected
        self.metrics_.update(
            flat_channels=flat_channels,
            clipped_channels=clipped_channels,
            nan_channels=nan_channels,
            flat_segments_s=round(flat_segments, 3),
            channels=per_type,
        )
        self._judge(
            "n_flat_channels",
            len(flat_channels),
            warn=0,
            what=f"channels flat for more than {self.bad_percent:g} % of the time",
            detail=", ".join(flat_channels[:15]),
        )
        self._judge(
            "flat_segments_s",
            round(flat_segments, 3),
            warn=0,
            what="flat segments in the other channels",
            unit=" s",
            detail="signal dropouts",
        )
        self._judge(
            "n_clipped_channels",
            len(clipped_channels),
            warn=0,
            what="clipped channels",
            detail="amplifier range exceeded in " + ", ".join(clipped_channels[:15]),
        )
        self._judge(
            "n_nan_channels",
            len(nan_channels),
            warn=0,
            what="channels with missing (NaN) samples",
            detail=", ".join(nan_channels[:15]),
        )

    def _plot_channels(self, inst: Any) -> Any:
        import matplotlib.pyplot as plt

        types = list(self.metrics_["channels"])
        fig, axes = plt.subplots(
            len(types), 1, figsize=(8, 2.6 * len(types)), squeeze=False, layout="constrained"
        )
        for ax, ch_type in zip(axes[:, 0], types, strict=True):
            affected = self.metrics_["channels"][ch_type]
            names = sorted(affected, key=lambda n: -max(affected[n].values()))[:40]
            if not names:
                ax.text(0.5, 0.5, "no flat, clipped or NaN samples", ha="center", va="center")
                ax.set_axis_off()
                ax.set_title(ch_type.upper())
                continue
            x = np.arange(len(names))
            for i, (key, label, color) in enumerate(
                (
                    ("flat_pct", "flat", "0.4"),
                    ("clipped_pct", "clipped", "C3"),
                    ("nan_pct", "NaN", "C1"),
                )
            ):
                values = np.array([affected[n][key] for n in names])
                bars = ax.bar(x + (i - 1) * 0.27, np.where(values > 0, values, np.nan), 0.27,
                              label=label, color=color)  # fmt: skip
                for bar, v in zip(bars, values, strict=True):
                    if v > 0:
                        ax.annotate(f"{v:.3g}", (bar.get_x() + bar.get_width() / 2, v),
                                    ha="center", va="bottom", fontsize=6)  # fmt: skip
            ax.axhline(self.bad_percent, color="0.4", ls="--", lw=1)
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 300)
            ax.set_xticks(x, names, rotation=90, fontsize=7)
            ax.set(ylabel="% of recording (log)", title=ch_type.upper())
            ax.legend(frameon=False, fontsize="small", ncols=3, loc="upper right")
        return fig


class Muscle(Check):
    """Share of the recording contaminated by muscle activity.

    Uses :func:`mne.preprocessing.annotate_muscle_zscore` with MNE's defaults:
    the 110-140 Hz envelope, z-scored and averaged over channels, above 4.

    Parameters
    ----------
    threshold : float
        z-score threshold (MNE default 4).
    filter_freq : tuple of float
        Band in which muscle activity is measured (MNE default 110-140 Hz).
    ch_type : str | None
        Channel type to use; ``None`` lets MNE choose (mag, then grad, then eeg).
    warn_fraction : float
        Fraction of time above which the recording is flagged. There is no
        established limit; the default is 10 %.

    Attributes
    ----------
    metrics_ : dict
        ``fraction`` (of the recording), ``n_segments`` and ``ch_type``.

    Notes
    -----
    Figures: ``"scores"`` (muscle z-score over time with the threshold).
    """

    name: ClassVar[str] = "muscle"
    plot_kinds: ClassVar[dict[str, bool]] = {"scores": False}

    def __init__(
        self,
        *,
        threshold: float = 4.0,
        filter_freq: tuple[float, float] = (110.0, 140.0),
        ch_type: str | None = None,
        warn_fraction: float = 0.1,
    ) -> None:
        self.threshold = threshold
        self.filter_freq = filter_freq
        self.ch_type = ch_type
        self.warn_fraction = warn_fraction

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
        top = self.filter_freq[1]
        if reason is None and min(raw.info["sfreq"] / 2, raw.info["lowpass"]) <= top:
            reason = f"needs data up to {top:g} Hz (sampling rate or low-pass too low)"
        return reason

    def _compute(self, raw: BaseRaw) -> None:
        annot, scores = mne.preprocessing.annotate_muscle_zscore(
            raw,
            threshold=self.threshold,
            ch_type=self.ch_type,
            filter_freq=tuple(self.filter_freq),
            verbose=False,
        )
        duration = raw.times[-1] + 1 / raw.info["sfreq"]
        fraction = float(np.sum(annot.duration) / duration)
        ch_type = self.ch_type or next(
            t for t in ("mag", "grad", "eeg") if t in raw.get_channel_types(unique=True)
        )
        bin_size = max(1, int(raw.info["sfreq"] / 10))  # keep the maximum of every 0.1 s
        n = len(scores) // bin_size * bin_size
        self.scores_ = scores[:n].reshape(-1, bin_size).max(axis=1).astype(np.float32)
        self.score_times_ = (np.arange(len(self.scores_)) * bin_size / raw.info["sfreq"]).astype(
            np.float32
        )
        self.metrics_.update(fraction=round(fraction, 4), n_segments=len(annot), ch_type=ch_type)
        self._judge(
            "fraction",
            round(fraction, 4),
            warn=self.warn_fraction,
            what="fraction of time with muscle activity",
            detail="ask the participant to relax jaw, face and neck",
        )

    def _plot_scores(self, inst: Any) -> Any:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 3), layout="constrained")
        ax.plot(self.score_times_, self.scores_, lw=0.7, color="C0")
        ax.axhline(self.threshold, color="C3", ls="--", lw=1, label=f"threshold {self.threshold:g}")
        ax.fill_between(
            self.score_times_,
            0,
            self.scores_,
            where=self.scores_ > self.threshold,
            color="C3",
            alpha=0.3,
        )
        ax.set(
            xlabel="Time (s)",
            ylabel="z-score",
            title=(
                f"Muscle activity ({self.metrics_['ch_type'].upper()}, "
                f"{self.filter_freq[0]:g}-{self.filter_freq[1]:g} Hz): "
                f"{100 * self.metrics_['fraction']:.1f} % of the recording"
            ),
        )
        ax.legend(frameon=False, loc="upper right")
        return fig


class OutlierChannels(Check):
    """Channels whose signal is unlike that of the other channels (LOF).

    Uses :func:`mne.preprocessing.find_bad_channels_lof` (Local Outlier
    Factor; Kumaravel et al., 2022) on each channel type separately, after a
    1 Hz high-pass on a copy so that slow drifts do not dominate. It
    complements PREP and Maxwell detection and works for any system.

    Parameters
    ----------
    n_neighbors : int
        LOF neighbourhood size (MNE default 20); reduced to the number of
        channels minus one for small montages.
    threshold : float
        LOF score above which a channel is an outlier (MNE default 1.5).
    l_freq : float | None
        High-pass applied to the copy before scoring.
    min_channels : int
        Channel types with fewer channels are skipped.

    Attributes
    ----------
    metrics_ : dict
        Per channel type, the ``outliers``; ``scores`` per channel.

    Notes
    -----
    Figures: ``"scores"`` (LOF score per channel with the threshold).
    """

    name: ClassVar[str] = "outlier_channels"
    plot_kinds: ClassVar[dict[str, bool]] = {"scores": False}

    def __init__(
        self,
        *,
        n_neighbors: int = 20,
        threshold: float = 1.5,
        l_freq: float | None = 1.0,
        min_channels: int = 10,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.threshold = threshold
        self.l_freq = l_freq
        self.min_channels = min_channels

    def _compute(self, raw: BaseRaw) -> None:
        filtered = raw.copy()
        if self.l_freq is not None:
            filtered.filter(self.l_freq, None, picks="data", verbose=False)
        scores: dict[str, dict[str, float]] = {}
        outliers: dict[str, list[str]] = {}
        for ch_type, picks in picks_by_type(raw.info).items():  # MNE skips marked bads
            if len(picks) < self.min_channels:
                continue
            _, type_scores = mne.preprocessing.find_bad_channels_lof(
                filtered,
                n_neighbors=min(self.n_neighbors, len(picks) - 1),
                picks=picks,
                threshold=self.threshold,
                return_scores=True,
                verbose=False,
            )
            names = [raw.ch_names[p] for p in picks]
            scores[ch_type] = {  # MNE returns sklearn's negative outlier factor
                n: round(abs(float(s)), 4) for n, s in zip(names, type_scores, strict=True)
            }
            outliers[ch_type] = [n for n in names if scores[ch_type][n] > self.threshold]
        if not scores:
            raise ValueError(f"No channel type has at least {self.min_channels} channels.")
        self.metrics_.update(outliers=outliers, scores=scores)
        for ch_type, names in outliers.items():
            self._judge(
                f"n_outliers_{ch_type}",
                len(names),
                warn=0,
                what=f"outlying {ch_type.upper()} channels (LOF > {self.threshold:g})",
                detail=", ".join(names[:15]),
            )

    def _plot_scores(self, inst: Any) -> Any:
        import matplotlib.pyplot as plt

        types = list(self.metrics_["scores"])
        fig, axes = plt.subplots(
            len(types), 1, figsize=(10, 2.8 * len(types)), squeeze=False, layout="constrained"
        )
        for ax, ch_type in zip(axes[:, 0], types, strict=True):
            scores = self.metrics_["scores"][ch_type]
            names = list(scores)
            bad = set(self.metrics_["outliers"][ch_type])
            ax.bar(
                range(len(names)),
                list(scores.values()),
                color=["C3" if n in bad else "C0" for n in names],
            )
            ax.axhline(self.threshold, color="C3", ls="--", lw=1)
            if len(names) <= 80:
                ax.set_xticks(range(len(names)), names, rotation=90, fontsize=6)
                for label in ax.get_xticklabels():
                    if label.get_text() in bad:
                        label.set(color="C3", fontweight="bold")
            ax.set(ylabel="LOF score", title=f"{ch_type.upper()}: outlying channels {sorted(bad)}")
        return fig


class NarrowbandNoise(Check):
    """Power-line noise and other narrowband noise in the raw spectrum.

    Narrowband peaks are found with the ZapLine-plus detector (Klug &
    Kloosterman, 2022; :func:`mne_denoise.zapline.adaptive.find_noise_freqs`):
    a peak of the channel-averaged log spectrum more than ``threshold_db``
    above the mean of the outer thirds of a ``window`` Hz neighbourhood.
    Peaks at the line frequency or its harmonics are line noise, which is
    reported but not flagged (it is removed in preprocessing); other peaks
    point to equipment in or near the recording room. Channels picking up
    much more line noise than the others usually have poor contact.

    Parameters
    ----------
    fline : float | None
        Power-line frequency; ``None`` uses ``info["line_freq"]``.
    fmin, fmax : float | None
        Search range in Hz (ZapLine-plus default 17-99 Hz; ``fmax=None``
        searches up to the low-pass or 95 % of the Nyquist frequency).
    threshold_db : float
        Peak height above the local baseline (ZapLine-plus default 4 dB).
    window : float
        Width of the local baseline in Hz (ZapLine-plus default 6 Hz).
    tolerance : float
        Peaks within this distance (Hz) of a harmonic are line noise.
    outlier_z : float
        Robust z-score (median / MAD) of a channel's line-noise amplitude
        relative to its noise floor above which it is flagged (the threshold
        PREP uses for its noise criteria, 5).

    Attributes
    ----------
    metrics_ : dict
        Per channel type: ``line_db`` (median peak height of each harmonic),
        ``other_peaks`` (Hz) and ``line_outliers`` (channels).

    Notes
    -----
    Figures: ``"psd"`` (spectra with line harmonics and other peaks marked)
    and ``"line_channels"`` (line-noise level of every channel).
    """

    name: ClassVar[str] = "narrowband_noise"
    plot_kinds: ClassVar[dict[str, bool]] = {"psd": False, "line_channels": False}

    def __init__(
        self,
        *,
        fline: float | None = None,
        fmin: float = 17.0,
        fmax: float | None = None,
        threshold_db: float = 4.0,
        window: float = 6.0,
        tolerance: float = 0.5,
        outlier_z: float = 5.0,
    ) -> None:
        self.fline = fline
        self.fmin = fmin
        self.fmax = fmax
        self.threshold_db = threshold_db
        self.window = window
        self.tolerance = tolerance
        self.outlier_z = outlier_z

    def _compute(self, raw: BaseRaw) -> None:
        from mne_denoise.zapline.adaptive import find_noise_freqs

        sfreq = raw.info["sfreq"]
        fline = self.fline if self.fline is not None else raw.info["line_freq"]
        fmax = self.fmax
        if fmax is None:
            fmax = min(0.95 * sfreq / 2, raw.info["lowpass"] or sfreq / 2)
        harmonics = [] if fline is None else list(np.arange(fline, fmax + self.tolerance, fline))
        if fline is None:
            self._note(
                "fline", None, "power-line frequency unknown: peaks are not classified", "warn"
            )

        self.psd_: dict[str, dict[str, Any]] = {}
        result: dict[str, dict[str, Any]] = {}
        for ch_type, picks in picks_by_type(raw.info).items():
            data = raw.get_data(picks=picks)
            live = data.std(axis=1) > 0
            names = [raw.ch_names[p] for p, ok in zip(picks, live, strict=True) if ok]
            data = data[live]
            if len(data) == 0:
                continue
            peaks = find_noise_freqs(
                data, sfreq, fmin=self.fmin, fmax=fmax, window_length=self.window,
                threshold_factor=self.threshold_db,
            )  # fmt: skip
            freqs, psd = psd_summary(data, sfreq)
            line_db = {
                f"{h:g}": round(float(np.median(_peak_height_db(freqs, psd, h, self.window))), 2)
                for h in harmonics
            }
            other = [round(float(f), 2) for f in peaks if not _near(f, harmonics, self.tolerance)]
            outliers: list[str] = []
            channel_line_db: dict[str, float] = {}
            if harmonics:
                heights = _peak_height_db(freqs, psd, harmonics[0], self.window)
                channel_line_db = {
                    n: round(float(v), 2) for n, v in zip(names, heights, strict=True)
                }
                z = _robust_z(10 ** (heights / 20))  # amplitude ratio, as PREP's criteria
                outliers = [n for n, zi in zip(names, z, strict=True) if zi > self.outlier_z]
            result[ch_type] = {
                "line_db": line_db,
                "other_peaks": other,
                "line_outliers": outliers,
                "channel_line_db": channel_line_db,
            }
            self.psd_[ch_type] = {"freqs": freqs, "psd": psd, "peaks": [float(p) for p in peaks]}
            if harmonics:
                self._note(
                    f"line_db_{ch_type}",
                    line_db[f"{harmonics[0]:g}"],
                    f"{ch_type.upper()} line noise at {harmonics[0]:g} Hz: "
                    f"{line_db[f'{harmonics[0]:g}']:.1f} dB above the floor (median channel)",
                )
                self._judge(
                    f"n_line_outliers_{ch_type}",
                    len(outliers),
                    warn=0,
                    what=f"{ch_type.upper()} channels with unusually strong line noise",
                    detail="check their contact: " + ", ".join(outliers[:15]),
                )
            self._judge(
                f"n_other_peaks_{ch_type}",
                len(other),
                warn=0,
                what=f"{ch_type.upper()} narrowband peaks other than line noise",
                detail="at " + ", ".join(f"{f:g} Hz" for f in other[:10]),
            )
        self.harmonics_ = harmonics
        self.metrics_.update(fline=fline, channel_types=result)

    def _plot_psd(self, inst: Any) -> Any:
        import matplotlib.pyplot as plt

        types = list(self.psd_)
        fig, axes = plt.subplots(
            1, len(types), figsize=(max(7.0, 5 * len(types)), 3.6), squeeze=False,
            layout="constrained",
        )  # fmt: skip
        for ax, ch_type in zip(axes[0], types, strict=True):
            spec = self.psd_[ch_type]
            scale, unit = UNITS.get(ch_type, (1.0, "a.u."))
            db = 10 * np.log10(np.maximum(spec["psd"] * scale**2, 1e-30))
            lo, mid, hi = np.percentile(db, [5, 50, 95], axis=0)
            ax.plot(spec["freqs"], mid, color="C0", lw=1)
            ax.fill_between(spec["freqs"], lo, hi, color="C0", alpha=0.15, lw=0)
            for h in self.harmonics_:
                ax.axvline(h, color="C3", ls=":", lw=0.8)
            other = self.metrics_["channel_types"][ch_type]["other_peaks"]
            if other:
                ax.plot(
                    other,
                    np.interp(other, spec["freqs"], hi) + 3,
                    "v",
                    color="C1",
                    label="other peaks",
                )
                ax.legend(frameon=False, fontsize="small")
            label = f"({unit})²/Hz" if "/" in unit else f"{unit}²/Hz"
            ax.set(
                xscale="log",
                xlim=(max(spec["freqs"][1], 1.0), spec["freqs"][-1]),
                xlabel="Frequency (Hz)",
                ylabel=f"PSD (dB re 1 {label})",
                title=ch_type.upper(),
            )
        fig.suptitle("Raw spectrum: line harmonics (red dotted), other peaks (orange)")
        return fig

    def _plot_line_channels(self, inst: Any) -> Any:
        import matplotlib.pyplot as plt

        types = [t for t, r in self.metrics_["channel_types"].items() if r["channel_line_db"]]
        fig, axes = plt.subplots(
            max(1, len(types)), 1, figsize=(10, 2.8 * max(1, len(types))), squeeze=False,
            layout="constrained",
        )  # fmt: skip
        if not types:
            axes[0, 0].text(0.5, 0.5, "power-line frequency unknown", ha="center", va="center")
            axes[0, 0].set_axis_off()
        for ax, ch_type in zip(axes[:, 0], types, strict=False):
            result = self.metrics_["channel_types"][ch_type]
            values = result["channel_line_db"]
            bad = set(result["line_outliers"])
            names = list(values)
            ax.bar(
                range(len(names)),
                list(values.values()),
                color=["C3" if n in bad else "C0" for n in names],
            )
            if len(names) <= 80:
                ax.set_xticks(range(len(names)), names, rotation=90, fontsize=6)
            ax.set(
                ylabel="dB above floor",
                title=f"{ch_type.upper()} line noise at {self.harmonics_[0]:g} Hz per channel",
            )
        return fig


class Blinks(Check):
    """Blink rate from the EOG channel(s), a test that the EOG works.

    Blinks are detected with :func:`mne.preprocessing.find_eog_events` (MNE
    defaults). Spontaneous blink rates of healthy adults are around 17 per
    minute at rest and fall to about 4.5 per minute while reading
    (Bentivoglio et al., 1997, Mov Disord 12:1028); a much lower rate
    usually means a detached or misplaced EOG electrode.

    Parameters
    ----------
    min_rate : float
        Blinks per minute below which the EOG is flagged.

    Attributes
    ----------
    metrics_ : dict
        ``n_blinks``, ``rate_per_min`` and ``channels``.

    Notes
    -----
    Figures: ``"blinks"`` (blink rate over time and the average blink).
    """

    name: ClassVar[str] = "blinks"
    plot_kinds: ClassVar[dict[str, bool]] = {"blinks": False}

    def __init__(self, *, min_rate: float = 2.0) -> None:
        self.min_rate = min_rate

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
        if "eog" not in raw.get_channel_types(unique=True):
            return "needs an EOG channel"
        return None

    def _compute(self, raw: BaseRaw) -> None:
        eog = mne.pick_types(raw.info, eog=True, exclude=[])
        live = [
            raw.ch_names[i]
            for i, x in zip(eog, raw.get_data(picks=eog), strict=True)
            if x.std() > 0
        ]
        sfreq = raw.info["sfreq"]
        if live:
            events = mne.preprocessing.find_eog_events(raw, ch_name=live, verbose=False)
        else:  # MNE's detector fails on a constant signal
            events = np.zeros((0, 3), int)
        minutes = (raw.times[-1] + 1 / sfreq) / 60
        times = (events[:, 0] - raw.first_samp) / sfreq
        self.blink_times_ = times.astype(np.float32)
        self.duration_ = float(raw.times[-1])
        self.average_ = _event_average(
            raw, eog[0], events[:, 0] - raw.first_samp, (-0.5, 0.5), (1.0, 10.0)
        )
        rate = len(events) / minutes
        self.metrics_.update(
            n_blinks=len(events),
            rate_per_min=round(rate, 2),
            channels=[raw.ch_names[i] for i in eog],
            flat_channels=[raw.ch_names[i] for i in eog if raw.ch_names[i] not in live],
        )
        self._judge(
            "rate_per_min",
            round(rate, 2),
            warn=self.min_rate,
            below=True,
            what="blink rate",
            unit="/min",
            detail="check the EOG electrodes" + (" (EOG signal is flat)" if not live else ""),
        )

    def _plot_blinks(self, inst: Any) -> Any:
        return _rate_figure(
            self.blink_times_, self.duration_, self.average_, "Blinks", "/min",
            f"{self.metrics_['n_blinks']} blinks, {self.metrics_['rate_per_min']:.1f}/min",
            "EOG (µV)", 1e6,
        )  # fmt: skip


class HeartRate(Check):
    """Heart rate from the ECG channel (or MEG magnetometers), a test of the ECG.

    Heartbeats are detected with :func:`mne.preprocessing.find_ecg_events`
    (MNE defaults; without an ECG channel MNE builds a synthetic ECG from
    the MEG magnetometers). A resting heart rate outside 40-120 bpm usually
    means the detection failed, i.e. the ECG electrodes are poorly placed;
    the normal adult range is 60-100 bpm, and well-trained people can be
    below 60.

    Parameters
    ----------
    min_bpm, max_bpm : float
        Plausible range of the detected heart rate.

    Attributes
    ----------
    metrics_ : dict
        ``heart_rate_bpm``, ``n_beats``, ``rr_sd_ms`` and ``channel``
        (``"synthetic"`` when derived from MEG).

    Notes
    -----
    Figures: ``"heartbeats"`` (heart rate over time and the average beat).
    """

    name: ClassVar[str] = "heart_rate"
    plot_kinds: ClassVar[dict[str, bool]] = {"heartbeats": False}

    def __init__(self, *, min_bpm: float = 40.0, max_bpm: float = 120.0) -> None:
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm

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
        types = raw.get_channel_types(unique=True)
        if "ecg" not in types and "mag" not in types:
            return "needs an ECG channel (or MEG magnetometers)"
        return None

    def _compute(self, raw: BaseRaw) -> None:
        events, ch, _, ecg = mne.preprocessing.find_ecg_events(raw, return_ecg=True, verbose=False)
        sfreq = raw.info["sfreq"]
        samples = events[:, 0] - raw.first_samp
        rr = np.diff(samples) / sfreq
        bpm = 60 / float(np.median(rr)) if len(rr) else 0.0
        channel = raw.ch_names[ch] if ch is not None else "synthetic"
        self.beat_times_ = (samples / sfreq).astype(np.float32)
        self.duration_ = float(raw.times[-1])
        self.average_ = _array_average(np.ravel(ecg), samples, sfreq, (-0.3, 0.5))
        self.metrics_.update(
            heart_rate_bpm=round(bpm, 1),
            n_beats=len(events),
            rr_sd_ms=round(float(np.std(rr) * 1000), 1) if len(rr) else None,
            channel=channel,
        )
        self._judge(
            "heart_rate_low",
            round(bpm, 1),
            warn=self.min_bpm,
            below=True,
            what="heart rate",
            unit=" bpm",
            detail="ECG detection probably failed; check the ECG electrodes",
        )
        self._judge(
            "heart_rate_high",
            round(bpm, 1),
            warn=self.max_bpm,
            what="heart rate",
            unit=" bpm",
            detail="ECG detection probably failed; check the ECG electrodes",
        )

    def _plot_heartbeats(self, inst: Any) -> Any:
        return _rate_figure(
            self.beat_times_, self.duration_, self.average_, "Heart rate", " bpm",
            f"{self.metrics_['n_beats']} beats, {self.metrics_['heart_rate_bpm']:.0f} bpm "
            f"({self.metrics_['channel']})",
            "ECG (a.u.)", 1.0,
        )  # fmt: skip


# ----------------------------------------------------------------------


def _plateau_percent(x: np.ndarray, min_flat: int, min_clip: int) -> tuple[float, float]:
    """Percentage of samples in flat runs and in clipped runs (at the extremes)."""
    same = np.diff(x) == 0
    if not same.any():
        return 0.0, 0.0
    edges = np.diff(np.concatenate([[0], same.astype(np.int8), [0]]))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    lengths = stops - starts + 1  # samples in the run
    values = x[starts]
    finite = x[np.isfinite(x)]
    lo, hi = (finite.min(), finite.max()) if finite.size else (np.nan, np.nan)
    flat = lengths[lengths >= min_flat].sum()
    at_extreme = (values == lo) | (values == hi)
    clip = lengths[at_extreme & (lengths >= min_clip)].sum()
    return float(100 * flat / len(x)), float(100 * clip / len(x))


def _peak_height_db(freqs: np.ndarray, psd: np.ndarray, f0: float, window: float) -> np.ndarray:
    """Height (dB) of each channel's spectrum at f0 above the outer thirds of the window."""
    log = 10 * np.log10(np.maximum(psd, 1e-40))
    sel = np.flatnonzero(np.abs(freqs - f0) <= window / 2)
    third = max(1, len(sel) // 3)
    baseline = np.concatenate([log[:, sel[:third]], log[:, sel[-third:]]], axis=1).mean(axis=1)
    peak = (
        log[:, sel[third:-third]].max(axis=1) if len(sel) > 2 * third else log[:, sel].max(axis=1)
    )
    return np.asarray(peak - baseline)


def _near(f: float, harmonics: list[float], tolerance: float) -> bool:
    return any(abs(f - h) <= tolerance for h in harmonics)


def _robust_z(x: np.ndarray) -> np.ndarray:
    mad = np.median(np.abs(x - np.median(x))) * 1.4826
    return np.asarray((x - np.median(x)) / mad) if mad > 0 else np.zeros_like(x)


def _event_average(
    raw: BaseRaw,
    pick: int,
    samples: np.ndarray,
    window: tuple[float, float],
    band: tuple[float, float],
) -> dict[str, np.ndarray]:
    """Average of one channel around events, band-passed like MNE's EOG detection."""
    x = mne.filter.filter_data(
        raw.get_data(picks=[pick])[0], raw.info["sfreq"], *band, verbose=False
    )
    return _array_average(x, samples, raw.info["sfreq"], window)


def _array_average(
    x: np.ndarray, samples: np.ndarray, sfreq: float, window: tuple[float, float]
) -> dict[str, np.ndarray]:
    start, stop = round(window[0] * sfreq), round(window[1] * sfreq)
    keep = [s for s in samples if s + start >= 0 and s + stop <= len(x)]
    times = np.arange(start, stop) / sfreq
    if not keep:
        return {"times": times.astype(np.float32), "mean": np.full(len(times), np.nan, np.float32)}
    segments = np.stack([x[s + start : s + stop] for s in keep])
    return {"times": times.astype(np.float32), "mean": segments.mean(axis=0).astype(np.float32)}


def _rate_figure(
    times: np.ndarray,
    duration: float,
    average: dict[str, np.ndarray],
    what: str,
    unit: str,
    title: str,
    ylabel: str,
    scale: float,
) -> Any:
    """Event rate per minute over time, and the event-locked average."""
    import matplotlib.pyplot as plt

    fig, (ax_rate, ax_avg) = plt.subplots(
        1, 2, figsize=(10, 3.2), width_ratios=(2, 1), layout="constrained"
    )
    if what == "Heart rate" and len(times) > 1:
        ax_rate.plot(times[1:], 60 / np.diff(times), ".", ms=3, color="C0")
        ax_rate.set(ylabel=f"{what} ({unit.strip()})")
    else:
        width = max(10.0, duration / 20)  # bin width in seconds
        edges = np.arange(0, duration + width, width)
        counts, _ = np.histogram(times, bins=edges)
        ax_rate.bar(edges[:-1], counts * 60 / width, width=0.9 * width, align="edge", color="C0")
        ax_rate.set(ylabel=f"{what} per minute")
    ax_rate.set(xlabel="Time (s)", title=title)
    ax_avg.plot(average["times"], average["mean"] * scale, color="C0")
    ax_avg.axvline(0, color="0.5", lw=0.8)
    ax_avg.set(xlabel="Time (s)", ylabel=ylabel, title="Average")
    return fig
