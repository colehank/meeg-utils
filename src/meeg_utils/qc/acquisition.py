"""Second-batch acquisition checks: empty room, cHPI SNR, SQUID jumps, photodiode, BIDS metadata."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any, ClassVar

import mne
import numpy as np
from mne.io import BaseRaw

from ..io.system import MEG_SYSTEMS
from ..steps._plotting import UNITS, psd_summary
from ..steps._utils import picks_by_type
from ._base import Check

#: SQUID-based MEG systems (OPMs have no flux jumps).
SQUID_SYSTEMS = frozenset(MEG_SYSTEMS) - {"opm"}
#: Data files whose header file MNE reads instead (BrainVision, EEGLAB).
_HEADER_FILES = {".eeg": ".vhdr", ".vmrk": ".vhdr", ".fdt": ".set"}
#: Modified z-score above which a channel is an outlier (Iglewicz & Hoaglin, 1993).
OUTLIER_Z = 3.5


class SquidJumps(Check):
    """Flux jumps of SQUID sensors: sudden, lasting steps in an MEG channel.

    Follows FieldTrip's jump detection (``ft_artifact_zvalue`` as configured
    in its artifact-rejection tutorial): each channel is median filtered
    (order 9), differentiated, and the absolute derivative z-scored over
    time; values above ``cutoff`` (20) are jumps. A jump must also shift the
    signal's level, which tells it from a spike.

    Parameters
    ----------
    cutoff : float
        z-score threshold of the derivative (FieldTrip tutorial: 20).
    min_shift : float
        A jump must change the median level 50 ms after it, compared with
        50 ms before, by at least this fraction of its size.

    Attributes
    ----------
    metrics_ : dict
        ``n_jumps``, ``channels`` (channel -> number of jumps) and
        ``times`` (s, first 50).

    Notes
    -----
    Figures: ``"jumps"``, the affected channels around their first jump.
    """

    name: ClassVar[str] = "squid_jumps"
    modalities: ClassVar[frozenset[str]] = frozenset({"meg"})
    systems: ClassVar[frozenset[str] | None] = SQUID_SYSTEMS
    plot_kinds: ClassVar[dict[str, bool]] = {"jumps": False}

    def __init__(self, *, cutoff: float = 20.0, min_shift: float = 0.5) -> None:
        self.cutoff = cutoff
        self.min_shift = min_shift

    def _compute(self, raw: BaseRaw) -> None:
        from scipy.signal import medfilt  # type: ignore[import-untyped]

        sfreq = raw.info["sfreq"]
        half = max(1, round(0.05 * sfreq))
        picks = mne.pick_types(raw.info, meg=True, ref_meg=False, exclude="bads")
        jumps: dict[str, list[int]] = {}
        snippets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for pick in picks:
            x = raw._data[pick]
            filtered = medfilt(x, 9)
            d = np.abs(np.diff(filtered))
            std = d.std()
            if std == 0:
                continue
            candidates = np.flatnonzero((d - d.mean()) / std > self.cutoff)
            found = []
            for i in _first_of_runs(candidates):
                before = np.median(filtered[max(0, i - half) : i])
                after = np.median(filtered[i + 1 : i + 1 + half])
                size = np.abs(filtered[i + 1] - filtered[i])
                if i >= half and abs(after - before) >= self.min_shift * size:
                    found.append(int(i))
            if found:
                name = raw.ch_names[pick]
                jumps[name] = found
                lo, hi = max(0, found[0] - 10 * half), min(len(x), found[0] + 10 * half)
                snippets[name] = (
                    (np.arange(lo, hi) / sfreq).astype(np.float32),
                    x[lo:hi].astype(np.float32),
                )
        times = sorted(i / sfreq for found in jumps.values() for i in found)
        self.snippets_ = snippets
        self.metrics_.update(
            n_jumps=len(times),
            channels={ch: len(v) for ch, v in jumps.items()},
            times=[round(t, 3) for t in times[:50]],
        )
        self._judge(
            "n_jumps",
            len(times),
            warn=0,
            what="SQUID jumps",
            detail=f"in {sorted(jumps)}; mark the channels bad or annotate the jumps",
        )

    def _plot_jumps(self, inst: Any) -> Any:
        import matplotlib.pyplot as plt

        n = max(1, min(len(self.snippets_), 6))
        fig, axes = plt.subplots(
            n, 1, figsize=(8, 1.6 * n + 0.6), squeeze=False, layout="constrained"
        )
        if not self.snippets_:
            axes[0, 0].text(0.5, 0.5, "No SQUID jumps", ha="center", va="center")
            axes[0, 0].set_axis_off()
        for ax, (ch, (t, x)) in zip(axes[:, 0], list(self.snippets_.items())[:n], strict=False):
            ax.plot(t, x, lw=0.8, color="C0")
            ax.set(title=f"{ch}: {self.metrics_['channels'][ch]} jump(s)", ylabel="T or T/m")
        axes[-1, 0].set_xlabel("Time (s, from the start of the data)")
        return fig


class ChpiSNR(Check):
    """Signal-to-noise ratio of the continuous head-position (cHPI) coils.

    :func:`mne.chpi.compute_chpi_snr` estimates, over time, the power of
    each coil's frequency relative to the residual. A coil much weaker than
    the others is loose, badly placed or broken, and head positions computed
    from it are unreliable.

    Parameters
    ----------
    max_drop_db : float | None
        A coil whose median SNR is this much below the median over coils is
        flagged. There is no published limit (coils differ with their
        distance to the sensors, so some spread is normal); ``None``
        (default) reports each coil's drop without a verdict. 10 dB (a tenth
        of the power) is a reasonable starting point.

    Attributes
    ----------
    metrics_ : dict
        Per channel type (``mag``/``grad``): median SNR per coil (dB) and
        the coil frequencies.

    Notes
    -----
    Figures: ``"snr"`` (:func:`mne.viz.plot_chpi_snr`).
    """

    name: ClassVar[str] = "chpi_snr"
    modalities: ClassVar[frozenset[str]] = frozenset({"meg"})
    systems: ClassVar[frozenset[str] | None] = frozenset({"neuromag"})
    plot_kinds: ClassVar[dict[str, bool]] = {"snr": False}

    def __init__(self, *, max_drop_db: float | None = None) -> None:
        self.max_drop_db = max_drop_db

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
        if reason is None:
            freqs = mne.chpi.get_chpi_info(raw.info, on_missing="ignore", verbose=False)[0]
            if not len(freqs):
                reason = "found no continuous head localization (cHPI) in the recording"
        return reason

    def _compute(self, raw: BaseRaw) -> None:
        snr = mne.chpi.compute_chpi_snr(raw, verbose=False)
        self.snr_ = snr
        freqs = [float(f) for f in snr["freqs"]]
        self.metrics_["freqs"] = freqs
        for ch_type in ("mag", "grad"):
            key = f"{ch_type}_snr"
            if key not in snr:
                continue
            median = np.median(snr[key], axis=0)
            self.metrics_[ch_type] = [round(float(v), 2) for v in median]
            reference = float(np.median(median))
            for i, value in enumerate(median):
                self._judge(
                    f"{ch_type}_coil{i + 1}_drop_db",
                    round(reference - float(value), 2),
                    warn=self.max_drop_db,
                    what=f"{ch_type.upper()} SNR of HPI coil {i + 1} ({freqs[i]:g} Hz) below "
                    "the median coil",
                    unit=" dB",
                    detail="coil loose, badly placed or broken",
                    typical="compare across coils and recordings",
                )

    def _plot_snr(self, inst: Any) -> Any:
        return mne.viz.plot_chpi_snr(self.snr_)


class Photodiode(Check):
    """Delay and jitter between the triggers and what the screen showed.

    The photodiode channel is thresholded half-way between its low and high
    levels (or at ``threshold``); each trigger is matched with the first
    photodiode onset within ``max_delay`` after it. A constant delay can be
    corrected by shifting the events; jitter cannot, and smears evoked
    responses.

    Parameters
    ----------
    channel : str
        The photodiode channel.
    events : {"auto", "annotations", "stim"}
        Where the triggers are (as :class:`~meeg_utils.steps.Epoch`).
    event_id : list of str | None
        Trigger names to use (``None``: all).
    threshold : float | None
        Onset threshold in the channel's units; ``None`` uses the midpoint
        of its 5th and 95th percentiles.
    max_delay : float
        Longest accepted delay (s).
    max_jitter_ms : float | None
        Standard deviation of the delay above which a warning is raised;
        ``None`` (default) only reports it, as the acceptable jitter depends
        on the analysis (no published limit).

    Attributes
    ----------
    metrics_ : dict
        ``n_triggers``, ``n_matched``, ``delay_ms`` (median), ``jitter_ms``
        (standard deviation), ``delay_range_ms``.

    Notes
    -----
    Figures: ``"delays"`` (delay of each trigger over time, and their
    distribution).
    """

    name: ClassVar[str] = "photodiode"
    plot_kinds: ClassVar[dict[str, bool]] = {"delays": False}

    def __init__(
        self,
        channel: str,
        *,
        events: str = "auto",
        event_id: list[str] | None = None,
        threshold: float | None = None,
        max_delay: float = 0.1,
        max_jitter_ms: float | None = None,
    ) -> None:
        self.channel = channel
        self.events = events
        self.event_id = event_id
        self.threshold = threshold
        self.max_delay = max_delay
        self.max_jitter_ms = max_jitter_ms

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
        if reason is None and self.channel not in raw.ch_names:
            reason = f"found no photodiode channel {self.channel!r}"
        return reason

    def _compute(self, raw: BaseRaw) -> None:
        from ..steps.epochs import Epoch

        sfreq = raw.info["sfreq"]
        epoch = Epoch(self.event_id, events=self.events).fit(raw)
        triggers, _ = epoch._find_events(raw, epoch.event_id_)
        triggers = triggers[np.isin(triggers[:, 2], list(epoch.event_id_.values()))]
        samples = triggers[:, 0] - raw.first_samp
        signal = raw.get_data(self.channel)[0]
        low, high = np.percentile(signal, [5, 95])
        threshold = (low + high) / 2 if self.threshold is None else self.threshold
        above = signal > threshold
        if high < low or not above.any() or above.all():
            raise ValueError(
                f"The photodiode channel {self.channel!r} never crosses its threshold."
            )
        onsets = np.flatnonzero(above[1:] & ~above[:-1]) + 1
        delays, matched_at = [], []
        for sample in samples:
            after = onsets[(onsets >= sample) & (onsets <= sample + self.max_delay * sfreq)]
            if len(after):
                delays.append((after[0] - sample) / sfreq)
                matched_at.append(sample / sfreq)
        self.delays_ = np.asarray(delays, dtype=np.float32)
        self.matched_times_ = np.asarray(matched_at, dtype=np.float32)
        n, matched = len(samples), len(delays)
        self.metrics_.update(n_triggers=n, n_matched=matched, threshold=float(threshold))
        self._judge(
            "n_unmatched",
            n - matched,
            warn=0,
            what="triggers without a photodiode onset",
            detail=f"within {1000 * self.max_delay:g} ms (stimulus not shown, or a longer delay)",
        )
        if matched:
            ms = 1000 * self.delays_
            self.metrics_.update(
                delay_ms=round(float(np.median(ms)), 2),
                jitter_ms=round(float(ms.std()), 2),
                delay_range_ms=[round(float(ms.min()), 2), round(float(ms.max()), 2)],
            )
            self._judge(
                "jitter_ms",
                self.metrics_["jitter_ms"],
                warn=self.max_jitter_ms,
                what="photodiode delay jitter (SD)",
                unit=" ms",
                detail=f"median delay {self.metrics_['delay_ms']:g} ms",
            )

    def _plot_delays(self, inst: Any) -> Any:
        import matplotlib.pyplot as plt

        fig, (ax_t, ax_h) = plt.subplots(
            1, 2, figsize=(10, 3.2), width_ratios=(2, 1), layout="constrained"
        )
        ms = 1000 * self.delays_
        ax_t.plot(self.matched_times_, ms, ".", color="C0")
        ax_t.set(xlabel="Trigger time (s)", ylabel="Delay (ms)", title="Trigger to photodiode")
        ax_h.hist(ms, bins=max(5, min(40, len(ms) // 3)), color="C0")
        ax_h.set(xlabel="Delay (ms)", ylabel="Triggers")
        if len(ms):
            ax_h.set_title(f"median {np.median(ms):.1f} ms, SD {ms.std():.2f} ms")
        return fig


class BidsMetadata(Check):
    """Whether the BIDS sidecar describes the recording correctly.

    Compares the BIDS ``*_meg.json`` / ``*_eeg.json`` sidecar with the data
    file (sampling rate, duration) and with ``channels.tsv`` (channel counts
    per type), checks that ``channels.tsv`` lists the file's channels, and
    that the line frequency is given. A wrong sidecar silently breaks
    analyses that rely on it (e.g. the line frequency used to remove line
    noise). Only applies to recordings in a BIDS dataset.

    Attributes
    ----------
    metrics_ : dict
        ``mismatches`` (field -> ``{"sidecar": ..., "data": ...}``) and
        ``missing`` (recommended fields absent from the sidecar).
    """

    name: ClassVar[str] = "bids_metadata"
    #: Sidecar field -> channels.tsv types counted for it.
    _COUNTS: ClassVar[dict[str, tuple[str, ...]]] = {
        "MEGChannelCount": ("MEGMAG", "MEGGRADAXIAL", "MEGGRADPLANAR", "MEGOTHER"),
        "MEGREFChannelCount": ("MEGREFMAG", "MEGREFGRADAXIAL", "MEGREFGRADPLANAR"),
        "EEGChannelCount": ("EEG",),
        "EOGChannelCount": ("EOG", "VEOG", "HEOG"),
        "ECGChannelCount": ("ECG",),
        "EMGChannelCount": ("EMG",),
        "MiscChannelCount": ("MISC",),
        "TriggerChannelCount": ("TRIG",),
    }

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
        if reason is None and _bids_path(raw) is None:
            reason = "the recording is not in a BIDS dataset"
        return reason

    def _compute(self, raw: BaseRaw) -> None:
        import json

        path = _bids_path(raw)
        assert path is not None
        sidecar_path = path.copy().update(extension=".json", split=None)
        sidecar = json.loads(Path(str(sidecar_path.fpath)).read_text(encoding="utf-8"))
        header = _HEADER_FILES.get(str(path.extension).lower())
        data_file = path.copy().update(extension=header) if header else path
        native = mne.io.read_raw(str(data_file.fpath), preload=False, verbose=False)
        data: dict[str, Any] = {
            "SamplingFrequency": float(native.info["sfreq"]),
            "RecordingDuration": float(native.times[-1] + 1 / native.info["sfreq"]),
        }
        mismatches: dict[str, Any] = {}
        channels_tsv = Path(
            str(path.copy().update(suffix="channels", extension=".tsv", split=None).fpath)
        )
        if channels_tsv.exists():
            names = _tsv_column(channels_tsv, "name")
            types = [t.upper() for t in _tsv_column(channels_tsv, "type")]
            for field, counted in self._COUNTS.items():
                data[field] = sum(t in counted for t in types)
            if sorted(names) != sorted(native.ch_names):
                mismatches["channels.tsv"] = {
                    "sidecar": sorted(set(names) - set(native.ch_names)),
                    "data": sorted(set(native.ch_names) - set(names)),
                }
        missing = []
        for field, value in data.items():
            if field not in sidecar:
                if field in ("SamplingFrequency", "RecordingDuration"):
                    missing.append(field)
                continue
            expected = sidecar[field]
            same = (
                np.isclose(float(expected), value, rtol=1e-3, atol=0.5 / native.info["sfreq"])
                if isinstance(value, float)
                else int(expected) == value
            )
            if not same:
                mismatches[field] = {"sidecar": expected, "data": value}
        if "PowerLineFrequency" not in sidecar or sidecar["PowerLineFrequency"] in (None, "n/a"):
            missing.append("PowerLineFrequency")
        self.metrics_.update(mismatches=mismatches, missing=missing)
        for field, values in mismatches.items():
            level = "fail" if field == "SamplingFrequency" else "warn"
            self._note(
                field,
                values,
                f"{field}: sidecar says {values['sidecar']}, the data have {values['data']}",
                level,
            )
        if missing:
            self._note("missing", missing, f"sidecar lacks {', '.join(missing)}", "warn")


class EmptyRoom(Check):
    """The empty-room recording of the session: sensor noise and noisy sensors.

    The empty-room recording (no subject) shows the noise of the sensors and
    the room on the day. It is found in the BIDS dataset
    (:meth:`mne_bids.BIDSPath.find_empty_room`, the one closest in date) or
    given. Reported per channel type: the median noise floor (amplitude
    spectral density, 20-100 Hz without line harmonics) and how much the
    recording exceeds it; flagged: sensors that are outliers in the empty
    room (noisy or dead sensors to mark bad). The days between the empty
    room and the recording are reported.

    Parameters
    ----------
    empty_room : "auto" | str | Path | Raw
        ``"auto"`` searches the BIDS dataset of the recording.
    band : tuple of float
        Band of the noise floor (Hz).
    max_days : float | None
        An empty room more than this many days from the recording is
        flagged. Same-day empty rooms are usual (MNE-BIDS-Pipeline picks the
        closest in date), but how fast the room's noise changes depends on
        the site, so ``None`` (default) reports the gap without a verdict.

    Attributes
    ----------
    metrics_ : dict
        ``empty_room`` (file), ``days_apart``, per channel type
        ``noise_floor`` (fT/√Hz or fT/cm/√Hz), ``recording_above_db``
        (recording minus empty room, median over channels) and
        ``outlier_channels``.

    Notes
    -----
    Figures: ``"psd"`` (median spectra of the recording and the empty room,
    per channel type).
    """

    name: ClassVar[str] = "empty_room"
    modalities: ClassVar[frozenset[str]] = frozenset({"meg"})
    systems: ClassVar[frozenset[str] | None] = SQUID_SYSTEMS
    plot_kinds: ClassVar[dict[str, bool]] = {"psd": False}

    def __init__(
        self,
        *,
        empty_room: Any = "auto",
        band: tuple[float, float] = (20.0, 100.0),
        max_days: float | None = None,
    ) -> None:
        self.empty_room = empty_room
        self.band = band
        self.max_days = max_days

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
        if reason is None and isinstance(self.empty_room, str) and self.empty_room == "auto":
            path = _bids_path(raw)
            if path is None:
                return "needs an empty-room recording (the recording is not in a BIDS dataset)"
            if _find_empty_room(path) is None:
                return "found no empty-room recording in the BIDS dataset"
        return reason

    def _compute(self, raw: BaseRaw) -> None:
        if isinstance(self.empty_room, BaseRaw):
            er, source = self.empty_room, "<Raw>"
        else:
            fname = (
                _find_empty_room(_bids_path(raw))
                if isinstance(self.empty_room, str) and self.empty_room == "auto"
                else self.empty_room
            )
            er = (
                _read_bids(fname)
                if not isinstance(fname, str | Path)
                else mne.io.read_raw(fname, verbose=False)
            )
            source = str(getattr(fname, "fpath", fname))
        er = er.copy().load_data(verbose=False)
        self.metrics_["empty_room"] = source
        if raw.info["meas_date"] is not None and er.info["meas_date"] is not None:
            days = abs((raw.info["meas_date"] - er.info["meas_date"]).total_seconds()) / 86400
            self.metrics_["days_apart"] = round(days, 2)
            self._judge(
                "days_apart",
                round(days, 2),
                warn=self.max_days,
                what="days between the recording and its empty room",
                detail="the room's noise may have changed",
                typical="same day is usual",
            )

        line = raw.info["line_freq"] or er.info["line_freq"]
        self.psd_: dict[str, dict[str, np.ndarray]] = {}
        good_er = [ch for ch in er.ch_names if ch not in er.info["bads"]]
        for ch_type, picks in picks_by_type(raw.info).items():
            if ch_type not in ("mag", "grad"):
                continue
            names = [raw.ch_names[p] for p in picks if raw.ch_names[p] in good_er]
            if not names:
                continue
            freqs, psd_rec = psd_summary(raw.get_data(names), raw.info["sfreq"])
            freqs_er, psd_er = psd_summary(er.get_data(names), er.info["sfreq"])
            psd_er = np.array([np.interp(freqs, freqs_er, p) for p in psd_er])
            self.psd_[ch_type] = {"freqs": freqs, "recording": psd_rec, "empty_room": psd_er}
            sel = (freqs >= self.band[0]) & (freqs <= self.band[1])
            if line:
                for harmonic in np.arange(line, freqs[-1], line):
                    sel &= np.abs(freqs - harmonic) > 2
            scale, unit = UNITS[ch_type]
            asd_er = np.sqrt(np.median(psd_er[:, sel], axis=1)) * scale  # per channel
            power_rec = np.median(psd_rec[:, sel], axis=1)
            power_er = np.median(psd_er[:, sel], axis=1)
            live = (power_er > 0) & (power_rec > 0)
            z = _modified_z(np.log10(asd_er[live])) if live.sum() > 2 else np.zeros(live.sum())
            outliers = [
                str(n)
                for n, zi in zip(np.array(names)[live], z, strict=True)
                if abs(zi) > OUTLIER_Z
            ]
            dead = [n for n, ok in zip(names, live, strict=True) if not ok]
            self.metrics_[ch_type] = {
                "noise_floor": round(float(np.median(asd_er[live])), 3) if live.any() else None,
                "unit": f"{unit}/√Hz",
                "recording_above_db": round(
                    float(np.median(10 * np.log10(power_rec[live] / power_er[live]))), 2
                )
                if live.any()
                else None,
                "outlier_channels": sorted(outliers + dead),
            }
            self._judge(
                f"{ch_type}_outlier_channels",
                len(outliers) + len(dead),
                warn=0,
                what=f"{ch_type.upper()} sensors that are outliers in the empty room",
                detail=f"{sorted(outliers + dead)}; mark them bad",
            )

    def _plot_psd(self, inst: Any) -> Any:
        from ..steps._plotting import plot_psd_comparison

        spectra = {
            t: {"freqs": s["freqs"], "before": s["empty_room"], "after": s["recording"]}
            for t, s in self.psd_.items()
        }
        return plot_psd_comparison(
            spectra, title="Recording and empty room", labels=("empty room", "recording")
        )


# ----------------------------------------------------------------------


def _first_of_runs(indices: np.ndarray) -> list[int]:
    """First index of each run of consecutive indices."""
    if not len(indices):
        return []
    starts = [indices[0]] + [b for a, b in pairwise(indices) if b > a + 1]
    return [int(i) for i in starts]


def _modified_z(x: np.ndarray) -> np.ndarray:
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    return np.asarray(0.6745 * (x - median) / mad) if mad > 0 else np.zeros_like(x)


def _bids_path(raw: BaseRaw) -> Any:
    from ..io.read import as_bids_path

    fname = raw.filenames[0] if raw.filenames else None
    return as_bids_path(Path(fname)) if fname is not None else None


def _find_empty_room(path: Any) -> Any:
    try:
        return path.find_empty_room(verbose=False)
    except Exception:
        return None


def _read_bids(path: Any) -> BaseRaw:
    """Read a BIDSPath with mne-bids (sidecar metadata applied)."""
    from mne_bids import read_raw_bids

    return read_raw_bids(path, verbose=False)


def _tsv_column(fname: Path, column: str) -> list[str]:
    lines = fname.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    index = header.index(column)
    return [line.split("\t")[index] for line in lines[1:] if line.strip()]
