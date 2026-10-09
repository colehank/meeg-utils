"""Artifact subspace reconstruction and sensor noise suppression (mne-denoise)."""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

import numpy as np
from mne import Evoked
from mne.epochs import BaseEpochs
from mne.io import BaseRaw

from ..core import Step
from ..core.step import Inst
from ._plotting import plot_psd_comparison, psd_summary
from ._utils import picks_by_type

#: High-pass (Hz) below which ASR is warned about: EEGLAB's clean_rawdata
#: filters with a 0.25-0.75 Hz transition band before ASR.
ASR_MIN_HIGHPASS = 0.25


class ASR(Step):
    """Artifact subspace reconstruction: repair high-amplitude bursts in continuous EEG.

    Wraps :class:`mne_denoise.asr.ASR` (Kothe & Jung, 2016; Chang et al.,
    2020, IEEE Trans Biomed Eng 67:1114). ``fit`` finds clean calibration
    windows in the recording and learns the statistics of clean data;
    ``transform`` slides a window over the data and, where components exceed
    ``cutoff`` standard deviations of the calibration data, reconstructs them
    from the rest.

    Parameters
    ----------
    cutoff : float
        Threshold in standard deviations (default 20, EEGLAB's
        ``clean_rawdata`` default; Chang et al. 2020 recommend 20-30). Lower
        values repair more, and remove more brain signal.
    picks : str
        Channel type to process (default ``"eeg"``). Bad channels are left
        out: detect them first.
    window_length : float
        Processing window (s).
    max_dims : float | int
        Maximum fraction (or number) of dimensions reconstructed per window.
    calibration : {"auto", "manual"}
        ``"auto"`` selects clean calibration windows; ``"manual"`` uses all of
        the data passed to ``fit`` (e.g. a clean baseline recording).

    Attributes
    ----------
    asr_ : mne_denoise.asr.ASR
        The fitted model.
    qc_ : dict
        ``n_channels``, ``fraction_reconstructed_windows``,
        ``fraction_reconstructed_samples``, ``max_components_reconstructed``
        and ``variance_removed_pct`` (last transform).

    Notes
    -----
    High-pass filter first (EEGLAB filters at 0.25-0.75 Hz): drifts inflate
    the calibration statistics. ASR changes the data only where it detects
    bursts; to mark segments instead, see :class:`BadSegments`.

    Figures (:meth:`plot`): ``"reconstruction"`` (components reconstructed
    over time) and ``"psd"``.
    """

    accepts: ClassVar[tuple[type, ...]] = (BaseRaw,)
    plot_kinds: ClassVar[dict[str, bool]] = {"reconstruction": False, "psd": False}

    def __init__(
        self,
        cutoff: float = 20.0,
        *,
        picks: str = "eeg",
        window_length: float = 0.5,
        max_dims: float | int = 0.66,
        calibration: str = "auto",
    ) -> None:
        self.cutoff = cutoff
        self.picks = picks
        self.window_length = window_length
        self.max_dims = max_dims
        self.calibration = calibration

    def _fit(self, inst: BaseRaw) -> None:
        from mne_denoise.asr import ASR as _ASR

        names = _good(inst, self.picks)
        if len(names) < 2:
            raise ValueError(f"ASR needs at least 2 good {self.picks} channels.")
        if inst.info["highpass"] < ASR_MIN_HIGHPASS:
            warnings.warn(
                f"The data are high-passed at {inst.info['highpass']:g} Hz; ASR expects drifts "
                f"removed first (e.g. S.Filter({ASR_MIN_HIGHPASS:g}, None) or higher).",
                stacklevel=4,
            )
        self.asr_ = _ASR(
            cutoff=self.cutoff,
            window_length=self.window_length,
            max_dims=self.max_dims,
            calibration=self.calibration,
            picks=names,
            verbose=False,
        ).fit(inst)
        self.ch_names_ = names
        self.qc_.update(
            n_channels=len(names),
            calibration_fraction=round(float(np.mean(self.asr_.clean_window_mask_)), 4),
        )

    def _transform(self, inst: BaseRaw) -> BaseRaw:
        idx = _indices(inst, self.ch_names_)
        before = inst._data[idx].copy()
        cleaned = self.asr_.transform(inst, verbose=False)
        after = cleaned.get_data(self.ch_names_)
        inst._data[idx] = after
        asr = self.asr_
        sfreq = inst.info["sfreq"]
        self.windows_ = {
            "times": (np.asarray(asr.window_starts_) / sfreq).astype(np.float32),
            "n_components": np.asarray(asr.n_components_reconstructed_, dtype=np.int16),
        }
        self.duration_ = inst.n_times / sfreq
        freqs, psd_before = psd_summary(before, sfreq)
        self.psd_ = {
            self.picks: {
                "freqs": freqs,
                "before": psd_before,
                "after": psd_summary(after, sfreq)[1],
            }
        }
        self.qc_.update(
            fraction_reconstructed_windows=round(float(asr.fraction_reconstructed_windows_), 4),
            fraction_reconstructed_samples=round(float(asr.fraction_reconstructed_samples_), 4),
            max_components_reconstructed=int(asr.max_components_reconstructed_),
            variance_removed_pct=round(_variance_removed(before, after), 2),
        )
        return inst

    def _plot_unavailable(self, kind: str) -> str | None:
        if not hasattr(self, "windows_"):
            return "the step has not transformed data yet (use fit_transform)"
        return None

    def _plot_reconstruction(self, inst: Any) -> Any:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 2.8), layout="constrained")
        ax.step(self.windows_["times"], self.windows_["n_components"], where="post", lw=0.8)
        ax.set(
            xlabel="Time (s, from the start of the data)",
            ylabel="Components\nreconstructed",
            xlim=(0, self.duration_),
            title=(
                f"ASR (cutoff {self.cutoff:g}): "
                f"{100 * self.qc_['fraction_reconstructed_samples']:.1f} % of samples changed"
            ),
        )
        return fig

    def _plot_psd(self, inst: Any) -> Any:
        return plot_psd_comparison(self.psd_, title="Spectra before and after ASR")


class SNS(Step):
    """Sensor noise suppression: replace each channel by its projection on its neighbours.

    Wraps :class:`mne_denoise.sns.SNS` (de Cheveigné & Simon, 2008, J Neurosci
    Methods 168:195). Noise specific to single sensors (e.g. sensor or
    amplifier noise, a loose electrode) is not shared with other channels and
    is removed; activity seen by several sensors (brain signal, and also
    shared artifacts, which SNS does not remove) is kept. Each data channel
    type is processed separately; the operator is learned in ``fit``.

    Parameters
    ----------
    n_neighbors : int
        Number of most correlated channels each channel is projected on (10
        in de Cheveigné & Simon, 2008); 0 uses all channels.
    skip : int
        Number of most correlated neighbours to leave out (e.g. to keep
        a channel's own noise from leaking through near-duplicates).
    n_iter : int
        Number of times the projection is applied.

    Attributes
    ----------
    models_ : dict
        Channel type mapped to the fitted :class:`mne_denoise.sns.SNS`.
    qc_ : dict
        Per channel type: ``n_channels`` and ``variance_removed_pct`` (last
        transform).

    Notes
    -----
    Bad channels are left out and unchanged. SNS lowers the rank of the data
    only slightly, but channels become more correlated; run it before ICA.

    Figures (:meth:`plot`): ``"psd"``, spectra before and after.
    """

    accepts: ClassVar[tuple[type, ...]] = (BaseRaw, BaseEpochs, Evoked)
    plot_kinds: ClassVar[dict[str, bool]] = {"psd": False}

    def __init__(self, n_neighbors: int = 10, *, skip: int = 0, n_iter: int = 1) -> None:
        self.n_neighbors = n_neighbors
        self.skip = skip
        self.n_iter = n_iter

    def _fit(self, inst: Inst) -> None:
        from mne_denoise.sns import SNS as _SNS

        self.models_: dict[str, Any] = {}
        self.ch_names_: dict[str, list[str]] = {}
        for ch_type, picks in picks_by_type(inst.info).items():
            if len(picks) < 3:
                continue
            names = [inst.ch_names[p] for p in picks]
            n_neighbors = min(self.n_neighbors, len(names) - 1 - self.skip)
            model = _SNS(
                n_neighbors=n_neighbors, skip=self.skip, n_iter=self.n_iter,
                preserve_mean=True, verbose=False,
            )  # fmt: skip
            model.fit(_as_2d(inst.get_data(names)))
            self.models_[ch_type] = model
            self.ch_names_[ch_type] = names
            self.qc_[ch_type] = {"n_channels": len(names), "n_neighbors": n_neighbors}
        if not self.models_:
            raise ValueError("SNS needs at least 3 good channels of a data channel type.")

    def _transform(self, inst: Inst) -> Inst:
        sfreq = inst.info["sfreq"]
        self.psd_ = {}
        for ch_type, model in self.models_.items():
            idx = _indices(inst, self.ch_names_[ch_type])
            before = inst._data[..., idx, :].copy()
            after = model.transform(_as_2d(before))
            after = _restore(after, before.shape)
            inst._data[..., idx, :] = after
            freqs, psd_before = psd_summary(before, sfreq)
            self.psd_[ch_type] = {
                "freqs": freqs,
                "before": psd_before,
                "after": psd_summary(after, sfreq)[1],
            }
            self.qc_[ch_type]["variance_removed_pct"] = round(_variance_removed(before, after), 2)
        return inst

    def _plot_unavailable(self, kind: str) -> str | None:
        if kind == "psd" and not hasattr(self, "psd_"):
            return "the step has not transformed data yet (use fit_transform)"
        return None

    def _plot_psd(self, inst: Any) -> Any:
        return plot_psd_comparison(self.psd_, title="Spectra before and after SNS")


# ----------------------------------------------------------------------


def _good(inst: Inst, ch_type: str) -> list[str]:
    types = inst.get_channel_types()
    return [
        ch
        for ch, t in zip(inst.ch_names, types, strict=True)
        if t == ch_type and ch not in inst.info["bads"]
    ]


def _indices(inst: Inst, names: list[str]) -> list[int]:
    missing = [ch for ch in names if ch not in inst.ch_names]
    if missing:
        raise ValueError(f"The data lack channels the step was fitted on: {missing}.")
    return [inst.ch_names.index(ch) for ch in names]


def _as_2d(data: np.ndarray) -> np.ndarray:
    """Channels x samples: epochs are concatenated in time."""
    if data.ndim == 3:
        return np.ascontiguousarray(data.transpose(1, 0, 2).reshape(data.shape[1], -1))
    return data


def _restore(data: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if len(shape) == 3:
        n_epochs, n_channels, n_times = shape
        return data.reshape(n_channels, n_epochs, n_times).transpose(1, 0, 2)
    return data


def _variance_removed(before: np.ndarray, after: np.ndarray) -> float:
    before = before - before.mean(axis=-1, keepdims=True)
    after = after - after.mean(axis=-1, keepdims=True)
    total = float(np.sum(before**2))
    return 100 * (1 - float(np.sum(after**2)) / total) if total > 0 else 0.0
