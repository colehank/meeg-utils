"""Temporal filtering and resampling."""

from __future__ import annotations

from typing import ClassVar

from mne import Evoked
from mne.epochs import BaseEpochs
from mne.io import BaseRaw

from ..core import Step
from ..core.step import Inst


class Filter(Step):
    """Band-pass, high-pass or low-pass filter the data.

    Thin wrapper of :meth:`mne.io.Raw.filter`; the defaults are MNE's
    (zero-phase FIR, ``firwin`` design, automatic transition bands).

    Parameters
    ----------
    l_freq : float | None
        High-pass cutoff in Hz; ``None`` for a low-pass filter.
    h_freq : float | None
        Low-pass cutoff in Hz; ``None`` for a high-pass filter.
    picks : str | list | None
        Channels to filter; ``None`` means all data channels.
    method : {"fir", "iir"}
        Filter type.
    iir_params : dict | None
        IIR parameters when ``method="iir"``.
    phase : str
        ``"zero"`` (default), ``"zero-double"``, ``"minimum"`` or ``"causal"``.
    l_trans_bandwidth, h_trans_bandwidth : float | "auto"
        Transition bandwidths in Hz.
    filter_length : str | int
        FIR filter length.
    fir_window : str
        FIR window.
    fir_design : str
        FIR design method.
    n_jobs : int | str | None
        Parallel jobs, or ``"cuda"``.

    Attributes
    ----------
    qc_ : dict
        ``highpass`` and ``lowpass`` of the filtered data (from ``info``).
    """

    accepts: ClassVar[tuple[type, ...]] = (BaseRaw, BaseEpochs, Evoked)

    def __init__(
        self,
        l_freq: float | None,
        h_freq: float | None,
        *,
        picks: str | list | None = None,
        method: str = "fir",
        iir_params: dict | None = None,
        phase: str = "zero",
        l_trans_bandwidth: float | str = "auto",
        h_trans_bandwidth: float | str = "auto",
        filter_length: str | int = "auto",
        fir_window: str = "hamming",
        fir_design: str = "firwin",
        n_jobs: int | str | None = None,
    ) -> None:
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.picks = picks
        self.method = method
        self.iir_params = iir_params
        self.phase = phase
        self.l_trans_bandwidth = l_trans_bandwidth
        self.h_trans_bandwidth = h_trans_bandwidth
        self.filter_length = filter_length
        self.fir_window = fir_window
        self.fir_design = fir_design
        self.n_jobs = n_jobs

    def _fit(self, inst: Inst) -> None:
        if self.l_freq is None and self.h_freq is None:
            raise ValueError("Filter needs l_freq and/or h_freq.")
        nyquist = inst.info["sfreq"] / 2.0
        if self.h_freq is not None and self.h_freq >= nyquist:
            raise ValueError(
                f"h_freq ({self.h_freq} Hz) must be below the Nyquist frequency ({nyquist} Hz)."
            )
        if self.l_freq is not None and self.h_freq is not None and self.l_freq >= self.h_freq:
            raise ValueError(f"l_freq ({self.l_freq} Hz) must be below h_freq ({self.h_freq} Hz).")

    def _transform(self, inst: Inst) -> Inst:
        inst.filter(
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            picks=self.picks,
            method=self.method,
            iir_params=self.iir_params,
            phase=self.phase,
            l_trans_bandwidth=self.l_trans_bandwidth,
            h_trans_bandwidth=self.h_trans_bandwidth,
            filter_length=self.filter_length,
            fir_window=self.fir_window,
            fir_design=self.fir_design,
            n_jobs=self.n_jobs,
            verbose=False,
        )
        self.qc_.update(highpass=inst.info["highpass"], lowpass=inst.info["lowpass"])
        return inst


class Resample(Step):
    """Resample the data.

    Thin wrapper of :meth:`mne.io.Raw.resample`, which applies its own
    anti-aliasing filter; annotations and events keep their timing.

    Parameters
    ----------
    sfreq : float
        New sampling frequency in Hz.
    npad : int | str
        Padding.
    window : str
        Frequency-domain window.
    method : {"fft", "polyphase"}
        Resampling method.
    n_jobs : int | str | None
        Parallel jobs, or ``"cuda"``.

    Attributes
    ----------
    qc_ : dict
        ``sfreq_before`` and ``sfreq_after``.
    """

    accepts: ClassVar[tuple[type, ...]] = (BaseRaw, BaseEpochs, Evoked)
    changes_times: ClassVar[bool] = True

    def __init__(
        self,
        sfreq: float,
        *,
        npad: int | str = "auto",
        window: str = "auto",
        method: str = "fft",
        n_jobs: int | str | None = None,
    ) -> None:
        self.sfreq = sfreq
        self.npad = npad
        self.window = window
        self.method = method
        self.n_jobs = n_jobs

    def _fit(self, inst: Inst) -> None:
        if self.sfreq <= 0:
            raise ValueError(f"sfreq must be positive, got {self.sfreq}.")
        lowpass = inst.info["lowpass"]
        if lowpass is not None and lowpass >= self.sfreq / 2.0:
            self.qc_["note"] = (
                f"Data low-pass ({lowpass} Hz) is above the new Nyquist frequency "
                f"({self.sfreq / 2} Hz); MNE's anti-aliasing filter is applied."
            )

    def _transform(self, inst: Inst) -> Inst:
        before = inst.info["sfreq"]
        if before != self.sfreq:
            kwargs = dict(
                npad=self.npad,
                window=self.window,
                method=self.method,
                n_jobs=self.n_jobs,
                verbose=False,
            )
            inst.resample(self.sfreq, **kwargs)
        self.qc_.update(sfreq_before=before, sfreq_after=inst.info["sfreq"])
        return inst
