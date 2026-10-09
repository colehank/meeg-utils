"""Power-line noise removal."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from mne.io import BaseRaw
from scipy.signal import welch  # type: ignore[import-untyped]

from ..core import Step
from ._utils import picks_by_type

METHODS = ("zapline-plus", "zapline", "notch")


class LineNoise(Step):
    """Remove power-line noise.

    ZapLine methods come from mne-denoise. Each data channel type (``mag``,
    ``grad``, ``eeg``) is processed separately, because their units differ,
    and the cleaned data are written back in place, so all other channels,
    annotations and timing are preserved. Channels marked bad are neither
    used nor cleaned.

    Parameters
    ----------
    method : {"zapline-plus", "zapline", "notch"}
        - ``"zapline-plus"`` (default): adaptive ZapLine-plus (Klug &
          Kloosterman, 2022). It detects the noise around ``fline``,
          segments the recording and chooses the number of components per
          segment. It adapts to whatever data it transforms, so ``transform``
          re-estimates on new data.
        - ``"zapline"``: standard ZapLine (de Cheveigné, 2020). Spatial filters
          are learned in ``fit`` and reused by ``transform``.
        - ``"notch"``: notch filters at ``fline`` and its harmonics
          (:meth:`mne.io.Raw.notch_filter`).
    fline : float | None
        Line frequency in Hz. ``None`` (default) uses ``info["line_freq"]``,
        which mne-bids reads from the BIDS sidecar; fitting fails if neither
        is set rather than assuming 50 or 60 Hz.
    n_remove : int | "auto"
        ``"zapline"`` only: number of components to remove, or ``"auto"``
        (outlier-based selection, capped by mne-denoise at 20% of the
        components). Fixed counts should stay small: ZapLine's residual
        contains broadband activity above roughly ``fline / 2`` as well as
        the line noise, so removing many components distorts it.
    n_harmonics : int | None
        Number of harmonics to target; ``None`` uses all below Nyquist.
    nfft : int
        FFT length used by ZapLine.
    adaptive_params : dict | None
        ``"zapline-plus"`` only: options passed to mne-denoise's adaptive mode.

    Attributes
    ----------
    fline_ : float
        Line frequency used.
    qc_ : dict
        Per channel type: ``suppression_db`` (power reduction at ``fline``),
        ``distortion_db`` (mean absolute log-power change 1-45 Hz outside the
        line harmonics), ``distortion_db_broadband`` (same up to 90% of
        Nyquist), ``variance_removed_pct``, ``n_flat_skipped`` (flat channels
        left out of these metrics) and, for ZapLine, ``n_removed``.
    """

    accepts: ClassVar[tuple[type, ...]] = (BaseRaw,)

    def __init__(
        self,
        method: str = "zapline-plus",
        *,
        fline: float | None = None,
        n_remove: int | str = "auto",
        n_harmonics: int | None = None,
        nfft: int = 1024,
        adaptive_params: dict | None = None,
    ) -> None:
        self.method = method
        self.fline = fline
        self.n_remove = n_remove
        self.n_harmonics = n_harmonics
        self.nfft = nfft
        self.adaptive_params = adaptive_params

    def _fit(self, inst: BaseRaw) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}.")
        fline = self.fline if self.fline is not None else inst.info["line_freq"]
        if fline is None:
            raise ValueError(
                "The power line frequency is unknown: set LineNoise(fline=50 or 60), "
                "or info['line_freq'] (BIDS: PowerLineFrequency in the *_meg/eeg.json sidecar)."
            )
        if fline >= inst.info["sfreq"] / 2:
            raise ValueError(
                f"fline ({fline} Hz) must be below the Nyquist frequency "
                f"({inst.info['sfreq'] / 2} Hz)."
            )
        self.fline_ = float(fline)
        self.picks_ = {k: list(v) for k, v in picks_by_type(inst.info).items()}
        if not self.picks_:
            raise ValueError("LineNoise found no good MEG or EEG channels.")

        self.zaplines_: dict[str, Any] = {}
        if self.method == "zapline":
            for ch_type, picks in self.picks_.items():
                model = self._make_zapline(inst.info["sfreq"], adaptive=False)
                model.fit(inst.get_data(picks=picks))
                self.zaplines_[ch_type] = model

    def _transform(self, inst: BaseRaw) -> BaseRaw:
        if self.method == "notch":
            before = {t: inst.get_data(picks=p) for t, p in self.picks_.items()}
            picks = [p for ps in self.picks_.values() for p in ps]
            inst.notch_filter(self._harmonics(inst.info["sfreq"]), picks=picks, verbose=False)
            for ch_type, ch_picks in self.picks_.items():
                self.qc_[ch_type] = _qc(
                    before[ch_type], inst.get_data(picks=ch_picks), inst.info["sfreq"], self
                )
            return inst

        for ch_type, picks in self.picks_.items():
            data = inst.get_data(picks=picks)
            if self.method == "zapline":
                model = self.zaplines_[ch_type]
                cleaned = model.transform(data)
            else:
                model = self._make_zapline(inst.info["sfreq"], adaptive=True)
                cleaned = model.fit_transform(data)
            inst._data[picks] = cleaned
            self.qc_[ch_type] = _qc(data, cleaned, inst.info["sfreq"], self)
            self.qc_[ch_type]["n_removed"] = _to_int(model.n_removed_)
        return inst

    # ------------------------------------------------------------------

    def _make_zapline(self, sfreq: float, *, adaptive: bool):
        from mne_denoise.zapline import ZapLine

        return ZapLine(
            sfreq=sfreq,
            line_freq=self.fline_,
            n_select=self.n_remove if not adaptive else "auto",
            n_harmonics=self.n_harmonics,
            nfft=self.nfft,
            adaptive=adaptive,
            adaptive_params=self.adaptive_params if adaptive else None,
            verbose=False,
        )

    def _harmonics(self, sfreq: float) -> np.ndarray:
        n_max = int((sfreq / 2 - 1e-9) // self.fline_)
        n = n_max if self.n_harmonics is None else min(self.n_harmonics + 1, n_max)
        return self.fline_ * np.arange(1, n + 1)


def _qc(before: np.ndarray, after: np.ndarray, sfreq: float, step: LineNoise) -> dict[str, Any]:
    from mne_denoise import qa

    # Flat channels have no spectrum to compare; leave them out of the metrics.
    live = before.std(axis=-1) > 0
    before, after = before[live], after[live]
    nperseg = int(min(before.shape[-1], 4 * sfreq))
    freqs, psd_before = welch(before, fs=sfreq, nperseg=nperseg)
    _, psd_after = welch(after, fs=sfreq, nperseg=nperseg)
    n_harm = len(step._harmonics(sfreq)) - 1
    fmax = min(45.0, 0.9 * sfreq / 2)
    return {
        "fline": step.fline_,
        "suppression_db": float(qa.suppression_ratio(freqs, psd_before, psd_after, step.fline_)),
        "distortion_db": float(
            np.mean(
                qa.below_noise_distortion_db(
                    freqs, psd_before, psd_after, step.fline_, fmax=fmax, n_harmonics=n_harm
                )
            )
        ),
        "distortion_db_broadband": float(
            np.mean(
                qa.below_noise_distortion_db(
                    freqs,
                    psd_before,
                    psd_after,
                    step.fline_,
                    fmax=0.9 * sfreq / 2,
                    n_harmonics=n_harm,
                )
            )
        ),
        "variance_removed_pct": float(qa.variance_removed(before, after)),
        "n_flat_skipped": int((~live).sum()),
    }


def _to_int(value: Any) -> int | list[int] | None:
    if value is None:
        return None
    if np.ndim(value) == 0:
        return int(value)
    return [int(v) for v in np.ravel(value)]
