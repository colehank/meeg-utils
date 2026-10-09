"""Data-driven epoch rejection and repair with autoreject."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from mne.epochs import BaseEpochs

from ..core import Step
from ._utils import DATA_CH_TYPES

AUTOREJECT_METHODS = ("local", "global")


class AutoReject(Step):
    """Reject or repair bad epochs with thresholds learned from the data.

    Wraps autoreject (Jas et al., 2017, NeuroImage 159:417):

    - ``"local"``: :class:`autoreject.AutoReject` learns a peak-to-peak
      threshold per channel by cross-validation; in each epoch the worst
      channels are interpolated, and epochs with too many bad channels are
      dropped. Needs sensor positions.
    - ``"global"``: :func:`autoreject.get_rejection_threshold` learns one
      peak-to-peak threshold per channel type; epochs exceeding it are
      dropped.

    Parameters
    ----------
    method : {"local", "global"}
        See above.
    n_interpolate : list of int | None
        Local: candidate numbers of channels to interpolate per epoch
        (``None``: autoreject's default, 1, 4 and 32).
    consensus : list of float | None
        Local: candidate fractions of bad channels that make an epoch bad
        (``None``: autoreject's default, 0 to 1 in steps of 0.1).
    cv : int
        Cross-validation folds (autoreject defaults: 10 local, 5 global).
    thresh_method : {"bayesian_optimization", "random_search"}
        Local: how thresholds are searched.
    random_state : int | None
        Seed.
    n_jobs : int
        Parallel jobs.

    Attributes
    ----------
    reject_log_ : autoreject.RejectLog
        Bad epochs and channel labels on the fitted data.
    thresholds_ : dict
        Learned thresholds: per channel (local) or per channel type (global).
    qc_ : dict
        ``n_epochs``, ``n_dropped``, ``fraction_dropped``,
        ``n_interpolated`` (channels interpolated per epoch, mean) and the
        most interpolated channels.

    Notes
    -----
    Figures (:meth:`plot`): ``"reject_log"`` (epochs x channels: bad,
    interpolated, dropped) and ``"thresholds"``.
    """

    accepts: ClassVar[tuple[type, ...]] = (BaseEpochs,)
    drops_epochs: ClassVar[bool] = True
    plot_kinds: ClassVar[dict[str, bool]] = {"reject_log": False, "thresholds": False}

    def __init__(
        self,
        method: str = "local",
        *,
        n_interpolate: list[int] | None = None,
        consensus: list[float] | None = None,
        cv: int | None = None,
        thresh_method: str = "bayesian_optimization",
        random_state: int | None = 42,
        n_jobs: int = 1,
    ) -> None:
        self.method = method
        self.n_interpolate = n_interpolate
        self.consensus = consensus
        self.cv = cv
        self.thresh_method = thresh_method
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _fit(self, inst: BaseEpochs) -> None:
        import autoreject

        if self.method not in AUTOREJECT_METHODS:
            raise ValueError(f"method must be one of {AUTOREJECT_METHODS}, got {self.method!r}")
        picks = _data_picks(inst)
        ch_names = [inst.ch_names[p] for p in picks]
        if self.method == "local":
            ar = autoreject.AutoReject(
                n_interpolate=None
                if self.n_interpolate is None
                else np.asarray(self.n_interpolate),
                consensus=None if self.consensus is None else np.asarray(self.consensus),
                cv=10 if self.cv is None else self.cv,
                picks=picks,
                thresh_method=self.thresh_method,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=False,
            )
            ar.fit(inst)
            self.ar_ = ar
            self.reject_log_ = ar.get_reject_log(inst)
            self.thresholds_ = {
                ch: float(ar.threshes_[ch]) for ch in ch_names if ch in ar.threshes_
            }
            self.qc_.update(
                n_interpolate=_as_plain(ar.n_interpolate_), consensus=_as_plain(ar.consensus_)
            )
        else:
            reject = autoreject.get_rejection_threshold(
                inst, random_state=self.random_state, cv=5 if self.cv is None else self.cv,
                verbose=False,
            )  # fmt: skip
            self.thresholds_ = {k: float(v) for k, v in reject.items()}
            self.reject_log_ = _global_reject_log(inst, picks, self.thresholds_)
        labels = self.reject_log_.labels
        bad = self.reject_log_.bad_epochs
        interpolated = (labels == 2).sum(axis=0)
        self.qc_.update(
            n_epochs=len(inst),
            n_dropped=int(bad.sum()),
            fraction_dropped=round(float(bad.mean()), 4),
            n_interpolated=round(float((labels == 2).sum(axis=1).mean()), 2),
            most_interpolated={
                self.reject_log_.ch_names[i]: int(interpolated[i])
                for i in np.argsort(interpolated)[::-1][:10]
                if interpolated[i] > 0
            },
        )

    def _transform(self, inst: BaseEpochs) -> BaseEpochs:
        if self.method == "local":
            return self.ar_.transform(inst)
        return inst.drop_bad(reject=self.thresholds_, verbose=False)

    def _plot_reject_log(self, inst: Any) -> Any:
        return self.reject_log_.plot(orientation="horizontal", show=False)

    def _plot_thresholds(self, inst: Any) -> Any:
        import matplotlib.pyplot as plt

        from ._plotting import UNITS

        fig, ax = plt.subplots(figsize=(6, 3.2), layout="constrained")
        if self.method == "global":
            types = list(self.thresholds_)
            values = [self.thresholds_[t] * UNITS.get(t, (1.0, ""))[0] for t in types]
            ax.bar(types, values, color="C0")
            ax.set(ylabel="Peak-to-peak threshold (µV / fT / fT/cm)", title="Global thresholds")
        else:
            info = self.ar_.info if hasattr(self.ar_, "info") else None
            by_type: dict[str, list[float]] = {}
            for ch, value in self.thresholds_.items():
                ch_type = _channel_type(ch, info)
                by_type.setdefault(ch_type, []).append(value * UNITS.get(ch_type, (1.0, ""))[0])
            for i, (ch_type, values) in enumerate(by_type.items()):
                ax.hist(values, bins=30, alpha=0.7, color=f"C{i}", label=ch_type.upper())
            ax.set(
                xlabel="Peak-to-peak threshold (µV / fT / fT/cm)",
                ylabel="Channels",
                title="Channel thresholds (local autoreject)",
            )
            ax.legend(frameon=False)
        return fig


def _data_picks(inst: BaseEpochs) -> np.ndarray:
    import mne

    return np.asarray(mne.pick_types(inst.info, meg=True, eeg=True, ref_meg=False, exclude="bads"))


def _global_reject_log(inst: BaseEpochs, picks: np.ndarray, reject: dict[str, float]) -> Any:
    """RejectLog of global rejection: a channel is bad where it exceeds its type's threshold."""
    import autoreject

    data = inst.get_data(picks=picks)
    ptp = data.max(axis=-1) - data.min(axis=-1)
    types = inst.get_channel_types(picks=picks)
    limits = np.array([reject.get(t, np.inf) for t in types])
    labels = (ptp > limits).astype(float)
    bad = labels.any(axis=1)
    return autoreject.RejectLog(bad, labels, [inst.ch_names[p] for p in picks])


def _channel_type(ch: str, info: Any) -> str:
    if info is not None and ch in info["ch_names"]:
        import mne

        ch_type = mne.channel_type(info, info["ch_names"].index(ch))
        return ch_type if ch_type in DATA_CH_TYPES else "other"
    return "eeg"


def _as_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _as_plain(v) for k, v in value.items()}
    if hasattr(value, "item"):
        return value.item()
    return value
