"""Regressing reference or artifact channels out of the data."""

from __future__ import annotations

from typing import Any, ClassVar

import mne
import numpy as np
from mne import Evoked
from mne.epochs import BaseEpochs
from mne.io import BaseRaw

from ..core import Step
from ..core.step import Inst
from ._plotting import plot_psd_comparison, psd_summary
from ._utils import picks_by_type


class Regression(Step):
    """Remove what reference or artifact channels record from the data, by linear regression.

    Wraps :class:`mne.preprocessing.EOGRegression`: each data channel is
    regressed on the artifact channels (means removed) and the fitted
    contribution is subtracted. Coefficients are learned in ``fit`` and
    reused by ``transform``.

    - ``artifact="ref_meg"`` (default): environmental noise recorded by the
      reference magnetometers of KIT/Ricoh, BTi/4D, Artemis123 and CTF
      systems is removed from the MEG channels (least-squares reference
      regression; for CTF, the vendor's synthetic gradiometers,
      :class:`Reference` with ``ctf_grade=3``, are the usual alternative and
      the data must be at grade 0).
    - ``artifact="eog"`` (or ``"ecg"``, a list of channels): regression of
      ocular artifacts (Gratton et al., 1983, Electroencephalogr Clin
      Neurophysiol 55:468). It also removes the brain activity the EOG
      electrodes pick up; ICA is usually preferable. Set the EEG reference
      first (:class:`Reference`); MNE requires it.

    Parameters
    ----------
    artifact : str | list of str
        Channel type or names of the predictors.
    picks : str | list of str | None
        Channels to clean; ``None`` uses the MEG channels for
        ``artifact="ref_meg"`` and all data channels otherwise. Bad channels
        are left out.

    Attributes
    ----------
    model_ : mne.preprocessing.EOGRegression
        The fitted regression (``coef_``: data channels x artifact channels).
    qc_ : dict
        ``n_channels``, ``n_artifact_channels``, ``artifact_channels``,
        ``variance_removed_pct`` (fitted data) and per channel type
        ``power_change_db`` (last transform). Both ignore offsets (the mean
        over time), which the regression leaves alone.
    psd_ : dict
        Per channel type, PSDs of the cleaned channels before and after the
        last transform.

    Notes
    -----
    Figures (:meth:`plot`): ``"psd"``, spectra before and after.
    """

    accepts: ClassVar[tuple[type, ...]] = (BaseRaw, BaseEpochs, Evoked)
    plot_kinds: ClassVar[dict[str, bool]] = {"psd": False}

    def __init__(self, artifact: str | list[str] = "ref_meg", *, picks: Any = None) -> None:
        self.artifact = artifact
        self.picks = picks

    def _fit(self, inst: Inst) -> None:
        from mne.preprocessing import EOGRegression

        artifact = self.artifact if isinstance(self.artifact, str) else list(self.artifact)
        if artifact == "ref_meg":
            if not mne.pick_types(inst.info, meg=False, ref_meg=True).size:
                raise ValueError(
                    f"The data have no reference MEG channels ({self.system_!r} system); "
                    "artifact='ref_meg' needs a system with reference sensors (KIT/Ricoh, BTi, "
                    "Artemis123, CTF)."
                )
            if self.system_ == "ctf" and inst.compensation_grade != 0:
                raise ValueError(
                    f"The CTF data are already compensated (grade {inst.compensation_grade}); "
                    "regress at grade 0, or keep the synthetic gradiometers instead."
                )
        spec = (
            self.picks if self.picks is not None else ("meg" if artifact == "ref_meg" else "data")
        )
        art = _resolve(inst.info, artifact, exclude_bads=False)
        picked = np.setdiff1d(_resolve(inst.info, spec, exclude_bads=True), art)
        if not art.size or not picked.size:
            raise ValueError(f"No {artifact} channels or no channels to clean (picks={spec!r}).")
        model = EOGRegression(picks=picked, exclude=[], picks_artifact=art, proj=False)
        model.fit(inst)
        self.model_ = model
        self.ch_names_ = [inst.ch_names[p] for p in picked]
        self.qc_.update(
            n_channels=len(picked),
            n_artifact_channels=len(art),
            artifact_channels=[inst.ch_names[p] for p in art],
            variance_removed_pct=round(_variance_removed(inst, model, picked), 2),
        )

    def _transform(self, inst: Inst) -> Inst:
        types = {
            t: [inst.ch_names[i] for i in idx if inst.ch_names[i] in self.ch_names_]
            for t, idx in picks_by_type(inst.info, exclude_bads=False).items()
        }
        types = {t: names for t, names in types.items() if names}
        sfreq = inst.info["sfreq"]
        before = {t: _centered(inst.get_data(names)) for t, names in types.items()}
        self.model_.apply(inst, copy=False)
        self.psd_ = {}
        change: dict[str, float] = {}
        for t, names in types.items():
            after = _centered(inst.get_data(names))
            freqs, psd_before = psd_summary(before[t], sfreq)
            self.psd_[t] = {
                "freqs": freqs,
                "before": psd_before,
                "after": psd_summary(after, sfreq)[1],
            }
            power = float(np.mean(before[t] ** 2))
            if power > 0:
                change[t] = round(10 * np.log10(np.mean(after**2) / power), 2)
        self.qc_["power_change_db"] = change
        return inst

    def _plot_unavailable(self, kind: str) -> str | None:
        if kind == "psd" and not hasattr(self, "psd_"):
            return "the step has not transformed data yet (use fit_transform)"
        return None

    def _plot_psd(self, inst: Any) -> Any:
        return plot_psd_comparison(
            self.psd_, title=f"Spectra before and after regressing {self.artifact}"
        )


def _centered(data: np.ndarray) -> np.ndarray:
    """Data with the mean over time removed (the regression leaves offsets alone)."""
    return np.asarray(data - data.mean(axis=-1, keepdims=True))


def _variance_removed(inst: Inst, model: Any, picks: np.ndarray) -> float:
    """Percentage of the (mean-removed) variance of the cleaned channels explained by the fit."""
    data = inst.get_data(picks)
    data = data - data.mean(axis=-1, keepdims=True)
    cleaned = model.apply(inst.copy(), copy=False).get_data(picks)
    cleaned = cleaned - cleaned.mean(axis=-1, keepdims=True)
    total = float(np.sum(data**2))
    return 100 * (1 - float(np.sum(cleaned**2)) / total) if total > 0 else 0.0


def _resolve(info: mne.Info, spec: Any, *, exclude_bads: bool) -> np.ndarray:
    """Channel indices for a type ("meg", "data", "eog", "ref_meg", ...) or a list of names."""
    bads = set(info["bads"]) if exclude_bads else set()
    types = info.get_channel_types()
    if isinstance(spec, str):
        wanted = {"meg": {"mag", "grad"}, "data": {"mag", "grad", "eeg", "seeg", "ecog", "dbs"}}
        allowed = wanted.get(spec, {spec})
        idx = [i for i, t in enumerate(types) if t in allowed]
    else:
        missing = [ch for ch in spec if ch not in info.ch_names]
        if missing:
            raise ValueError(f"Channels not in the data: {missing}.")
        idx = [info.ch_names.index(ch) for ch in spec]
    return np.array([i for i in idx if info.ch_names[i] not in bads], dtype=int)
