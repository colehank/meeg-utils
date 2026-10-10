"""ICA-based artifact removal."""

from __future__ import annotations

import re
import warnings
from typing import Any, ClassVar, Self

import mne
import numpy as np
from mne.io import BaseRaw

from ..core import Step
from ..io import get_datatypes

#: Labels treated as artifacts by default, per labeler.
ARTIFACT_LABELS: dict[str, tuple[str, ...]] = {
    "iclabel": ("muscle artifact", "eye blink", "heart beat", "line noise", "channel noise"),
    "megnet": ("eye blink", "heart beat", "eye movement"),
}


class ICA(Step):
    """Decompose the data with ICA, label components and remove artifacts.

    ICA is fitted on a high-pass filtered copy of the data (1 Hz by default,
    which stabilises the decomposition) and the selected components are then
    removed from the data as they are, so the step does not change the
    data's own filtering.

    Components are excluded only when the labeler's predicted probability
    for an artifact class reaches ``threshold``; the arg-max label alone is
    not enough. Use :meth:`relabel` after fitting to apply manual labels.

    Parameters
    ----------
    n_components : int | float
        Number of components, or the fraction of variance to keep (default
        0.99). An integer above the data rank (reduced by average
        referencing and interpolated channels) is lowered to the rank.
    picks : {"auto", "eeg", "meg"}
        Modality to decompose. ``"auto"`` uses the only modality present and
        raises for MEG+EEG recordings: use one ICA step per modality.
    method : str
        ICA algorithm (default extended Infomax, which ICLabel was trained on).
    fit_params : dict | None
        Algorithm parameters; ``None`` uses ``extended=True`` for Infomax and Picard.
    fit_l_freq, fit_h_freq : float | None
        Band-pass applied to the copy ICA is fitted on. The default 1-100 Hz
        matches what ICLabel and MEGnet expect; ``fit_h_freq`` is dropped if
        it is not below Nyquist.
    labeler : {"auto", "iclabel", "megnet"} | None
        Automatic labeling via mne-icalabel; ``"auto"`` uses ICLabel for EEG
        and MEGnet for MEG. ``None`` labels nothing (use :meth:`relabel`).
    threshold : float
        Minimum predicted probability for excluding an artifact component.
    exclude_labels : list of str | None
        Labels treated as artifacts; ``None`` uses :data:`ARTIFACT_LABELS` for
        the labeler (both labelers' artifact labels when ``labeler=None``).
    max_iter : int | "auto"
        Maximum iterations.
    random_state : int | None
        Seed.
    reject_by_annotation : bool
        Ignore ``BAD_`` annotated segments when fitting.
    decim : int | None
        Use every ``decim``-th sample when fitting.

    Attributes
    ----------
    ica_ : mne.preprocessing.ICA
        The fitted ICA; its ``exclude`` holds the excluded components.
    labels_ : list of str
        Label of each component (``"unlabeled"`` without a labeler).
    proba_ : ndarray
        Predicted probability of each label (1.0 for manual labels).
    exclude_ : list of int
        Excluded components.
    qc_ : dict
        ``rank``, ``n_components``, ``excluded``, ``label_counts`` and
        ``excluded_variance`` (fraction of the fitted data's variance).

    Notes
    -----
    Figures (:meth:`plot`):

    - ``"components"``: component topographies titled with label and
      probability, excluded components in red.
    - ``"labels"``: predicted probability of each component's label, with
      the exclusion threshold.
    - ``"properties"`` (needs ``inst``): :meth:`mne.preprocessing.ICA.plot_properties`
      of the excluded components (the first five if none are excluded);
      a list of figures.
    - ``"overlay"`` (needs ``inst``): signals before and after removing the
      excluded components (:meth:`mne.preprocessing.ICA.plot_overlay`).
    """

    accepts: ClassVar[tuple[type, ...]] = (BaseRaw,)
    plot_kinds: ClassVar[dict[str, bool]] = {
        "components": False,
        "labels": False,
        "properties": True,
        "overlay": True,
    }

    def __init__(
        self,
        n_components: int | float = 0.99,
        *,
        picks: str = "auto",
        method: str = "infomax",
        fit_params: dict | None = None,
        fit_l_freq: float | None = 1.0,
        fit_h_freq: float | None = 100.0,
        labeler: str | None = "auto",
        threshold: float = 0.8,
        exclude_labels: list[str] | tuple[str, ...] | None = None,
        max_iter: int | str = "auto",
        random_state: int | None = 42,
        reject_by_annotation: bool = True,
        decim: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.picks = picks
        self.method = method
        self.fit_params = fit_params
        self.fit_l_freq = fit_l_freq
        self.fit_h_freq = fit_h_freq
        self.labeler = labeler
        self.threshold = threshold
        self.exclude_labels = exclude_labels
        self.max_iter = max_iter
        self.random_state = random_state
        self.reject_by_annotation = reject_by_annotation
        self.decim = decim

    def _fit(self, inst: BaseRaw) -> None:
        modality = self._modality(inst)
        fit_raw = inst.copy().pick(modality, exclude="bads")
        h_freq = self.fit_h_freq
        if h_freq is not None and h_freq >= fit_raw.info["sfreq"] / 2:
            h_freq = None
        if self.fit_l_freq is not None or h_freq is not None:
            fit_raw.filter(self.fit_l_freq, h_freq, verbose=False)

        # The data rank can exceed the rank in info (after SSS, steps such as ZapLine
        # remove patterns estimated in sensor space); ICA cannot use more than either.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            data_rank = mne.compute_rank(fit_raw, rank=None, verbose=False)
        info_rank = mne.compute_rank(fit_raw, rank="info", verbose=False)
        rank = int(sum(min(r, info_rank.get(t, r)) for t, r in data_rank.items()))
        n_components: int | float = self.n_components
        if isinstance(n_components, int | np.integer) and n_components > rank:
            self.qc_["note"] = f"n_components lowered from {n_components} to the data rank {rank}."
            n_components = rank

        fit_params = self.fit_params
        if fit_params is None and self.method in ("infomax", "picard"):
            fit_params = {"extended": True}
        ica = mne.preprocessing.ICA(
            n_components=n_components,
            method=self.method,
            fit_params=fit_params,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        ica.fit(
            fit_raw,
            decim=self.decim,
            reject_by_annotation=self.reject_by_annotation,
            verbose=False,
        )
        self.ica_ = ica
        self.modality_ = modality

        labeler = self._labeler(modality)
        self.labeler_ = labeler
        if labeler is None:
            self.labels_ = ["unlabeled"] * ica.n_components_
            self.proba_ = np.full(ica.n_components_, np.nan)
        else:
            from mne_icalabel import label_components

            result = label_components(fit_raw, ica, method=labeler)
            self.labels_ = list(result["labels"])
            self.proba_ = np.asarray(result["y_pred_proba"], dtype=float)

        self.qc_.update(rank=rank, n_components=int(ica.n_components_), labeler=labeler)
        self._update_exclude(fit_raw)

    def _transform(self, inst: BaseRaw) -> BaseRaw:
        self.ica_.apply(inst, exclude=self.exclude_, verbose=False)
        return inst

    def relabel(self, labels: dict[int, str], inst: BaseRaw | None = None) -> Self:
        """Override component labels, e.g. after visual inspection.

        Manually labeled components get probability 1.0, so they are excluded
        if (and only if) their label is an artifact label.

        Parameters
        ----------
        labels : dict of int to str
            Component index mapped to its label.
        inst : Raw | None
            Data to recompute ``qc_["excluded_variance"]`` on; without it the
            metric is cleared, as it no longer matches the excluded components.

        Returns
        -------
        self : ICA
            The step, with ``labels_``, ``proba_`` and ``exclude_`` updated.
        """
        if not hasattr(self, "ica_"):
            raise RuntimeError("Fit the ICA step before relabeling components.")
        for idx, label in labels.items():
            if not 0 <= idx < len(self.labels_):
                raise IndexError(
                    f"Component {idx} does not exist ({len(self.labels_)} components)."
                )
            self.labels_[idx] = label
            self.proba_[idx] = 1.0
        self.qc_.setdefault("manual_labels", {}).update({str(k): v for k, v in labels.items()})
        self._update_exclude(inst.copy().pick(self.modality_, exclude="bads") if inst else None)
        return self

    # ------------------------------------------------------------------
    # Figures

    def _plot_components(self, inst: BaseRaw | None, **kwargs: Any) -> Any:
        figs = self.ica_.plot_components(show=False, **kwargs)
        for fig in figs if isinstance(figs, list) else [figs]:
            for ax in fig.axes:
                title = ax.get_title()
                match = re.match(r"ICA(\d+)", title)  # "ICA003", or "ICA003 (mag)" for Neuromag
                if match is None:
                    continue
                idx = int(match.group(1))
                proba = self.proba_[idx]
                suffix = "" if np.isnan(proba) else f" {proba:.2f}"
                ax.set_title(
                    f"{title}\n{self.labels_[idx]}{suffix}",
                    color="C3" if idx in self.exclude_ else "k",
                    fontsize=8,
                )
        return figs

    def _plot_labels(self, inst: BaseRaw | None) -> Any:
        import matplotlib.pyplot as plt

        n = len(self.labels_)
        fig, ax = plt.subplots(figsize=(min(11.0, max(6.0, 0.3 * n)), 3.6), layout="constrained")
        names = sorted(set(self.labels_))
        cmap = plt.get_cmap("tab10")
        colors = [cmap(names.index(label) % 10) for label in self.labels_]
        ax.bar(np.arange(n), np.nan_to_num(self.proba_), color=colors)
        for idx in self.exclude_:
            ax.annotate(
                "x", (idx, np.nan_to_num(self.proba_[idx])), ha="center", va="bottom", color="C3"
            )
        ax.axhline(self.threshold, color="C3", ls="--", lw=1)
        ax.set(xlabel="Component", ylabel="Label probability", ylim=(0, 1.08), xlim=(-1, n))
        ax.set_xticks(np.arange(0, n, 1 if n <= 30 else 5))
        handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i % 10)) for i in range(len(names))]
        fig.legend(
            handles,
            names,
            ncols=len(names),
            fontsize="small",
            frameon=False,
            loc="outside lower center",
        )
        fig.suptitle(f"ICA labels ({self.labeler_ or 'manual'}); x = excluded")
        return fig

    def _plot_properties(self, inst: BaseRaw | None, **kwargs: Any) -> Any:
        assert inst is not None
        picks = self.exclude_ or list(range(min(5, self.ica_.n_components_)))
        return self.ica_.plot_properties(
            inst.copy().pick(self.ica_.ch_names), picks=picks, show=False, verbose=False, **kwargs
        )

    def _plot_overlay(self, inst: BaseRaw | None, **kwargs: Any) -> Any:
        assert inst is not None
        return self.ica_.plot_overlay(
            inst.copy().pick(self.ica_.ch_names),
            exclude=self.exclude_,
            show=False,
            verbose=False,
            **kwargs,
        )

    # ------------------------------------------------------------------

    def _modality(self, inst: BaseRaw) -> str:
        datatypes = get_datatypes(inst)
        if self.picks == "auto":
            if len(datatypes) != 1:
                raise ValueError(
                    "The recording contains MEG and EEG; use one ICA step per modality "
                    "with picks='meg' and picks='eeg'."
                )
            return next(iter(datatypes))
        if self.picks not in ("eeg", "meg"):
            raise ValueError(f"picks must be 'auto', 'eeg' or 'meg', got {self.picks!r}.")
        if self.picks not in datatypes:
            raise ValueError(f"picks={self.picks!r} but the recording has no such channels.")
        return self.picks

    def _labeler(self, modality: str) -> str | None:
        if self.labeler is None:
            return None
        if self.labeler == "auto":
            return "iclabel" if modality == "eeg" else "megnet"
        if self.labeler not in ARTIFACT_LABELS:
            raise ValueError(
                f"labeler must be 'auto', 'iclabel', 'megnet' or None, got {self.labeler!r}."
            )
        return self.labeler

    def _update_exclude(self, data: BaseRaw | None) -> None:
        if self.exclude_labels is not None:
            artifact = set(self.exclude_labels)
        elif self.labeler_ is not None:
            artifact = set(ARTIFACT_LABELS[self.labeler_])
        else:  # manual labels only: accept either labeler's vocabulary
            artifact = {label for labels in ARTIFACT_LABELS.values() for label in labels}
        self.exclude_ = [
            i
            for i, (label, proba) in enumerate(zip(self.labels_, self.proba_, strict=True))
            if label in artifact and proba >= self.threshold
        ]
        self.ica_.exclude = list(self.exclude_)
        counts: dict[str, Any] = {}
        for label in self.labels_:
            counts[label] = counts.get(label, 0) + 1
        self.qc_.update(
            excluded=list(self.exclude_),
            labels=list(self.labels_),
            proba=[None if np.isnan(p) else round(float(p), 4) for p in self.proba_],
            label_counts=counts,
        )
        if not self.exclude_:
            self.qc_["excluded_variance"] = {}
        elif data is not None:
            ratios = self.ica_.get_explained_variance_ratio(data, components=self.exclude_)
            self.qc_["excluded_variance"] = {k: float(v) for k, v in ratios.items()}
        else:
            self.qc_.pop("excluded_variance", None)
