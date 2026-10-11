"""Different processing for different channel types."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Self

import mne
import numpy as np
from mne.epochs import BaseEpochs
from mne.io import BaseRaw

from ..core import Step
from ..core.step import Inst

#: Branch keys and the channel types they select ("meg" keeps the CTF
#: reference sensors with the MEG channels, as gradient compensation needs them).
CHANNEL_GROUPS: dict[str, tuple[str, ...]] = {
    "meg": ("mag", "grad", "ref_meg"),
    "mag": ("mag",),
    "grad": ("grad",),
    "eeg": ("eeg",),
    "eog": ("eog",),
    "ecg": ("ecg",),
    "emg": ("emg",),
    "seeg": ("seeg",),
    "ecog": ("ecog",),
    "dbs": ("dbs",),
    "misc": ("misc",),
}


class ByChannelType(Step):
    """Process channel types with separate steps, like scikit-learn's ``ColumnTransformer``.

    Each branch selects some channel types and runs its own steps on a copy
    holding only those channels; the results are written back, so all
    channels stay in place. Typical use: MEG and EEG recorded together, with
    a different ICA or bad-channel method for each.

    Parameters
    ----------
    branches : dict
        Channel group (``"meg"``, ``"mag"``, ``"grad"``, ``"eeg"``, ``"eog"``,
        ...) mapped to a step or a list of ``(name, step)``. Groups must not
        overlap. ``"meg"`` includes CTF reference sensors.
    on_missing : {"raise", "ignore"}
        What to do when the data have no channels of a branch.

    Attributes
    ----------
    pipelines_ : dict
        Group mapped to the fitted :class:`~meeg_utils.Pipeline` of the branch
        (it shares its step objects with ``branches``).
    qc_ : dict
        Group mapped to the ``qc_`` of its pipeline (``{step: metrics}``).

    Notes
    -----
    Steps in a branch must keep the channels and the time axis (no
    resampling, epoching or epoch rejection). What the branch changes is
    carried back: the data, new annotations (e.g. from :class:`BadSegments`),
    ``info["bads"]`` and the channel descriptions
    of its channels (e.g. CTF compensation), projectors added or removed,
    the EEG reference flag, and, from a MEG branch, the device-to-head
    transform and the Maxwell-filtering history. ``info["highpass"]`` /
    ``info["lowpass"]`` describe all channels and are left unchanged, as MNE
    does when filtering a subset of channels.

    Parameters of branch steps are ``<group>__<param>`` (a single step) or
    ``<group>__<step>__<param>``, e.g.
    ``pipe.set_params(by_type__eeg__ica__n_components=15)``.

    Figures (:meth:`plot`) are those of the branch steps, named
    ``"<group>/<step>/<kind>"``.

    Examples
    --------
    >>> from meeg_utils import steps as S
    >>> step = S.ByChannelType({
    ...     "meg": [("ica", S.ICA(picks="meg", labeler="megnet"))],
    ...     "eeg": [("bads", S.BadChannels("prep")), ("ica", S.ICA(picks="eeg"))],
    ... })
    """

    accepts: ClassVar[tuple[type, ...]] = (BaseRaw, BaseEpochs)
    adds_annotations: ClassVar[bool] = True  # branch steps may add BAD_ segments

    def __init__(self, branches: dict[str, Any], *, on_missing: str = "raise") -> None:
        self.branches = branches
        self.on_missing = on_missing

    # ------------------------------------------------------------------
    # Fitting and transforming

    def fit_transform(
        self, inst: Inst, y: Any = None, *, system: str | None = None, copy: bool = True
    ) -> Inst:
        """Fit every branch and return the processed data, running each branch once.

        Parameters
        ----------
        inst : Raw | Epochs
            The data.
        y : None
            Ignored; present for scikit-learn API compatibility.
        system : str | None
            Acquisition system of the data; detected if ``None``.
        copy : bool
            Whether to leave ``inst`` untouched.

        Returns
        -------
        Raw | Epochs
            The processed data.
        """
        self._outputs: dict[str, tuple[list, Inst]] | None = {}
        try:
            self.fit(inst, system=system)
            outputs = self._outputs
        finally:
            self._outputs = None
        out = inst.copy() if copy else inst
        for key, (projs, sub) in outputs.items():
            self._write_back(out, key, projs, sub)
        return out

    def _fit(self, inst: Inst) -> None:
        from ..core import Pipeline

        if self.on_missing not in ("raise", "ignore"):
            raise ValueError(f"on_missing must be 'raise' or 'ignore', got {self.on_missing!r}.")
        self.pipelines_: dict[str, Pipeline] = {}
        self.picks_: dict[str, list[str]] = {}
        taken: dict[str, str] = {}
        for key, steps in _branches(self.branches).items():
            names = self._channels(inst.info, key)
            if not names:
                continue
            for ch in names:
                if ch in taken:
                    raise ValueError(
                        f"Channel {ch} is in both the {taken[ch]!r} and {key!r} branches."
                    )
                taken[ch] = key
            for name, step in steps:
                if step.changes_channels or step.changes_times or step.drops_epochs or step.returns:
                    raise ValueError(
                        f"Step {name!r} ({type(step).__name__}) of the {key!r} branch changes the "
                        "channels, the time axis or the data type; put it outside ByChannelType."
                    )
            pipe = Pipeline(steps, copy=False)
            sub = inst.copy().pick(names)
            if self._outputs is not None:
                projs = list(sub.info["projs"])
                self._outputs[key] = (projs, pipe.fit_transform(sub))
            else:
                pipe.fit(sub)
            self.pipelines_[key] = pipe
            self.picks_[key] = names
            self.qc_[key] = pipe.qc_

    _outputs: dict[str, tuple[list, Inst]] | None = None

    def _transform(self, inst: Inst) -> Inst:
        for key, pipe in self.pipelines_.items():
            missing = [ch for ch in self.picks_[key] if ch not in inst.ch_names]
            if missing:
                raise ValueError(f"The data lack channels of the {key!r} branch: {missing}.")
            sub = inst.copy().pick(self.picks_[key])
            projs = list(sub.info["projs"])
            self._write_back(inst, key, projs, pipe.transform(sub))
        return inst

    def _channels(self, info: mne.Info, key: str) -> list[str]:
        if key not in CHANNEL_GROUPS:
            raise ValueError(f"Unknown channel group {key!r}; use one of {sorted(CHANNEL_GROUPS)}.")
        types = CHANNEL_GROUPS[key]
        names = [
            ch for ch, t in zip(info.ch_names, info.get_channel_types(), strict=True) if t in types
        ]
        if not names and self.on_missing == "raise":
            raise ValueError(
                f"The data have no {key} channels; remove the {key!r} branch or use "
                "on_missing='ignore'."
            )
        return names

    @staticmethod
    def _write_back(inst: Inst, key: str, projs_before: list, sub: Inst) -> None:
        """Copy what a branch changed into ``inst`` (in place)."""
        info = inst.info
        idx = [info.ch_names.index(ch) for ch in sub.ch_names]
        if isinstance(inst, BaseEpochs):
            inst._data[:, idx] = sub.get_data()
        else:
            inst._data[idx] = sub.get_data()
        branch = set(sub.ch_names)
        after = sub.info["projs"]
        removed = [p for p in projs_before if not any(_same_proj(p, q) for q in after)]
        added = [q for q in after if not any(_same_proj(p, q) for p in projs_before)]
        with info._unlock():
            for i, ch in zip(idx, sub.info["chs"], strict=True):
                info["chs"][i] = ch
            info["bads"] = [ch for ch in info["bads"] if ch not in branch] + list(sub.info["bads"])
            info["projs"] = [
                p for p in info["projs"] if not any(_same_proj(p, q) for q in removed)
            ] + added
            if key == "eeg" or "eeg" in sub.get_channel_types(unique=True):
                info["custom_ref_applied"] = sub.info["custom_ref_applied"]
            if set(sub.get_channel_types(unique=True)) & {"mag", "grad"}:
                info["dev_head_t"] = sub.info["dev_head_t"]
                info["proc_history"] = sub.info["proc_history"]
        info._check_consistency()
        if isinstance(inst, BaseRaw):  # e.g. BAD_ segments found by the branch
            existing = set(
                zip(
                    inst.annotations.onset,
                    inst.annotations.duration,
                    inst.annotations.description,
                    strict=True,
                )
            )
            for annot in sub.annotations:
                if (annot["onset"], annot["duration"], annot["description"]) not in existing:
                    inst.annotations.append(annot["onset"], annot["duration"], annot["description"])

    # ------------------------------------------------------------------
    # Parameters

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters, including those of the branch steps.

        Parameters
        ----------
        deep : bool
            If ``True``, also return ``<group>__<param>`` (single step) or
            ``<group>__<step>__<param>``.

        Returns
        -------
        dict
            Parameter names mapped to their values.
        """
        params: dict[str, Any] = {"branches": self.branches, "on_missing": self.on_missing}
        if deep:
            for key, value in self.branches.items():
                if isinstance(value, Step):
                    for name, v in value.get_params(deep=True).items():
                        params[f"{key}__{name}"] = v
                    continue
                for name, step in _as_steps(value):
                    params[f"{key}__{name}"] = step
                    for param, v in step.get_params(deep=True).items():
                        params[f"{key}__{name}__{param}"] = v
        return params

    def set_params(self, **params: Any) -> Self:
        """Set parameters, including ``<group>__[<step>__]<param>`` of branch steps.

        Parameters
        ----------
        **params
            ``branches``, ``on_missing`` or nested branch parameters.

        Returns
        -------
        self : ByChannelType
            The step.
        """
        if "branches" in params:
            self.branches = params.pop("branches")
        if "on_missing" in params:
            self.on_missing = params.pop("on_missing")
        for full, value in params.items():
            key, _, rest = full.partition("__")
            if key not in self.branches or not rest:
                raise ValueError(f"Invalid parameter {full!r} for ByChannelType.")
            branch = self.branches[key]
            if isinstance(branch, Step):
                branch.set_params(**{rest: value})
                continue
            steps = dict(_as_steps(branch))
            name, _, param = rest.partition("__")
            if name not in steps:
                raise ValueError(f"No step {name!r} in the {key!r} branch.")
            if param:
                steps[name].set_params(**{param: value})
            else:
                self.branches[key] = [(n, value if n == name else s) for n, s in _as_steps(branch)]
        return self

    def __sklearn_clone__(self) -> ByChannelType:
        """Return an unfitted copy with cloned branch steps."""
        from sklearn.base import clone

        branches = {
            key: clone(value)
            if isinstance(value, Step)
            else [(name, clone(step)) for name, step in _as_steps(value)]
            for key, value in self.branches.items()
        }
        return type(self)(branches, on_missing=self.on_missing)

    # ------------------------------------------------------------------
    # Figures

    @property
    def plot_kinds(self) -> dict[str, bool]:  # type: ignore[override]
        """Figures of the branch steps, as ``"<group>/<step>/<kind>"``."""
        return {
            f"{key}/{name}/{kind}": needs_data
            for key, steps in _branches(self.branches).items()
            for name, step in steps
            for kind, needs_data in step.plot_kinds.items()
        }

    def _plot_unavailable(self, kind: str) -> str | None:
        key, name, sub_kind = kind.split("/", 2)
        if key not in getattr(self, "pipelines_", {}):
            return f"the data had no {key} channels"
        return self.pipelines_[key].named_steps[name]._plot_unavailable(sub_kind)

    def plot(self, kind: str | None = None, *, inst: Any = None, **kwargs: Any) -> dict[str, Any]:
        """Draw the figures of the branch steps.

        Parameters
        ----------
        kind : str | None
            ``"<group>/<step>/<kind>"``, or ``None`` for every figure the
            inputs allow.
        inst : Raw | Epochs | None
            Data for the figures that need it; each branch replays its steps
            on its own channels, so every step plots the data it received.
        **kwargs
            Passed to the plotting method of ``kind``.

        Returns
        -------
        dict
            Kind mapped to its figure.
        """
        from sklearn.exceptions import NotFittedError

        if not hasattr(self, "pipelines_"):
            raise NotFittedError("This ByChannelType instance is not fitted yet.")
        if kind is not None:
            if kind not in self.plot_kinds:
                raise ValueError(
                    f"ByChannelType has no {kind!r} plot; available: {sorted(self.plot_kinds)}."
                )
            reason = self._plot_unavailable(kind)
            if reason is not None:
                raise ValueError(f"No {kind!r} plot: {reason}.")
            key, name, sub_kind = kind.split("/", 2)
            pipe = self.pipelines_[key]
            data = None
            if inst is not None:
                data = inst.copy().pick(self.picks_[key])
                index = [n for n, _ in pipe.steps].index(name)
                if index:
                    data = pipe[:index].transform(data)
            return {kind: pipe.named_steps[name].plot(sub_kind, inst=data, **kwargs)[sub_kind]}
        if kwargs:
            raise TypeError("Plot options can only be passed together with kind.")
        figures: dict[str, Any] = {}
        for key, pipe in self.pipelines_.items():
            data = None if inst is None else inst.copy().pick(self.picks_[key])
            for name, step_figures in pipe.plot(inst=data).items():
                for sub_kind, figure in step_figures.items():
                    figures[f"{key}/{name}/{sub_kind}"] = figure
        return figures


def _as_steps(value: Any) -> list[tuple[str, Step]]:
    """A branch as a list of (name, step); a lone step is named after its class."""
    if isinstance(value, Step):
        return [(type(value).__name__.lower(), value)]
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError(f"A branch must be a step or a list of (name, step), got {value!r}.")
    steps = []
    for item in value:
        if len(item) != 2 or not isinstance(item[1], Step):
            raise TypeError(f"A branch must be a list of (name, step), got item {item!r}.")
        steps.append((str(item[0]), item[1]))
    return steps


def _branches(branches: dict[str, Any]) -> dict[str, list[tuple[str, Step]]]:
    if not isinstance(branches, dict) or not branches:
        raise ValueError("branches must be a non-empty dict of channel group -> steps.")
    return {key: _as_steps(value) for key, value in branches.items()}


def _same_proj(a: Any, b: Any) -> bool:
    return (
        a["desc"] == b["desc"]
        and list(a["data"]["col_names"]) == list(b["data"]["col_names"])
        and np.array_equal(a["data"]["data"], b["data"]["data"])
    )
