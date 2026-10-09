"""Composition of processing steps."""

from __future__ import annotations

import importlib
import platform
import time
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self, TypeAlias

from mne_bids import BIDSPath
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from ..io import detect_system, read
from .step import Inst, Step

Source: TypeAlias = Inst | str | Path | BIDSPath


class Pipeline(BaseEstimator):
    """A sequence of named processing steps.

    Follows scikit-learn conventions: :meth:`fit`, :meth:`transform` and
    :meth:`fit_transform`; nested parameters (``pipe.set_params(ica__threshold=0.9)``);
    :func:`sklearn.base.clone`; slicing (``pipe[:3]``).

    Unlike :class:`sklearn.pipeline.Pipeline`, every step is fitted *and*
    transformed in sequence (each step learns from the output of the previous
    one), steps may change the data type (e.g. Raw -> Epochs), and side
    outputs are collected on the fitted pipeline.

    Parameters
    ----------
    steps : list of (str, Step)
        The steps, in order. Names must be unique and must not contain ``"__"``.
    copy : bool
        If ``True`` (default), the input data are never modified. If
        ``False``, the input may be modified in place to save memory.

    Attributes
    ----------
    system_ : str
        Acquisition system detected when fitting.
    qc_ : dict
        Quality-control metrics of each step, keyed by step name.
    provenance_ : dict
        Software versions, detected system and, for each step, its class,
        parameters, run time and the warnings it raised.
    """

    def __init__(self, steps: list[tuple[str, Step]], *, copy: bool = True) -> None:
        self.steps = steps
        self.copy = copy

    # ------------------------------------------------------------------
    # Fitting and transforming

    def fit(self, inst: Source, y: Any = None) -> Self:
        """Fit every step in sequence.

        Each step except the last is fitted and then transforms the data
        passed to the next step; the last step is only fitted.

        Parameters
        ----------
        inst : Raw | Epochs | Evoked | str | Path | BIDSPath
            The data, or a recording to read with :func:`meeg_utils.io.read`.
        y : None
            Ignored; present for scikit-learn API compatibility.

        Returns
        -------
        self : Pipeline
            The fitted pipeline.
        """
        self._fit(inst, transform_last=False)
        return self

    def fit_transform(self, inst: Source, y: Any = None) -> Inst:
        """Fit every step in sequence and return the fully processed data.

        Parameters
        ----------
        inst : Raw | Epochs | Evoked | str | Path | BIDSPath
            The data, or a recording to read with :func:`meeg_utils.io.read`.
        y : None
            Ignored; present for scikit-learn API compatibility.

        Returns
        -------
        Raw | Epochs | Evoked
            The processed data.
        """
        return self._fit(inst, transform_last=True)

    def transform(self, inst: Source) -> Inst:
        """Apply every fitted step in sequence, without refitting.

        Parameters
        ----------
        inst : Raw | Epochs | Evoked | str | Path | BIDSPath
            The data, or a recording to read with :func:`meeg_utils.io.read`.
            Must come from the same acquisition system as the data the
            pipeline was fitted on.

        Returns
        -------
        Raw | Epochs | Evoked
            The processed data.
        """
        check_is_fitted(self, "system_")
        inst, owned = _load(inst)
        self._validate(type(inst))
        system = detect_system(inst)
        if system != self.system_:
            raise ValueError(
                f"The pipeline was fitted on {self.system_!r} data and cannot "
                f"transform {system!r} data."
            )
        data = inst.copy() if self.copy and not owned else inst
        for _, step in self.steps:
            data = step.transform(data, copy=False)
        return data

    def _fit(self, inst: Source, *, transform_last: bool) -> Inst:
        inst, owned = _load(inst)
        self._validate(type(inst))
        self.system_ = detect_system(inst)
        self.qc_: dict[str, dict[str, Any]] = {}
        self.provenance_: dict[str, Any] = {
            "versions": _versions(),
            "system": self.system_,
            "steps": [],
        }

        data = inst.copy() if self.copy and not owned else inst
        last = len(self.steps) - 1
        for i, (name, step) in enumerate(self.steps):
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if i < last or transform_last:
                    data = step.fit_transform(data, system=self.system_, copy=False)
                else:
                    step.fit(data, system=self.system_)
            for w in caught:  # record the warnings, then let them through
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

            self.qc_[name] = step.qc_
            self.provenance_["steps"].append(
                {
                    "name": name,
                    "class": _class_path(type(step)),
                    "params": _to_serializable(step.get_params(deep=False)),
                    "duration_s": round(time.perf_counter() - start, 3),
                    "warnings": [f"{w.category.__name__}: {w.message}" for w in caught],
                }
            )
        return data

    def _validate(self, input_type: type) -> None:
        """Check step names and that each step accepts its predecessor's output."""
        names = [name for name, _ in self.steps]
        if len(set(names)) != len(names):
            raise ValueError(f"Step names must be unique, got {names}.")
        current = input_type
        for name, step in self.steps:
            if "__" in name:
                raise ValueError(f"Step names must not contain '__', got {name!r}.")
            if not isinstance(step, Step):
                raise TypeError(
                    f"Step {name!r} must be a meeg_utils Step, got {type(step).__name__}."
                )
            if not issubclass(current, step.accepts):
                accepted = ", ".join(cls.__name__ for cls in step.accepts)
                raise TypeError(
                    f"Step {name!r} ({type(step).__name__}) accepts {accepted}, "
                    f"but receives {current.__name__}."
                )
            current = step.returns or current

    # ------------------------------------------------------------------
    # Parameters (nested, scikit-learn style)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters, including those of each step as ``<step>__<param>``.

        Parameters
        ----------
        deep : bool
            If ``True``, also return each step and its parameters.

        Returns
        -------
        dict
            Parameter names mapped to their values.
        """
        params: dict[str, Any] = super().get_params(deep=False)
        if deep:
            for name, step in self.steps:
                params[name] = step
                for key, value in step.get_params(deep=True).items():
                    params[f"{name}__{key}"] = value
        return params

    def set_params(self, **params: Any) -> Self:
        """Set parameters; ``<step>=new_step`` replaces a step.

        Parameters
        ----------
        **params
            ``steps``, ``copy``, ``<step>`` or ``<step>__<param>``.

        Returns
        -------
        self : Pipeline
            The pipeline.
        """
        if "steps" in params:
            self.steps = params.pop("steps")
        names = [name for name, _ in self.steps]
        for key in list(params):
            if key in names:
                self.replace(key, params.pop(key))
        super().set_params(**params)
        return self

    # ------------------------------------------------------------------
    # Editing and indexing

    @property
    def named_steps(self) -> dict[str, Step]:
        """The steps, keyed by name."""
        return dict(self.steps)

    def __len__(self) -> int:
        """Return the number of steps."""
        return len(self.steps)

    def __getitem__(self, key: int | str | slice) -> Step | Pipeline:
        """Return a step by position or name, or a sub-pipeline for a slice.

        A sub-pipeline shares its step objects (and their fitted state) with
        this pipeline.
        """
        if isinstance(key, slice):
            sub = Pipeline(self.steps[key], copy=self.copy)
            if hasattr(self, "system_"):
                sub.system_ = self.system_
            return sub
        if isinstance(key, str):
            return self.named_steps[key]
        return self.steps[key][1]

    def replace(self, name: str, step: Step) -> Self:
        """Replace the step called ``name``.

        Parameters
        ----------
        name : str
            Name of the step to replace.
        step : Step
            The new step.

        Returns
        -------
        self : Pipeline
            The pipeline.
        """
        index = self._index(name)
        self.steps = [*self.steps[:index], (name, step), *self.steps[index + 1 :]]
        return self

    def insert_after(self, after: str, name: str, step: Step) -> Self:
        """Insert a step after the step called ``after``.

        Parameters
        ----------
        after : str
            Name of the existing step.
        name : str
            Name of the new step.
        step : Step
            The new step.

        Returns
        -------
        self : Pipeline
            The pipeline.
        """
        index = self._index(after) + 1
        self.steps = [*self.steps[:index], (name, step), *self.steps[index:]]
        return self

    def insert_before(self, before: str, name: str, step: Step) -> Self:
        """Insert a step before the step called ``before``.

        Parameters
        ----------
        before : str
            Name of the existing step.
        name : str
            Name of the new step.
        step : Step
            The new step.

        Returns
        -------
        self : Pipeline
            The pipeline.
        """
        index = self._index(before)
        self.steps = [*self.steps[:index], (name, step), *self.steps[index:]]
        return self

    def remove(self, name: str) -> Self:
        """Remove the step called ``name``.

        Parameters
        ----------
        name : str
            Name of the step to remove.

        Returns
        -------
        self : Pipeline
            The pipeline.
        """
        index = self._index(name)
        self.steps = [*self.steps[:index], *self.steps[index + 1 :]]
        return self

    def _index(self, name: str) -> int:
        for i, (step_name, _) in enumerate(self.steps):
            if step_name == name:
                return i
        raise KeyError(f"No step named {name!r}; steps are {[n for n, _ in self.steps]}.")

    # ------------------------------------------------------------------
    # Serialization

    def to_dict(self) -> dict[str, Any]:
        """Return the pipeline configuration as plain, YAML-safe data.

        Returns
        -------
        dict
            The configuration; see :meth:`from_dict`.
        """
        return {
            "meeg_utils_version": _versions()["meeg_utils"],
            "copy": self.copy,
            "steps": [{"name": name, **_estimator_to_dict(step)} for name, step in self.steps],
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Pipeline:
        """Build a pipeline from :meth:`to_dict` output.

        Loading a configuration imports the modules that define its steps, so
        only load configurations you trust.

        Parameters
        ----------
        config : dict
            The configuration.

        Returns
        -------
        Pipeline
            An unfitted pipeline.
        """
        steps = [(entry["name"], _estimator_from_dict(entry)) for entry in config["steps"]]
        return cls(steps, copy=config.get("copy", True))

    def to_yaml(self, fname: str | Path) -> None:
        """Write the pipeline configuration to a YAML file.

        Parameters
        ----------
        fname : str | Path
            Output file.
        """
        import yaml

        Path(fname).write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    @classmethod
    def from_yaml(cls, fname: str | Path) -> Pipeline:
        """Read a pipeline configuration from a YAML file.

        Loading a configuration imports the modules that define its steps, so
        only load configurations you trust.

        Parameters
        ----------
        fname : str | Path
            Input file.

        Returns
        -------
        Pipeline
            An unfitted pipeline.
        """
        import yaml

        return cls.from_dict(yaml.safe_load(Path(fname).read_text(encoding="utf-8")))


# ----------------------------------------------------------------------
# Helpers


def _load(source: Source) -> tuple[Inst, bool]:
    """Return the data and whether it was read here (so need not be copied)."""
    if isinstance(source, str | Path | BIDSPath):
        return read(source), True
    return source, False


def _class_path(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _estimator_to_dict(estimator: BaseEstimator) -> dict[str, Any]:
    return {
        "class": _class_path(type(estimator)),
        "params": _to_serializable(estimator.get_params(deep=False)),
    }


def _estimator_from_dict(entry: dict[str, Any]) -> Step:
    module_name, _, qualname = entry["class"].partition(":")
    obj: Any = importlib.import_module(module_name)
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    if not (isinstance(obj, type) and issubclass(obj, Step)):
        raise TypeError(f"{entry['class']} is not a meeg_utils Step.")
    params = {key: _from_serializable(value) for key, value in entry.get("params", {}).items()}
    return obj(**params)


def _to_serializable(value: Any, path: str = "") -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, BaseEstimator):
        return _estimator_to_dict(value)
    if isinstance(value, dict):
        return {str(k): _to_serializable(v, f"{path}.{k}") for k, v in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, bytes):
        return [_to_serializable(v, f"{path}[{i}]") for i, v in enumerate(value)]
    if hasattr(value, "item"):  # numpy scalars
        return value.item()
    raise TypeError(
        f"Parameter {path.lstrip('.') or 'value'} of type {type(value).__name__} "
        "cannot be serialized; step parameters must be plain data or steps."
    )


def _from_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        if "class" in value and "params" in value and len(value) == 2:
            return _estimator_from_dict(value)
        return {k: _from_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_serializable(v) for v in value]
    return value


def _versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    versions = {"python": platform.python_version()}
    for package in ("meeg-utils", "mne", "numpy", "scipy", "scikit-learn"):
        try:
            versions[package.replace("-", "_")] = version(package)
        except PackageNotFoundError:
            versions[package.replace("-", "_")] = "n/a"
    return versions
