"""Contract checks for processing steps and quality checks.

Every :class:`~meeg_utils.core.Step` shipped with meeg-utils must pass
:func:`check_step`, and every :class:`~meeg_utils.qc.Check` must pass
:func:`check_check`; custom ones can use them in their own tests.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
from mne import Evoked
from mne.epochs import BaseEpochs
from mne.io import BaseRaw
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from .core import Pipeline, Step
from .core.step import Inst

if TYPE_CHECKING:
    from .qc import Check


def check_step(step: Step, inst: Inst) -> None:
    """Check that a step honours the meeg-utils step contract.

    The checks are:

    1. Parameters survive ``get_params`` / ``set_params``, :func:`~sklearn.base.clone`
       and a round trip through the pipeline configuration format.
    2. ``transform`` before ``fit`` raises :class:`~sklearn.exceptions.NotFittedError`.
    3. ``fit`` returns the step itself and sets ``system_`` and ``qc_``.
    4. With ``copy=True``, neither ``fit`` nor ``transform`` modify the input.
    5. The output has the declared type.
    6. Unless the step declares ``changes_channels``, channel names and order
       are unchanged.
    7. Unless the step declares ``changes_times``, the sampling rate, number
       of samples, ``first_samp``, ``meas_date`` and annotations are unchanged
       (steps declaring ``adds_annotations`` may only add ``BAD_`` ones).
    8. Every kind in ``plot_kinds`` that is available for this fit draws a
       figure (or a list of figures), and ``plot()`` without data draws
       exactly the available kinds that need none.

    Parameters
    ----------
    step : Step
        An unfitted step.
    inst : Raw | Epochs | Evoked
        Preloaded data the step can process.

    Raises
    ------
    AssertionError
        If a check fails.
    """
    name = type(step).__name__

    # 1. Parameters
    params = step.get_params(deep=False)
    cloned = clone(step)
    _assert_params_equal(cloned.get_params(deep=False), params, f"{name}: clone")
    step.set_params(**params)
    _assert_params_equal(step.get_params(deep=False), params, f"{name}: set_params")
    restored = Pipeline.from_dict(Pipeline([("step", step)]).to_dict())["step"]
    assert type(restored) is type(step), f"{name}: config round trip changed the class"
    _assert_params_equal(
        restored.get_params(deep=False), params, f"{name}: config round trip", loose=True
    )

    # 2. Not fitted
    try:
        clone(step).transform(inst)
    except NotFittedError:
        pass
    else:
        raise AssertionError(f"{name}: transform before fit must raise NotFittedError")

    # 3-4. Fitting
    before = _snapshot(inst)
    fitted = step.fit(inst)
    assert fitted is step, f"{name}: fit must return self"
    assert isinstance(step.system_, str), f"{name}: fit must set system_"
    assert isinstance(step.qc_, dict), f"{name}: fit must set qc_ to a dict"
    _assert_unchanged(inst, before, f"{name}: fit modified its input")

    # 4-7. Transforming
    out = step.transform(inst)
    _assert_unchanged(inst, before, f"{name}: transform(copy=True) modified its input")

    expected_type = step.returns or type(inst)
    if not issubclass(type(out), expected_type):
        raise AssertionError(
            f"{name}: expected {expected_type.__name__} output, got {type(out).__name__}"
        )
    assert out is not inst, f"{name}: transform(copy=True) returned its input"

    if not step.changes_channels and step.returns is None:
        assert out.ch_names == inst.ch_names, (
            f"{name}: channels changed without declaring changes_channels\n"
            f"  before: {inst.ch_names}\n  after:  {out.ch_names}"
        )

    if not step.changes_times and step.returns is None:
        _assert_same_times(inst, out, name, step)

    # 8. Figures
    _check_plots(step, inst, name)


def check_check(check: Check, raw: BaseRaw) -> None:
    """Check that a quality check honours the meeg-utils check contract.

    The checks are:

    1. Parameters survive ``get_params`` / ``set_params`` and
       :func:`~sklearn.base.clone`.
    2. ``plot`` before ``compute`` raises :class:`~sklearn.exceptions.NotFittedError`.
    3. ``compute`` returns the check itself and does not modify the data.
    4. ``metrics_`` and the findings are JSON-serializable, every finding
       belongs to the check and has a valid level.
    5. Every kind in ``plot_kinds`` draws a figure.

    Parameters
    ----------
    check : Check
        A check that has not been computed.
    raw : Raw
        Preloaded data the check applies to.

    Raises
    ------
    AssertionError
        If a check fails.
    """
    from .qc import LEVELS

    name = type(check).__name__
    params = check.get_params(deep=False)
    _assert_params_equal(clone(check).get_params(deep=False), params, f"{name}: clone")
    check.set_params(**params)
    _assert_params_equal(check.get_params(deep=False), params, f"{name}: set_params")
    assert check.name, f"{name}: name must be set"

    try:
        clone(check).plot()
    except NotFittedError:
        pass
    else:
        raise AssertionError(f"{name}: plot before compute must raise NotFittedError")

    before = _snapshot(raw)
    assert check.compute(raw) is check, f"{name}: compute must return self"
    _assert_unchanged(raw, before, f"{name}: compute modified its input")

    try:
        json.dumps(check.metrics_, allow_nan=True)
        json.dumps([f.to_dict() for f in check.findings_], allow_nan=True, default=str)
    except TypeError as exc:
        raise AssertionError(
            f"{name}: metrics_ / findings_ are not JSON-serializable: {exc}"
        ) from exc
    for finding in check.findings_:
        assert finding.check == check.name, f"{name}: finding {finding} names another check"
        assert finding.level in LEVELS, f"{name}: invalid level {finding.level!r}"
    assert check.level in LEVELS

    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    try:
        drawn = check.plot(inst=raw)
        assert set(drawn) == set(check.plot_kinds), (
            f"{name}: plot drew {sorted(drawn)}, expected {sorted(check.plot_kinds)}"
        )
        for kind, figure in drawn.items():
            figs = figure if isinstance(figure, list) else [figure]
            assert figs and all(isinstance(f, Figure) for f in figs), (
                f"{name}: plot {kind!r} did not return matplotlib figures"
            )
    finally:
        plt.close("all")


def _check_plots(step: Step, inst: Inst, name: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    try:
        available = {k for k in step.plot_kinds if step._plot_unavailable(k) is None}
        without_data = step.plot()
        expected = {k for k in available if not step.plot_kinds[k]}
        assert set(without_data) == expected, (
            f"{name}: plot() drew {sorted(without_data)}, expected {sorted(expected)}"
        )
        with_data = step.plot(inst=inst)
        assert set(with_data) == available, (
            f"{name}: plot(inst=...) drew {sorted(with_data)}, expected {sorted(available)}"
        )
        for kind, drawn in with_data.items():
            figs = drawn if isinstance(drawn, list) else [drawn]
            assert figs and all(isinstance(f, Figure) for f in figs), (
                f"{name}: plot {kind!r} did not return matplotlib figures"
            )
    finally:
        plt.close("all")


# ----------------------------------------------------------------------


def _assert_params_equal(actual: dict, expected: dict, msg: str, *, loose: bool = False) -> None:
    assert actual.keys() == expected.keys(), f"{msg}: parameter names differ"
    for key, expected_value in expected.items():
        assert _values_equal(actual[key], expected_value, loose=loose), (
            f"{msg}: parameter {key!r} differs ({actual[key]!r} != {expected_value!r})"
        )


def _values_equal(other: Any, value: Any, *, loose: bool) -> bool:
    """Parameter equality; steps compare by class and parameters, containers recursively."""
    if loose and isinstance(value, tuple):  # tuples come back as lists
        value = list(value)
    if isinstance(value, Step):
        return type(other) is type(value) and all(
            _values_equal(other.get_params(deep=False)[k], v, loose=loose)
            for k, v in value.get_params(deep=False).items()
        )
    if isinstance(value, np.ndarray) or isinstance(other, np.ndarray):
        return bool(np.array_equal(np.asarray(other), np.asarray(value)))
    if isinstance(value, dict):
        return (
            isinstance(other, dict)
            and other.keys() == value.keys()
            and all(_values_equal(other[k], v, loose=loose) for k, v in value.items())
        )
    if isinstance(value, list | tuple) and any(
        isinstance(v, Step | list | tuple | dict) for v in value
    ):
        return (
            isinstance(other, list | tuple)
            and len(other) == len(value)
            and all(_values_equal(o, v, loose=loose) for o, v in zip(other, value, strict=True))
        )
    return bool(other == value or other is value)


def _snapshot(inst: Inst) -> dict:
    snap = {
        "data": inst.get_data().copy(),
        "ch_names": list(inst.ch_names),
        "sfreq": inst.info["sfreq"],
        "bads": list(inst.info["bads"]),
    }
    if isinstance(inst, BaseRaw):
        snap["first_samp"] = inst.first_samp
        snap["annotations"] = _annotation_tuple(inst)
    return snap


def _assert_unchanged(inst: Inst, snap: dict, msg: str) -> None:
    assert inst.ch_names == snap["ch_names"], f"{msg} (channels)"
    assert inst.info["sfreq"] == snap["sfreq"], f"{msg} (sfreq)"
    assert inst.info["bads"] == snap["bads"], f"{msg} (bads)"
    assert np.array_equal(inst.get_data(), snap["data"]), f"{msg} (data)"
    if isinstance(inst, BaseRaw):
        assert inst.first_samp == snap["first_samp"], f"{msg} (first_samp)"
        assert _annotation_tuple(inst) == snap["annotations"], f"{msg} (annotations)"


def _assert_same_times(inst: Inst, out: Inst, name: str, step: Step) -> None:
    msg = f"{name}: time axis changed without declaring changes_times"
    assert out.info["sfreq"] == inst.info["sfreq"], f"{msg} (sfreq)"
    assert out.info["meas_date"] == inst.info["meas_date"], f"{msg} (meas_date)"
    if isinstance(inst, BaseRaw):
        assert out.first_samp == inst.first_samp, (
            f"{msg} (first_samp {inst.first_samp} -> {out.first_samp})"
        )
        assert out.n_times == inst.n_times, f"{msg} (n_times)"
        if step.adds_annotations:
            _assert_annotations_added(inst, out, name)
        else:
            assert _annotation_tuple(out) == _annotation_tuple(inst), f"{msg} (annotations)"
    elif isinstance(inst, BaseEpochs | Evoked):
        assert np.array_equal(out.times, inst.times), f"{msg} (times)"
    if isinstance(inst, BaseEpochs):
        if step.drops_epochs:
            kept = {tuple(e) for e in inst.events}
            assert all(tuple(e) in kept for e in out.events), (
                f"{name}: epochs were changed, not dropped"
            )
        else:
            assert np.array_equal(out.events, inst.events), f"{msg} (events)"


def _assert_annotations_added(inst: BaseRaw, out: BaseRaw, name: str) -> None:
    """Existing annotations are kept unchanged; new ones are BAD_ annotations."""
    assert out.annotations.orig_time == inst.annotations.orig_time, f"{name}: orig_time changed"
    before = list(zip(*_annotation_tuple(inst)[1:], strict=True))
    after = list(zip(*_annotation_tuple(out)[1:], strict=True))
    remaining = list(after)
    for annot in before:
        assert annot in remaining, f"{name}: annotation {annot} was changed or removed"
        remaining.remove(annot)
    for annot in remaining:
        assert annot[2].startswith("BAD_"), f"{name}: added a non-BAD annotation {annot}"


def _annotation_tuple(raw: BaseRaw) -> tuple:
    annot = raw.annotations
    return (
        annot.orig_time,
        tuple(np.round(annot.onset, 9)),
        tuple(np.round(annot.duration, 9)),
        tuple(annot.description),
    )
