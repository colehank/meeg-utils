"""Combining the epochs of several runs.

Each run is preprocessed and epoched on its own (bad channels, line noise
and ICA differ between runs); :func:`combine` then checks that the runs are
compatible and concatenates them.
"""

from __future__ import annotations

from collections.abc import Sequence

import mne
import numpy as np
from loguru import logger
from mne.epochs import BaseEpochs

BAD_CHANNEL_POLICIES = ("union", "interpolate")


def combine(
    epochs_list: Sequence[BaseEpochs],
    *,
    bads: str = "union",
    max_head_shift_mm: float = 2.0,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.04),
) -> BaseEpochs:
    """Concatenate the epochs of several runs of one session.

    The runs must have the same channels (names and order), sampling rate
    and epoch times. Events are matched by name: if runs gave a condition
    different codes (MNE numbers the annotations of each run), the codes of
    the first run that has it are used.

    Parameters
    ----------
    epochs_list : sequence of Epochs
        The runs. They are not modified.
    bads : {"union", "interpolate"}
        ``"union"``: every channel bad in any run is marked bad in the
        result. ``"interpolate"``: those channels are interpolated in every
        run (needs sensor positions) and no channel is marked bad.
    max_head_shift_mm : float
        For MEG, the largest head displacement relative to the sensors
        allowed between runs. Larger differences raise: map the runs to a
        common head position first with :class:`~meeg_utils.steps.HeadAlign`
        and :func:`meeg_utils.io.average_dev_head_t`. Within the limit, the
        first run's head position is kept.
    origin : tuple of float
        Point (head frame, m) whose displacement is measured.

    Returns
    -------
    Epochs
        The concatenated epochs.

    Raises
    ------
    ValueError
        If the runs are not compatible.
    """
    from .io import get_datatypes

    if bads not in BAD_CHANNEL_POLICIES:
        raise ValueError(f"bads must be one of {BAD_CHANNEL_POLICIES}, got {bads!r}")
    runs = [e.copy() for e in epochs_list]
    if not runs:
        raise ValueError("No epochs to combine.")
    if not all(isinstance(e, BaseEpochs) for e in runs):
        raise TypeError("combine needs Epochs.")
    _check_compatible(runs)
    if "meg" in get_datatypes(runs[0]):
        _check_head_positions(runs, max_head_shift_mm, np.asarray(origin))
    first = runs[0]

    _unify_event_codes(runs)
    union = sorted({ch for run in runs for ch in run.info["bads"]}, key=first.ch_names.index)
    for run in runs:
        run.info["bads"] = list(union)
    if bads == "interpolate" and union:
        for run in runs:
            run.interpolate_bads(reset_bads=True, mode="accurate", verbose=False)
    logger.info(
        f"Combining {len(runs)} runs ({sum(len(r) for r in runs)} epochs); "
        f"bad channels ({bads}): {union}"
    )
    return mne.concatenate_epochs(runs, add_offset=True, on_mismatch="ignore", verbose=False)


def _check_compatible(runs: list[BaseEpochs]) -> None:
    first = runs[0]
    for i, run in enumerate(runs[1:], start=2):
        if run.ch_names != first.ch_names:
            only_first = sorted(set(first.ch_names) - set(run.ch_names))
            only_run = sorted(set(run.ch_names) - set(first.ch_names))
            raise ValueError(
                f"Run {i} has different channels than run 1 (only in run 1: {only_first}; "
                f"only in run {i}: {only_run}; or a different order)."
            )
        same_times = run.times.shape == first.times.shape and np.allclose(run.times, first.times)
        if run.info["sfreq"] != first.info["sfreq"] or not same_times:
            raise ValueError(f"Run {i} has a different sampling rate or epoch times than run 1.")


def _check_head_positions(runs: list[BaseEpochs], max_shift_mm: float, origin: np.ndarray) -> None:
    from .steps.head import _movement

    reference = runs[0].info["dev_head_t"]
    for i, run in enumerate(runs[1:], start=2):
        shift, rotation = _movement(reference, run.info["dev_head_t"], origin)
        if shift * 1000 > max_shift_mm:
            raise ValueError(
                f"The head position of run {i} differs from run 1 by {shift * 1000:.1f} mm "
                f"({rotation:.1f}°), more than max_head_shift_mm={max_shift_mm:g}. Map the "
                "runs to a common position first: S.HeadAlign(meu.io.average_dev_head_t(runs))."
            )


def _unify_event_codes(runs: list[BaseEpochs]) -> None:
    """Give every event name one code across runs (in place)."""
    mapping: dict[str, int] = {}
    next_code = 1 + max(int(c) for run in runs for c in run.event_id.values())
    for run in runs:
        for name, code in run.event_id.items():
            if name not in mapping:
                if code in mapping.values():  # taken by another name
                    mapping[name] = next_code
                    next_code += 1
                else:
                    mapping[name] = int(code)
    for run in runs:
        events = run.events.copy()
        for name, code in run.event_id.items():
            events[run.events[:, 2] == code, 2] = mapping[name]
        run.events = events
        run.event_id = {name: mapping[name] for name in run.event_id}
