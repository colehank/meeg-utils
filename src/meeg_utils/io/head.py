"""Head positions across MEG recordings."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from mne import Transform
from mne.io import BaseRaw
from mne_bids import BIDSPath

from .read import read


def average_dev_head_t(sources: Sequence[BaseRaw | str | Path | BIDSPath]) -> Transform:
    """Average device-to-head transform of several MEG recordings.

    Each recording's head position (``info["dev_head_t"]``) is weighted by
    its duration outside ``BAD`` annotations; rotations are averaged as
    quaternions (:func:`mne.preprocessing.compute_average_dev_head_t`). Use
    the result as the ``destination`` of
    :class:`~meeg_utils.steps.HeadAlign` so that every run is mapped to the
    same position, the one that needs the least movement on average.

    Parameters
    ----------
    sources : sequence of Raw | path | BIDSPath
        The recordings (typically the runs of one session). Paths are read
        without loading the data.

    Returns
    -------
    Transform
        The average device-to-head transform.

    Raises
    ------
    ValueError
        If there are no recordings or one lacks a head transform.
    """
    from mne.preprocessing import compute_average_dev_head_t
    from mne.transforms import rot_to_quat

    if not sources:
        raise ValueError("average_dev_head_t needs at least one recording.")
    raws = [s if isinstance(s, BaseRaw) else read(s, preload=False) for s in sources]
    positions = []
    for source, raw in zip(sources, raws, strict=True):
        trans = raw.info["dev_head_t"]
        if trans is None:
            raise ValueError(f"{source} has no device-to-head transform.")
        quat = rot_to_quat(trans["trans"][:3, :3])
        # one cHPI-like sample per recording: t, quaternion, translation, gof, err, v
        positions.append(
            np.array([[raw.first_samp / raw.info["sfreq"], *quat, *trans["trans"][:3, 3], 1, 0, 0]])
        )
    return compute_average_dev_head_t(raws, positions, verbose=False)
