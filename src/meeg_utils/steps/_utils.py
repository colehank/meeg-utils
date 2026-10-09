"""Helpers shared by the processing steps."""

from __future__ import annotations

import mne
import numpy as np
from mne import Info

#: Data channel types processed separately because their units differ.
DATA_CH_TYPES = ("mag", "grad", "eeg")


def picks_by_type(info: Info, *, exclude_bads: bool = True) -> dict[str, np.ndarray]:
    """Return the indices of each data channel type present in ``info``.

    Reference MEG channels are never included.

    Parameters
    ----------
    info : Info
        Measurement info.
    exclude_bads : bool
        Whether to leave out channels listed in ``info["bads"]``.

    Returns
    -------
    dict
        Channel type (``"mag"``, ``"grad"`` or ``"eeg"``) mapped to channel indices,
        for the types that have at least one channel.
    """
    exclude = "bads" if exclude_bads else []
    picks = {}
    for ch_type in DATA_CH_TYPES:
        if ch_type == "eeg":
            idx = mne.pick_types(info, meg=False, eeg=True, exclude=exclude)
        else:
            idx = mne.pick_types(info, meg=ch_type, ref_meg=False, exclude=exclude)
        if len(idx):
            picks[ch_type] = idx
    return picks
