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


#: Head-frame origin (m) used when "auto" cannot be fitted: MNE's default
#: sphere origin (``make_sphere_model(r0=...)``, field mapping).
DEFAULT_ORIGIN = (0.0, 0.0, 0.04)


def head_origin(origin: str | tuple[float, float, float], info: Info) -> tuple[float, float, float]:
    """Resolve ``origin="auto"`` by fitting a sphere to the head digitization.

    Recordings without head-shape points (e.g. only the three fiducials, as
    in many CTF datasets) cannot be fitted; :data:`DEFAULT_ORIGIN` is used
    instead, with a warning.

    Parameters
    ----------
    origin : "auto" | tuple of float
        The origin (head frame, m) or ``"auto"``.
    info : Info
        Measurement info with the digitization.

    Returns
    -------
    tuple of float
        The origin in metres.
    """
    import warnings

    if not (isinstance(origin, str) and origin == "auto"):
        return (float(origin[0]), float(origin[1]), float(origin[2]))
    try:
        _, center, _ = mne.bem.fit_sphere_to_headshape(info, units="m", verbose=False)
    except (ValueError, RuntimeError) as exc:
        warnings.warn(
            f"Cannot fit the head origin to the digitization ({exc}); using {DEFAULT_ORIGIN} m "
            "(head frame), MNE's default sphere origin. Pass origin=(x, y, z) to choose another.",
            stacklevel=3,
        )
        return DEFAULT_ORIGIN
    return (float(center[0]), float(center[1]), float(center[2]))
