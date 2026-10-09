"""Acquisition-system detection for MEG/EEG recordings."""

from __future__ import annotations

from collections import Counter

from mne import Info
from mne.io.constants import FIFF

# MEG system name -> coil types (lower 16 bits of ``ch["coil_type"]``) that
# identify it. Reference-sensor coils are listed so that systems are still
# recognised when only reference channels are present.
_SYSTEM_COILS: dict[str, frozenset[int]] = {
    "neuromag": frozenset(
        {
            FIFF.FIFFV_COIL_VV_PLANAR_W,
            FIFF.FIFFV_COIL_VV_PLANAR_T1,
            FIFF.FIFFV_COIL_VV_PLANAR_T2,
            FIFF.FIFFV_COIL_VV_PLANAR_T3,
            FIFF.FIFFV_COIL_VV_PLANAR_T4,
            FIFF.FIFFV_COIL_VV_MAG_W,
            FIFF.FIFFV_COIL_VV_MAG_T1,
            FIFF.FIFFV_COIL_VV_MAG_T2,
            FIFF.FIFFV_COIL_VV_MAG_T3,
            FIFF.FIFFV_COIL_VV_MAG_T4,
        }
    ),
    "ctf": frozenset(
        {
            FIFF.FIFFV_COIL_CTF_GRAD,
            FIFF.FIFFV_COIL_CTF_REF_MAG,
            FIFF.FIFFV_COIL_CTF_REF_GRAD,
            FIFF.FIFFV_COIL_CTF_OFFDIAG_REF_GRAD,
        }
    ),
    "kit": frozenset(
        {
            FIFF.FIFFV_COIL_KIT_GRAD,
            FIFF.FIFFV_COIL_KIT_REF_MAG,
            FIFF.FIFFV_COIL_BABY_GRAD,
            FIFF.FIFFV_COIL_BABY_MAG,
            FIFF.FIFFV_COIL_BABY_REF_MAG,
            FIFF.FIFFV_COIL_BABY_REF_MAG2,
        }
    ),
    "bti": frozenset(
        {
            FIFF.FIFFV_COIL_MAGNES_MAG,
            FIFF.FIFFV_COIL_MAGNES_GRAD,
            FIFF.FIFFV_COIL_MAGNES_REF_MAG,
            FIFF.FIFFV_COIL_MAGNES_REF_GRAD,
            FIFF.FIFFV_COIL_MAGNES_OFFDIAG_REF_GRAD,
        }
    ),
    "artemis123": frozenset(
        {
            FIFF.FIFFV_COIL_ARTEMIS123_GRAD,
            FIFF.FIFFV_COIL_ARTEMIS123_REF_MAG,
            FIFF.FIFFV_COIL_ARTEMIS123_REF_GRAD,
        }
    ),
    "opm": frozenset(
        {
            FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG,
            FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG2,
            FIFF.FIFFV_COIL_FIELDLINE_OPM_MAG_GEN1,
            FIFF.FIFFV_COIL_KERNEL_OPM_MAG_GEN1,
        }
    ),
}

MEG_SYSTEMS = (*_SYSTEM_COILS, "meg-other")
SYSTEMS = (*MEG_SYSTEMS, "eeg")

_MEG_KINDS = (FIFF.FIFFV_MEG_CH, FIFF.FIFFV_REF_MEG_CH)


def detect_system(inst) -> str:
    """Detect the acquisition system of a recording.

    Parameters
    ----------
    inst : Raw | Epochs | Evoked | Info
        The data, or its measurement info.

    Returns
    -------
    str
        One of :data:`SYSTEMS`. Recordings that contain MEG sensors are
        identified by their MEG system (``"neuromag"``, ``"ctf"``, ``"kit"``,
        ``"bti"``, ``"artemis123"``, ``"opm"`` or ``"meg-other"``), even if
        they also contain EEG channels; use :func:`get_datatypes` to find out
        which modalities are present. EEG-only recordings return ``"eeg"``.

    Raises
    ------
    ValueError
        If the recording contains neither MEG nor EEG channels.
    """
    info = inst if isinstance(inst, Info) else inst.info

    coils = Counter(ch["coil_type"] & 0xFFFF for ch in info["chs"] if ch["kind"] in _MEG_KINDS)
    if coils:
        votes = {
            system: sum(n for coil, n in coils.items() if coil in system_coils)
            for system, system_coils in _SYSTEM_COILS.items()
        }
        system, n_votes = max(votes.items(), key=lambda item: item[1])
        return system if n_votes > 0 else "meg-other"

    if "eeg" in get_datatypes(info):
        return "eeg"

    raise ValueError(
        "Cannot detect the acquisition system: the recording contains neither "
        f"MEG nor EEG channels (channel types: {sorted(set(info.get_channel_types()))})."
    )


def get_datatypes(inst) -> set[str]:
    """Return the modalities (``"meg"`` and/or ``"eeg"``) present in a recording.

    Parameters
    ----------
    inst : Raw | Epochs | Evoked | Info
        The data, or its measurement info.

    Returns
    -------
    set of str
        Subset of ``{"meg", "eeg"}``. Reference MEG channels alone do not
        count as MEG data.
    """
    info = inst if isinstance(inst, Info) else inst.info
    kinds = {ch["kind"] for ch in info["chs"]}
    datatypes = set()
    if FIFF.FIFFV_MEG_CH in kinds:
        datatypes.add("meg")
    if FIFF.FIFFV_EEG_CH in kinds:
        datatypes.add("eeg")
    return datatypes
