"""Tests for acquisition-system detection."""

from __future__ import annotations

import mne
import pytest
from mne.io.constants import FIFF

from meeg_utils.io import detect_system, get_datatypes


def _info(ch_types: list[str], coil_type: int | None = None, kind: int | None = None) -> mne.Info:
    info = mne.create_info([f"CH{i:03d}" for i in range(len(ch_types))], 250.0, ch_types)
    if coil_type is not None:
        with info._unlock():
            for ch in info["chs"]:
                if ch["kind"] in (FIFF.FIFFV_MEG_CH, FIFF.FIFFV_REF_MEG_CH):
                    ch["coil_type"] = coil_type
                    if kind is not None:
                        ch["kind"] = kind
    return info


@pytest.mark.parametrize(
    ("coil_type", "expected"),
    [
        (FIFF.FIFFV_COIL_CTF_GRAD, "ctf"),
        (FIFF.FIFFV_COIL_KIT_GRAD, "kit"),
        (FIFF.FIFFV_COIL_MAGNES_MAG, "bti"),
        (FIFF.FIFFV_COIL_ARTEMIS123_GRAD, "artemis123"),
        (FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG2, "opm"),
        (FIFF.FIFFV_COIL_FIELDLINE_OPM_MAG_GEN1, "opm"),
        (FIFF.FIFFV_COIL_POINT_MAGNETOMETER, "meg-other"),
    ],
)
def test_meg_systems(coil_type: int, expected: str) -> None:
    """MEG systems are identified from their sensor coil types."""
    assert detect_system(_info(["mag"] * 4, coil_type)) == expected


def test_neuromag_from_create_info() -> None:
    """mag/grad channels from create_info use Neuromag coils."""
    assert detect_system(_info(["mag", "grad", "grad"])) == "neuromag"


def test_ctf_with_compensation_bits() -> None:
    """CTF coil types carry the compensation grade in their upper bits."""
    grade3 = FIFF.FIFFV_COIL_CTF_GRAD | (3 << 16)
    assert detect_system(_info(["mag"] * 4, grade3)) == "ctf"


def test_meg_plus_eeg_is_identified_by_meg_system() -> None:
    """MEG+EEG recordings are MEG recordings that also contain EEG."""
    info = _info(["mag", "grad", "eeg", "eeg", "eog"])
    assert detect_system(info) == "neuromag"
    assert get_datatypes(info) == {"meg", "eeg"}


def test_eeg_only() -> None:
    """EEG-only recordings are 'eeg'."""
    info = _info(["eeg", "eeg", "eog", "stim"])
    assert detect_system(info) == "eeg"
    assert get_datatypes(info) == {"eeg"}


def test_reference_channels_alone_are_not_meg_data() -> None:
    """Reference sensors identify the system but are not MEG data."""
    info = _info(["ref_meg", "ref_meg", "eeg"], FIFF.FIFFV_COIL_CTF_REF_MAG)
    assert detect_system(info) == "ctf"
    assert get_datatypes(info) == {"eeg"}


def test_accepts_raw(small_raw) -> None:
    """Raw/Epochs/Evoked are accepted as well as Info."""
    assert detect_system(small_raw) == "eeg"


def test_no_meg_or_eeg_raises() -> None:
    """Recordings without MEG or EEG channels are rejected."""
    with pytest.raises(ValueError, match="neither MEG nor EEG"):
        detect_system(_info(["misc", "stim"]))
