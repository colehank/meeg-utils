"""Pytest configuration and shared fixtures for testing."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import mne
import pytest
from mne.io import BaseRaw
from mne_bids import BIDSPath


@pytest.fixture
def sample_eeg_raw() -> BaseRaw:
    """Create a sample EEG raw object for testing.

    Returns
    -------
    BaseRaw
        A sample EEG raw object with realistic properties.
    """
    import numpy as np

    # Create synthetic EEG data with standard 10-20 channel names
    sfreq = 500.0  # Sampling frequency
    n_samples = int(sfreq * 60)  # 60 seconds of data

    # Use standard 10-20 channel names
    ch_names = [
        "Fp1",
        "Fp2",
        "F7",
        "F3",
        "Fz",
        "F4",
        "F8",
        "T7",
        "C3",
        "Cz",
        "C4",
        "T8",
        "P7",
        "P3",
        "Pz",
        "P4",
        "P8",
        "O1",
        "O2",
    ]
    n_channels = len(ch_names)

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types="eeg",
    )

    # Create more realistic EEG-like data
    np.random.seed(42)

    # Generate base signal with multiple frequency components
    time = np.arange(n_samples) / sfreq

    # Create realistic EEG data with multiple components
    data = np.zeros((n_channels, n_samples))
    for i in range(n_channels):
        # Alpha rhythm (8-13 Hz) - dominant in occipital channels
        alpha_freq = 10.0 + np.random.randn() * 0.5
        alpha_amp = 5e-5 if "O" in ch_names[i] else 2e-5
        data[i] += alpha_amp * np.sin(2 * np.pi * alpha_freq * time)

        # Beta rhythm (13-30 Hz)
        beta_freq = 20.0 + np.random.randn() * 2
        beta_amp = 1e-5
        data[i] += beta_amp * np.sin(2 * np.pi * beta_freq * time)

        # Theta rhythm (4-8 Hz)
        theta_freq = 6.0 + np.random.randn() * 0.5
        theta_amp = 3e-5
        data[i] += theta_amp * np.sin(2 * np.pi * theta_freq * time)

        # Add some noise (but not too much)
        noise_amp = 0.5e-5
        data[i] += noise_amp * np.random.randn(n_samples)

        # Add slow drift
        drift_freq = 0.1
        drift_amp = 0.5e-5
        data[i] += drift_amp * np.sin(2 * np.pi * drift_freq * time)

    raw = mne.io.RawArray(data, info)

    # Add realistic montage
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="warn")

    return raw


@pytest.fixture
def sample_meg_raw() -> BaseRaw:
    """Create a sample MEG raw object for testing.

    Returns
    -------
    BaseRaw
        A sample MEG raw object with realistic properties.
    """
    import numpy as np

    # Create synthetic MEG data
    sfreq = 600.0  # Sampling frequency
    n_samples = int(sfreq * 60)  # 60 seconds

    # MEG typically has magnetometers and gradiometers
    n_mag = 102
    n_grad = 204

    ch_names_mag = [f"MEG{i:04d}" for i in range(1, n_mag + 1)]
    ch_names_grad = [f"MEG{i:04d}" for i in range(n_mag + 1, n_mag + n_grad + 1)]
    ch_names = ch_names_mag + ch_names_grad

    ch_types = ["mag"] * n_mag + ["grad"] * n_grad

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=ch_types,
    )

    # Create more realistic MEG data
    np.random.seed(42)
    n_channels = len(ch_names)
    time = np.arange(n_samples) / sfreq

    data = np.zeros((n_channels, n_samples))
    for i in range(n_channels):
        # MEG signals are typically smaller than EEG
        scale = 1e-12 if ch_types[i] == "mag" else 1e-13

        # Alpha-like activity
        alpha_freq = 10.0 + np.random.randn() * 0.3
        data[i] += scale * 5.0 * np.sin(2 * np.pi * alpha_freq * time)

        # Beta-like activity
        beta_freq = 20.0 + np.random.randn() * 1.0
        data[i] += scale * 2.0 * np.sin(2 * np.pi * beta_freq * time)

        # Add structured noise
        noise = scale * 0.5 * np.random.randn(n_samples)
        data[i] += noise

        # Add slow drift
        drift_freq = 0.05
        data[i] += scale * 1.0 * np.sin(2 * np.pi * drift_freq * time)

    raw = mne.io.RawArray(data, info)

    return raw


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory for testing.

    Parameters
    ----------
    tmp_path : Path
        Pytest's built-in temporary directory fixture.

    Returns
    -------
    Path
        Path to a temporary output directory.
    """
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def mock_bids_path(tmp_path: Path, sample_eeg_raw: BaseRaw) -> Generator[BIDSPath, None, None]:
    """Create a mock BIDS directory structure with sample data.

    Parameters
    ----------
    tmp_path : Path
        Pytest's built-in temporary directory fixture.
    sample_eeg_raw : BaseRaw
        Sample EEG raw object.

    Yields
    ------
    BIDSPath
        A BIDSPath object pointing to the mock data.
    """
    from mne_bids import write_raw_bids

    bids_root = tmp_path / "bids_dataset"

    bids_path = BIDSPath(
        subject="01",
        session="01",
        task="test",
        datatype="eeg",
        root=bids_root,
    )

    # Write sample data to BIDS format
    write_raw_bids(
        sample_eeg_raw,
        bids_path,
        allow_preload=True,
        format="BrainVision",
        overwrite=True,
        verbose=False,
    )

    yield bids_path

    # Cleanup is handled automatically by tmp_path


@pytest.fixture
def mock_bids_path_string(mock_bids_path: BIDSPath) -> str:
    """Get the string representation of a BIDS path.

    Parameters
    ----------
    mock_bids_path : BIDSPath
        A BIDSPath object.

    Returns
    -------
    str
        String representation of the BIDS path.
    """
    return str(mock_bids_path.fpath)


# ================================
# Mock Fixtures for Fast Testing
# ================================


@pytest.fixture
def mock_ica(mocker):
    """Mock ICA to avoid expensive computation.

    Trusts that MNE's ICA implementation works correctly.
    Only tests our wrapper logic.
    """
    from mne.preprocessing import ICA

    # Create a real ICA object but don't fit it
    mock_ica_obj = mocker.Mock(spec=ICA)
    mock_ica_obj.n_components_ = 5
    mock_ica_obj.exclude = []
    mock_ica_obj.labels_ = {"brain": [0, 2, 4], "eye blink": [1], "heart beat": [3]}

    # Mock the apply method to return input unchanged
    mock_ica_obj.apply.side_effect = lambda raw: raw

    # Mock the fit method
    mock_ica_obj.fit.return_value = mock_ica_obj

    return mock_ica_obj


@pytest.fixture
def mock_prep(mocker):
    """Mock PREP pipeline to avoid expensive computation.

    Trusts that PyPREP implementation works correctly.
    Only tests our wrapper logic.
    """
    # Mock the find_bad_channels function
    mock_finder = mocker.Mock()
    # Return empty lists for bad channels
    mock_finder.get_bads.return_value = []
    mock_finder.bad_by_deviation = []
    mock_finder.bad_by_correlation = []
    mock_finder.bad_by_ransac = []
    mock_finder.bad_by_SNR = []

    # Mock methods to do nothing
    mock_finder.find_all_bads.return_value = None
    mock_finder.find_bad_by_deviation.return_value = None
    mock_finder.find_bad_by_correlation.return_value = None
    mock_finder.find_bad_by_ransac.return_value = None

    return mock_finder


@pytest.fixture
def mock_zapline(mocker):
    """Mock Zapline to avoid expensive computation.

    Trusts that meegkit's Zapline implementation works correctly.
    Only tests our wrapper logic.
    """

    def mock_zapline_iter(data, *args, **kwargs):
        """Return data unchanged."""
        return data, None

    mocker.patch(
        "meeg_utils.preprocessing.line_noise.dss.dss_line_iter",
        side_effect=mock_zapline_iter,
    )

    return mock_zapline_iter


@pytest.fixture
def mock_iclabel(mocker):
    """Mock ICLabel to avoid expensive computation.

    Trusts that mne-icalabel implementation works correctly.
    Only tests our wrapper logic.
    """

    def mock_label_components(raw, ica, method="iclabel"):
        """Return mock labels."""
        n_components = ica.n_components_
        return {
            "labels": ["brain"] * n_components,
            "y_pred_proba": [0.9] * n_components,
        }

    mocker.patch(
        "meeg_utils.preprocessing.ica.label_components",
        side_effect=mock_label_components,
    )

    return mock_label_components
