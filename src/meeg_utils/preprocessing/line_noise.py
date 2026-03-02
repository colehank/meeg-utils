"""Line noise removal using zapline methods."""

from __future__ import annotations

import contextlib
import io

import mne
import numpy as np
from loguru import logger
from meegkit import dss
from mne.io import BaseRaw


def remove_line_noise_eeg(
    raw: BaseRaw,
    fline: float = 50.0,
) -> BaseRaw:
    """Remove line noise from EEG data using zapline iterative method.

    Parameters
    ----------
    raw : BaseRaw
        Raw EEG data.
    fline : float, optional
        Line noise frequency in Hz. Default is 50.0.

    Returns
    -------
    BaseRaw
        Raw data with line noise removed.
    """
    logger.trace(f"Removing {fline} Hz line noise from EEG using zapline_iter...")

    # Pick EEG channels
    raw_eeg = raw.copy().pick("eeg")

    # Check if we have enough data
    if len(raw_eeg.ch_names) < 2:
        logger.warning("Not enough EEG channels for line noise removal. Skipping.")
        return raw_eeg

    # Prepare data for zapline
    data = raw_eeg.get_data().T  # (n_samples, n_channels)
    data = np.expand_dims(data, axis=2)  # (n_samples, n_channels, n_trials=1)

    sfreq = raw_eeg.info["sfreq"]

    # Apply zapline iterative
    try:
        with contextlib.redirect_stdout(io.StringIO()):  # Suppress meegkit output
            cleaned_data, _ = dss.dss_line_iter(data, fline, sfreq=sfreq, nfft=400)

        # Reconstruct raw
        cleaned_data = cleaned_data.T.squeeze()  # Back to (n_channels, n_samples)
        cleaned_raw = mne.io.RawArray(cleaned_data, raw_eeg.info)
        cleaned_raw.set_annotations(raw_eeg.annotations)

        logger.trace("Line noise removal completed for EEG.")
        return cleaned_raw

    except Exception as e:
        logger.error(f"Zapline failed: {e}. Returning original data.")
        return raw_eeg


def remove_line_noise_meg(
    raw: BaseRaw,
    fline: float = 50.0,
    removing_ratio: float = 0.22,
) -> BaseRaw:
    """Remove line noise from MEG data using zapline method.

    Parameters
    ----------
    raw : BaseRaw
        Raw MEG data.
    fline : float, optional
        Line noise frequency in Hz. Default is 50.0.
    removing_ratio : float, optional
        Ratio of components to remove. Default is 0.22.

    Returns
    -------
    BaseRaw
        Raw data with line noise removed.
    """
    logger.trace(f"Removing {fline} Hz line noise from MEG using zapline...")

    # Pick MEG channels
    try:
        raw_meg = raw.copy().pick(["mag", "grad", "planar1", "planar2"], exclude=[])
    except Exception as e:
        logger.warning(f"Failed to pick MEG channels: {e}. Returning original data.")
        return raw.copy()

    # Check if we have enough channels
    if len(raw_meg.ch_names) < 2:
        logger.warning("Not enough MEG channels for line noise removal. Skipping.")
        return raw_meg

    # Prepare data
    data = raw_meg.get_data().T
    data = np.expand_dims(data, axis=2)

    sfreq = raw_meg.info["sfreq"]
    nremove = max(1, int(len(raw_meg.ch_names) * removing_ratio))

    # Apply zapline
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cleaned_data, _ = dss.dss_line(
                data,
                fline=fline,
                sfreq=sfreq,
                nremove=nremove,
                blocksize=1000,
                show=False,
            )

        # Reconstruct raw
        cleaned_data = cleaned_data.T.squeeze()
        cleaned_raw = mne.io.RawArray(cleaned_data, raw_meg.info)
        cleaned_raw.set_annotations(raw_meg.annotations)

        logger.trace("Line noise removal completed for MEG.")
        return cleaned_raw

    except Exception as e:
        logger.error(f"Zapline failed: {e}. Returning original data.")
        return raw_meg
