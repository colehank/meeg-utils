"""Bad channel detection for EEG and MEG data."""

from __future__ import annotations

from loguru import logger
from mne.io import BaseRaw
from mne.preprocessing import find_bad_channels_maxwell
from pyprep.find_noisy_channels import NoisyChannels


def detect_bad_channels_eeg(
    raw: BaseRaw,
    random_state: int = 42,
) -> list[str]:
    """Detect bad EEG channels using PREP pipeline.

    Parameters
    ----------
    raw : BaseRaw
        Raw EEG data.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    list[str]
        List of bad channel names.
    """
    logger.trace("Detecting bad EEG channels using PREP...")

    # Pick only EEG channels
    raw_eeg = raw.copy().pick("eeg")

    # Use PREP NoisyChannels finder
    try:
        finder = NoisyChannels(raw_eeg, random_state=random_state)

        # Run detection methods with error handling
        try:
            finder.find_bad_by_correlation()
        except Exception as e:
            logger.warning(f"Correlation-based detection failed: {e}")

        try:
            finder.find_bad_by_deviation()
        except Exception as e:
            logger.warning(f"Deviation-based detection failed: {e}")

        try:
            finder.find_bad_by_ransac()
        except Exception as e:
            logger.warning(f"RANSAC-based detection failed: {e}")

        bad_channels = list(finder.get_bads())

    except Exception as e:
        logger.error(f"PREP pipeline failed: {e}. Returning empty bad channels list.")
        bad_channels = []

    logger.trace(f"Found {len(bad_channels)} bad EEG channels: {bad_channels}")

    return bad_channels


def detect_bad_channels_meg(
    raw: BaseRaw,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.04),
) -> list[str]:
    """Detect bad MEG channels using Maxwell filtering.

    Parameters
    ----------
    raw : BaseRaw
        Raw MEG data.
    origin : tuple, optional
        Origin for Maxwell filtering. Default is (0.0, 0.0, 0.04).

    Returns
    -------
    list[str]
        List of bad channel names.
    """
    logger.trace("Detecting bad MEG channels using Maxwell filtering...")

    # Copy raw for detection
    raw_copy = raw.copy()

    # Check if required info is available
    if raw_copy.info.get("dev_head_t") is None:
        logger.warning(
            "Cannot run Maxwell filtering: dev_head_t is None. Returning empty bad channels list."
        )
        return []

    # CTF data may need compensation adjustment
    if hasattr(raw_copy, "compensation_grade") and raw_copy.compensation_grade != 0:
        logger.trace(
            f"CTF data has compensation grade {raw_copy.compensation_grade}, "
            "applying 0-compensation for bad channel detection."
        )
        try:
            raw_copy.apply_gradient_compensation(0)
        except Exception as e:
            logger.warning(f"Failed to apply gradient compensation: {e}")

    # Pick MEG channels
    try:
        raw_copy.pick(["mag", "grad", "planar1", "planar2"], exclude=[])
    except Exception as e:
        logger.warning(f"Failed to pick MEG channels: {e}")
        return []

    # Find bad channels
    try:
        auto_noisy, auto_flat, _ = find_bad_channels_maxwell(
            raw=raw_copy,
            return_scores=True,
            origin=origin,
            cross_talk=None,
            calibration=None,
            verbose=False,
        )

        bad_channels = list(set(auto_noisy + auto_flat))
        logger.trace(f"Found {len(bad_channels)} bad MEG channels: {bad_channels}")

    except Exception as e:
        logger.error(f"Maxwell filtering failed: {e}. Returning empty bad channels list.")
        bad_channels = []

    return bad_channels
