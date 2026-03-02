"""ICA-based artifact removal for EEG and MEG data."""

from __future__ import annotations

from typing import Literal

import numpy as np
from loguru import logger
from mne.io import BaseRaw
from mne.preprocessing import ICA
from mne_icalabel import label_components

DType = Literal["eeg", "meg"]


def apply_ica_pipeline(
    raw: BaseRaw,
    datatype: DType,
    n_components: int = 20,
    method: str = "infomax",
    regress: bool = False,
    manual_labels: list[str] | None = None,
    random_state: int = 42,
) -> BaseRaw:
    """Apply ICA pipeline for artifact removal.

    Parameters
    ----------
    raw : BaseRaw
        Raw data.
    datatype : {"eeg", "meg"}
        Data type.
    n_components : int, optional
        Number of ICA components. Default is 20.
    method : str, optional
        ICA method. Default is "infomax".
    regress : bool, optional
        Whether to regress out artifacts. Default is False.
    manual_labels : list[str] | None, optional
        Manual component labels. If None, uses automatic labeling.
    random_state : int, optional
        Random seed. Default is 42.

    Returns
    -------
    BaseRaw
        Raw data with ICA applied (if regress=True) or original (if regress=False).
    """
    logger.info(f"Running ICA with {n_components} components using {method}...")

    # Prepare raw for ICA (1 Hz highpass recommended)
    raw_ica = _prepare_raw_for_ica(raw, datatype)

    # Adjust n_components if needed (cannot exceed number of channels)
    n_channels = len(raw_ica.ch_names)
    if n_components > n_channels:
        logger.warning(
            f"Requested {n_components} ICA components but only {n_channels} channels available. "
            f"Reducing to {n_channels} components."
        )
        n_components = n_channels

    # Fit ICA
    ica = _fit_ica(raw_ica, n_components, method, random_state)

    # Label components
    if manual_labels is None:
        # Automatic labeling
        ica, _ = _label_components_auto(raw_ica, ica, datatype)
    else:
        # Manual labeling
        # First get auto labels, then override with manual
        ica_temp, _ = _label_components_auto(raw_ica, ica, datatype)
        ica = _label_components_manual(ica_temp, manual_labels)

    logger.info(f"ICA excluded components: {ica.exclude}")

    # Apply ICA if requested
    if regress:
        logger.info("Regressing out artifact components...")
        # Prepare raw for regression (matching preprocessing)
        raw_regress = _prepare_raw_for_regression(raw, datatype)
        cleaned = ica.apply(raw_regress)
        logger.success("ICA regression completed.")
        return cleaned
    else:
        logger.info("ICA fitted but not regressed (regress=False).")
        return raw


def _prepare_raw_for_ica(
    raw: BaseRaw,
    datatype: DType,
    highpass: float = 1.0,
    lowpass: float | None = None,
    sfreq: float | None = None,
) -> BaseRaw:
    """Prepare raw data for ICA decomposition.

    ICA typically works better with 1 Hz highpass filtering.
    """
    logger.trace("Preparing data for ICA (1 Hz highpass)...")

    raw_prep = raw.copy()

    # Pick appropriate channels
    if datatype == "eeg":
        raw_prep.pick("eeg")
    elif datatype == "meg":
        raw_prep.pick(["mag", "grad", "planar1", "planar2"], exclude=[])

    # Determine appropriate lowpass based on sampling rate
    if sfreq is None:
        sfreq = raw_prep.info["sfreq"]

    if lowpass is None:
        # Use current lowpass or set to 90% of Nyquist
        current_lowpass = raw_prep.info.get("lowpass")
        nyquist = sfreq / 2
        if current_lowpass is not None and current_lowpass < nyquist:
            lowpass = current_lowpass
        else:
            lowpass = nyquist * 0.9  # 90% of Nyquist to be safe

    # Ensure lowpass is less than Nyquist
    nyquist = sfreq / 2
    if lowpass >= nyquist:
        lowpass = nyquist * 0.9
        logger.warning(
            f"Lowpass frequency adjusted to {lowpass:.1f} Hz (90% of Nyquist = {nyquist:.1f} Hz)"
        )

    # Filter and resample
    raw_prep.filter(l_freq=highpass, h_freq=lowpass, verbose=False)
    if sfreq != raw_prep.info["sfreq"]:
        raw_prep.resample(sfreq, verbose=False)

    return raw_prep


def _prepare_raw_for_regression(
    raw: BaseRaw,
    datatype: DType,
    highpass: float = 0.1,
    lowpass: float | None = None,
    sfreq: float | None = None,
) -> BaseRaw:
    """Prepare raw data for ICA regression (artifact removal)."""
    logger.trace("Preparing data for ICA regression...")

    raw_prep = raw.copy()

    # Pick appropriate channels
    if datatype == "eeg":
        raw_prep.pick("eeg")
    elif datatype == "meg":
        raw_prep.pick(["mag", "grad", "planar1", "planar2"], exclude=[])

    # Determine appropriate lowpass based on sampling rate
    if sfreq is None:
        sfreq = raw_prep.info["sfreq"]

    if lowpass is None:
        # Use current lowpass or set to 90% of Nyquist
        current_lowpass = raw_prep.info.get("lowpass")
        nyquist = sfreq / 2
        if current_lowpass is not None and current_lowpass < nyquist:
            lowpass = current_lowpass
        else:
            lowpass = nyquist * 0.9

    # Ensure lowpass is less than Nyquist
    nyquist = sfreq / 2
    if lowpass >= nyquist:
        lowpass = nyquist * 0.9
        logger.warning(
            f"Lowpass frequency adjusted to {lowpass:.1f} Hz (90% of Nyquist = {nyquist:.1f} Hz)"
        )

    # Filter and resample
    raw_prep.filter(l_freq=highpass, h_freq=lowpass, verbose=False)
    if sfreq != raw_prep.info["sfreq"]:
        raw_prep.resample(sfreq, verbose=False)

    return raw_prep


def _fit_ica(
    raw: BaseRaw,
    n_components: int,
    method: str,
    random_state: int,
) -> ICA:
    """Fit ICA on prepared raw data."""
    logger.trace(f"Fitting ICA with {n_components} components...")

    # Set fit parameters for extended infomax
    fit_params = None
    if method in ("infomax", "picard"):
        fit_params = dict(extended=True)

    ica = ICA(
        n_components=n_components,
        method=method,
        random_state=random_state,
        max_iter="auto",
        fit_params=fit_params,
    )

    ica.fit(raw, verbose=False)
    logger.trace(f"ICA fitted: {ica.n_components_} components extracted.")

    return ica


def _label_components_auto(
    raw: BaseRaw,
    ica: ICA,
    datatype: DType,
) -> tuple[ICA, dict]:
    """Automatically label ICA components using ICLabel or MEGNet."""
    logger.trace("Labeling ICA components automatically...")

    # Choose method based on datatype
    method = "iclabel" if datatype == "eeg" else "megnet"
    good_labels = ["brain", "other", "brain/other"]

    # Run automatic labeling
    try:
        ic_labels = label_components(raw, ica, method=method)
    except Exception as e:
        logger.error(f"Automatic labeling failed: {e}. Using default labels.")
        # Create default labels (all "other")
        ic_labels = {
            "labels": ["other"] * ica.n_components_,
            "y_pred_proba": np.ones(ica.n_components_),
        }

    # Set labels in ICA object
    labels_dict = {}
    for label in set(ic_labels["labels"]):
        labels_dict[label] = [
            int(idx) for idx, val in enumerate(ic_labels["labels"]) if val == label
        ]
    ica.labels_ = labels_dict

    # Set exclude list (non-brain components)
    exclude = [int(idx) for idx, val in enumerate(ic_labels["labels"]) if val not in good_labels]
    ica.exclude = exclude

    logger.trace(f"Auto-labeled: {len(exclude)} components marked as artifacts.")

    return ica, ic_labels


def _label_components_manual(
    ica: ICA,
    manual_labels: list[str],
) -> ICA:
    """Apply manual labels to ICA components."""
    logger.trace("Applying manual labels to ICA components...")

    if len(manual_labels) != ica.n_components_:
        raise ValueError(
            f"Number of manual labels ({len(manual_labels)}) must match "
            f"number of ICA components ({ica.n_components_})."
        )

    # Set labels
    labels_dict = {}
    for label in set(manual_labels):
        labels_dict[label] = [int(idx) for idx, val in enumerate(manual_labels) if val == label]
    ica.labels_ = labels_dict

    # Set exclude list
    good_labels = ["brain", "other", "brain/other"]
    exclude = [int(idx) for idx, val in enumerate(manual_labels) if val not in good_labels]
    ica.exclude = exclude

    logger.trace(f"Manual labels applied: {len(exclude)} components marked as artifacts.")

    return ica
