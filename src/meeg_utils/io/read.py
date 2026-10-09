"""Reading recordings."""

from __future__ import annotations

import json
from pathlib import Path

import mne
from loguru import logger
from mne.io import BaseRaw
from mne_bids import BIDSPath, get_bids_path_from_fname, read_raw_bids


def read(path: str | Path | BIDSPath, **kwargs) -> BaseRaw:
    """Read a recording into memory, using BIDS metadata when available.

    BIDS recordings are read with :func:`mne_bids.read_raw_bids`, so channel
    types, bad channels (``channels.tsv``), events, line frequency and
    head-coordinate information come from the BIDS sidecars. A plain path
    that lies inside a BIDS dataset is read the same way; any other file is
    read with :func:`mne.io.read_raw`.

    Parameters
    ----------
    path : str | Path | BIDSPath
        The recording.
    **kwargs
        Passed to :func:`mne_bids.read_raw_bids` (as ``extra_params`` entries
        when reading BIDS data) or :func:`mne.io.read_raw`.

    Returns
    -------
    Raw
        The preloaded recording.

    Raises
    ------
    FileNotFoundError
        If the recording does not exist.
    """
    bids_path = path if isinstance(path, BIDSPath) else _as_bids_path(Path(path))

    if bids_path is not None:
        if bids_path.fpath is None or not Path(bids_path.fpath).exists():
            raise FileNotFoundError(f"BIDS recording not found: {bids_path}")
        raw = read_raw_bids(bids_path, extra_params=kwargs or None, verbose=False)
        raw.load_data(verbose=False)
    else:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Recording not found: {path}")
        raw = mne.io.read_raw(path, preload=True, verbose=False, **kwargs)

    if raw.info["line_freq"] is None:
        logger.warning(
            f"The power line frequency of {path} is unknown; steps that need it "
            "(e.g. line-noise removal) will ask for it explicitly."
        )
    return raw


def _as_bids_path(path: Path) -> BIDSPath | None:
    """Return the BIDSPath of a raw file inside a BIDS dataset, else None.

    Files inside BIDS derivatives are not raw BIDS data (they lack the raw
    sidecars) and are read as plain files.
    """
    root = _find_bids_root(path)
    if root is None or _is_derivative(root):
        return None
    try:
        bids_path = get_bids_path_from_fname(path, check=False, verbose=False)
    except Exception:  # not a BIDS file name
        return None
    if bids_path.subject is None:
        return None
    return bids_path.update(root=root)


def _find_bids_root(path: Path) -> Path | None:
    """Return the closest parent with a dataset_description.json, if any."""
    for parent in path.resolve().parents:
        if (parent / "dataset_description.json").is_file():
            return parent
    return None


def _is_derivative(root: Path) -> bool:
    try:
        description = json.loads((root / "dataset_description.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return str(description.get("DatasetType", "raw")).lower() == "derivative"
