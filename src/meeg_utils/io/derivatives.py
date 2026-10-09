"""Writing processed data as BIDS derivatives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from mne import Evoked
from mne.epochs import BaseEpochs
from mne.io import BaseRaw
from mne_bids import BIDSPath

from .system import get_datatypes

if TYPE_CHECKING:
    from ..core import Pipeline

BIDS_VERSION = "1.9.0"
CODE_URL = "https://github.com/colehank/meeg-utils"


def save_derivative(
    inst: BaseRaw | BaseEpochs | Evoked,
    source: BIDSPath | str | Path,
    root: str | Path,
    *,
    pipeline: Pipeline | None = None,
    desc: str = "preproc",
    overwrite: bool = False,
) -> Path:
    """Save processed data as a BIDS derivative, with a provenance sidecar.

    The data are written to
    ``<root>/sub-<sub>/[ses-<ses>/]<datatype>/<entities>_desc-<desc>_<suffix>.fif``,
    where ``<suffix>`` is the datatype (``meg``/``eeg``) for Raw data, ``epo``
    for Epochs and ``ave`` for Evoked. A JSON sidecar with the same name holds
    the source file and, if ``pipeline`` is given, its configuration,
    provenance and QC metrics. ``<root>/dataset_description.json`` is created
    if missing.

    For a non-BIDS ``source``, the file is written directly to ``root`` as
    ``<source stem>_desc-<desc>_<suffix>.fif``.

    Parameters
    ----------
    inst : Raw | Epochs | Evoked
        The processed data.
    source : BIDSPath | str | Path
        The recording the data were derived from; its BIDS entities name the
        output.
    root : str | Path
        Root of the derivatives dataset, e.g. ``<bids_root>/derivatives/meeg-utils``.
    pipeline : Pipeline | None
        The (fitted) pipeline that produced the data.
    desc : str
        Value of the BIDS ``desc`` entity.
    overwrite : bool
        Whether to overwrite existing files.

    Returns
    -------
    Path
        The data file written.

    Raises
    ------
    FileExistsError
        If the output exists and ``overwrite`` is False.
    """
    root = Path(root)
    datatype = _datatype(inst, source)
    suffix = _suffix(inst, datatype)

    if isinstance(source, BIDSPath):
        out = source.copy().update(
            root=root,
            datatype=datatype,
            description=desc,
            suffix=suffix,
            extension=".fif",
            split=None,
            check=False,
        )
        fname = Path(str(out.fpath))
        source_ref = _bids_uri(source)
    else:
        fname = root / f"{Path(source).stem}_desc-{desc}_{suffix}.fif"
        source_ref = str(source)

    sidecar = fname.with_suffix(".json")
    if not overwrite:
        for existing in (fname, sidecar):
            if existing.exists():
                raise FileExistsError(f"{existing} exists; pass overwrite=True to replace it.")

    fname.parent.mkdir(parents=True, exist_ok=True)
    _ensure_dataset_description(root)

    if isinstance(inst, Evoked):
        inst.save(fname, overwrite=overwrite, verbose=False)
    else:
        inst.save(fname, overwrite=overwrite, split_naming="bids", verbose=False)

    sidecar.write_text(
        json.dumps(_sidecar(source_ref, desc, pipeline), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Saved {type(inst).__name__} derivative to {fname}")
    return fname


def _datatype(inst: BaseRaw | BaseEpochs | Evoked, source: BIDSPath | str | Path) -> str:
    if isinstance(source, BIDSPath) and source.datatype in ("meg", "eeg"):
        return str(source.datatype)
    datatypes = get_datatypes(inst)
    if "meg" in datatypes:
        return "meg"
    if "eeg" in datatypes:
        return "eeg"
    raise ValueError("Cannot determine the datatype: the data contain neither MEG nor EEG.")


def _suffix(inst: BaseRaw | BaseEpochs | Evoked, datatype: str) -> str:
    if isinstance(inst, BaseEpochs):
        return "epo"
    if isinstance(inst, Evoked):
        return "ave"
    return datatype


def _bids_uri(source: BIDSPath) -> str:
    if source.fpath is None:
        return str(source)
    fpath = Path(str(source.fpath))
    try:
        return f"bids::{fpath.relative_to(Path(str(source.root))).as_posix()}"
    except (TypeError, ValueError):
        return str(fpath)


def _sidecar(source_ref: str, desc: str, pipeline: Pipeline | None) -> dict[str, Any]:
    from ..core.pipeline import _to_serializable

    sidecar: dict[str, Any] = {
        "Description": f"meeg-utils derivative (desc-{desc})",
        "Sources": [source_ref],
    }
    if pipeline is not None:
        meeg_utils: dict[str, Any] = {"pipeline": pipeline.to_dict()}
        if hasattr(pipeline, "provenance_"):
            meeg_utils["provenance"] = pipeline.provenance_
            meeg_utils["qc"] = _to_serializable(pipeline.qc_)
        sidecar["MeegUtils"] = meeg_utils
    return sidecar


def _ensure_dataset_description(root: Path) -> None:
    fname = root / "dataset_description.json"
    if fname.exists():
        return
    from importlib.metadata import PackageNotFoundError, version

    try:
        meeg_utils_version = version("meeg-utils")
    except PackageNotFoundError:  # pragma: no cover
        meeg_utils_version = "n/a"
    root.mkdir(parents=True, exist_ok=True)
    description = {
        "Name": "meeg-utils derivatives",
        "BIDSVersion": BIDS_VERSION,
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "meeg-utils", "Version": meeg_utils_version, "CodeURL": CODE_URL}],
    }
    fname.write_text(json.dumps(description, indent=2), encoding="utf-8")
