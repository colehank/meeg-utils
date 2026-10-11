"""Recommended pipelines: every parameter traced to a published default.

The sources of each value are listed in ``docs/presets/recommended.md``.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core import Pipeline

#: MEG systems with a recommended pipeline (others: build one from the steps).
MEG_SYSTEMS = ("neuromag", "ctf", "kit")
ANALYSES = ("erp", "rest")
STAGES = ("preprocessing", "epochs")

#: High-pass for all presets: Tanner, Morgan-Short & Luck (2015), Psychophysiology
#: 52:997, found artifactual ERP effects from 0.3 Hz up and recommend 0.1 Hz or lower.
HIGHPASS = 0.1
#: Low-pass of the ERP epochs: MNE-BIDS-Pipeline default ``h_freq``.
LOWPASS = 40.0
#: Epoch window and baseline: MNE-BIDS-Pipeline defaults.
TMIN, TMAX, BASELINE = -0.2, 0.5, (None, 0.0)
#: Candidate numbers of interpolated channels for local autoreject:
#: MNE-BIDS-Pipeline default ``autoreject_n_interpolate``.
N_INTERPOLATE = [4, 8, 16]
#: Minimum artifact probability for removing an ICA component:
#: MNE-BIDS-Pipeline default ``ica_exclusion_thresholds``.
ICA_THRESHOLD = 0.8
#: Sampling rate, number of components and algorithm MEGnet was trained with
#: (Treacher et al., 2021; checked by mne-icalabel).
MEGNET_SFREQ, MEGNET_COMPONENTS = 250.0, 20


def recommended(
    analysis: str,
    *,
    datatype: str,
    stage: str = "preprocessing",
    system: str | None = None,
    event_id: Any = None,
    tmin: float = TMIN,
    tmax: float = TMAX,
    epoch_duration: float | None = None,
    head_destination: Any = None,
) -> Pipeline:
    """Build a recommended pipeline (see :func:`eeg_erp` and the other wrappers)."""
    if analysis not in ANALYSES:
        raise ValueError(f"analysis must be one of {ANALYSES}, got {analysis!r}")
    if datatype not in ("eeg", "meg"):
        raise ValueError(f"datatype must be 'eeg' or 'meg', got {datatype!r}")
    if stage == "preprocessing":
        return _preprocessing(datatype, system)
    if stage == "epochs":
        return _epochs(analysis, datatype, event_id, tmin, tmax, epoch_duration, head_destination)
    raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")


def _preprocessing(datatype: str, system: str | None) -> Pipeline:
    from .. import steps as S
    from ..core import Pipeline

    if datatype == "eeg":
        return Pipeline(
            [
                ("bridged", S.BridgedElectrodes()),
                ("highpass", S.Filter(HIGHPASS, None)),
                ("line_noise", S.LineNoise("zapline-plus")),
                ("bads", S.BadChannels("prep")),
                ("interpolate", S.Interpolate()),
                ("reference", S.Reference("average")),
                (
                    "ica",
                    S.ICA(
                        n_components=0.999999,
                        picks="eeg",
                        method="infomax",
                        labeler="iclabel",
                        threshold=ICA_THRESHOLD,
                    ),
                ),
            ]
        )

    if system not in MEG_SYSTEMS:
        raise ValueError(
            f"The MEG presets need system= one of {MEG_SYSTEMS} (got {system!r}); "
            "meu.io.detect_system(raw) tells which. Other systems (BTi, Artemis123, OPM) have "
            "no recommended pipeline yet: build one from the steps (e.g. S.HFC for OPM)."
        )
    noise: list[tuple[str, Any]]
    if system == "neuromag":
        noise = [
            ("bads", S.BadChannels("maxwell")),
            ("sss", S.Maxwell()),  # rebuilds the bad channels
        ]
    elif system == "ctf":
        noise = [
            ("bads", S.BadChannels("maxwell")),
            ("interpolate", S.Interpolate()),
            ("reference", S.Reference(eeg=None, ctf_grade=3)),
        ]
    else:  # kit
        noise = [("regression", S.Regression("ref_meg"))]
    return Pipeline(
        [
            *noise,
            ("highpass", S.Filter(HIGHPASS, None)),
            # resample first: ZapLine-plus on 273 channels at 1200 Hz (HAD-MEEG, 6 min)
            # needed more than 15 GB of memory; MEGnet needs 250 Hz anyway
            ("resample", S.Resample(MEGNET_SFREQ)),
            ("line_noise", S.LineNoise("zapline-plus")),
            (
                "ica",
                S.ICA(
                    n_components=MEGNET_COMPONENTS,
                    picks="meg",
                    method="infomax",
                    labeler="megnet",
                    threshold=ICA_THRESHOLD,
                ),
            ),
        ]
    )


def _epochs(
    analysis: str,
    datatype: str,
    event_id: Any,
    tmin: float,
    tmax: float,
    epoch_duration: float | None,
    head_destination: Any,
) -> Pipeline:
    from .. import steps as S
    from ..core import Pipeline

    steps: list[tuple[str, Any]] = []
    if analysis == "erp":
        steps.append(("lowpass", S.Filter(None, LOWPASS)))
    if datatype == "meg" and head_destination is not None:
        steps.append(("align", S.HeadAlign(head_destination)))
    if analysis == "erp":
        if event_id is None:
            raise ValueError(
                "The ERP epochs stage needs event_id= (the conditions to epoch, e.g. "
                "['target', 'standard'])."
            )
        steps.append(("epoch", S.Epoch(event_id, tmin, tmax, baseline=BASELINE)))
    else:
        if epoch_duration is None:
            raise ValueError(
                "The resting-state epochs stage needs epoch_duration= (s); it sets the "
                "frequency resolution of later spectra (1 / duration)."
            )
        steps.append(("epoch", S.FixedLengthEpochs(epoch_duration)))
    steps.append(("autoreject", S.AutoReject("local", n_interpolate=list(N_INTERPOLATE))))
    return Pipeline(steps)


_COMMON = """
    ``stage="preprocessing"`` (per run) and ``stage="epochs"`` (applied to
    the preprocessed runs; combine them afterwards with
    :func:`meeg_utils.epochs.combine`). The source of every parameter is
    listed in ``docs/presets/recommended.md``.
"""


def eeg_erp(**kwargs: Any) -> Pipeline:
    """EEG event-related potentials: PREP, ZapLine-plus, ICLabel, autoreject.

    Preprocessing: bridged electrodes, 0.1 Hz high-pass, ZapLine-plus,
    PREP bad channels, spherical-spline interpolation, average reference,
    extended-Infomax ICA labelled with ICLabel. Epochs: 40 Hz low-pass,
    epochs of ``tmin``-``tmax`` (-0.2-0.5 s) around ``event_id`` with a
    pre-stimulus baseline, local autoreject.
    """
    return recommended("erp", datatype="eeg", **kwargs)


def eeg_rest(**kwargs: Any) -> Pipeline:
    """Resting-state EEG: the ERP preprocessing, then fixed-length epochs and autoreject.

    Epochs: consecutive windows of ``epoch_duration`` seconds (no low-pass,
    no baseline), local autoreject.
    """
    return recommended("rest", datatype="eeg", **kwargs)


def meg_erp(**kwargs: Any) -> Pipeline:
    """MEG event-related fields for Neuromag, CTF or KIT (``system=``).

    Preprocessing: system-specific noise reduction (Neuromag: Maxwell bad
    channels and SSS; CTF: Maxwell bad channels, interpolation, grade-3
    synthetic gradiometers; KIT: reference regression), 0.1 Hz high-pass,
    250 Hz, ZapLine-plus, 20-component Infomax ICA labelled with MEGnet.
    Epochs: as for EEG, after aligning to ``head_destination`` when given.
    """
    return recommended("erp", datatype="meg", **kwargs)


def meg_rest(**kwargs: Any) -> Pipeline:
    """Resting-state MEG: the MEG preprocessing, then fixed-length epochs and autoreject."""
    return recommended("rest", datatype="meg", **kwargs)


def _signature(datatype: str) -> inspect.Signature:
    params = [
        p
        for name, p in inspect.signature(recommended).parameters.items()
        if name not in ("analysis", "datatype") and (datatype == "meg" or name not in _MEG_ONLY)
    ]
    return inspect.Signature(params, return_annotation="Pipeline")


_MEG_ONLY = ("system", "head_destination")
for _f, _datatype in ((eeg_erp, "eeg"), (eeg_rest, "eeg"), (meg_erp, "meg"), (meg_rest, "meg")):
    _f.__doc__ = (_f.__doc__ or "") + _COMMON
    _f.__signature__ = _signature(_datatype)  # type: ignore[attr-defined]
