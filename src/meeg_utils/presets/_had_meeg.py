"""The HAD-MEEG preprocessing (https://github.com/colehank/HAD-MEEG), corrected."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core import Pipeline

#: Sphere origin (head frame, m) used by HAD-MEEG for MEG bad-channel
#: detection and interpolation.
MEG_ORIGIN = (0.0, 0.0, 0.04)


def had_meeg(
    *, datatype: str, stage: str = "preprocessing", head_destination: Any = None
) -> Pipeline:
    """HAD-MEEG per-run preprocessing for CTF MEG or EEG, with known issues fixed.

    The steps follow the original pipeline: 0.1-100 Hz band-pass, 250 Hz,
    bad channels (Maxwell for MEG, PREP for EEG), interpolation, ZapLine
    line-noise removal, average reference (EEG) or 3rd-order gradient
    compensation (CTF MEG), and ICA labelled with MEGnet (MEG) or ICLabel
    (EEG). The differences from the original code, each with its reason,
    are listed in ``docs/presets/had-meeg.md``; the main ones:

    - line noise: ZapLine-plus (adaptive number of components, guards
      against over-cleaning) instead of removing a fixed 22 % of components;
      the line frequency comes from the data (BIDS ``PowerLineFrequency``);
    - ICA: 20 components for MEG, as MEGnet was designed for; components
      are removed automatically only when the artifact probability is at
      least 0.8 (the original removed every arg-max artifact label and
      relied on manual review, which remains possible with
      :meth:`~meeg_utils.steps.ICA.relabel`);
    - EEG bad channels: all PREP criteria instead of three;
    - the data are filtered once (the original filtered again before
      applying ICA) and ``first_samp`` is preserved.

    ``stage="epochs"`` is the original epoching stage, applied per run to
    the preprocessed data, again with its issues fixed: mastoids (M1/M2) are
    removed *before* the average reference, only a 40 Hz low-pass is added
    (no third 0.1 Hz high-pass), 200 Hz, epochs of -0.1 to 2 s around
    ``"video on"``, mean baseline instead of z-scoring each epoch on 100 ms,
    and for MEG a head alignment in the right direction to
    ``head_destination``. Combine the runs with :func:`meeg_utils.epochs.combine`.

    Parameters
    ----------
    datatype : {"meg", "eeg"}
        Which recordings the pipeline is for.
    stage : {"preprocessing", "epochs"}
        Which stage of the original pipeline.
    head_destination : Transform | array | str | None
        MEG epochs stage: the common head position of the session's runs,
        usually ``meu.io.average_dev_head_t(runs)``. Required for MEG.

    Returns
    -------
    Pipeline
        The unfitted pipeline.
    """
    from .. import steps as S
    from ..core import Pipeline

    if datatype not in ("meg", "eeg"):
        raise ValueError(f"datatype must be 'meg' or 'eeg', got {datatype!r}")
    if stage == "epochs":
        return _epochs_stage(datatype, head_destination)
    if stage != "preprocessing":
        raise ValueError(f"stage must be 'preprocessing' or 'epochs', got {stage!r}")

    common = [
        ("filter", S.Filter(0.1, 100.0)),
        ("resample", S.Resample(250.0)),
    ]
    if datatype == "meg":
        specific = [
            ("bads", S.BadChannels(method="maxwell", origin=MEG_ORIGIN)),
            ("interpolate", S.Interpolate(origin=MEG_ORIGIN)),
            ("line_noise", S.LineNoise(method="zapline-plus")),
            ("reference", S.Reference(eeg=None, ctf_grade=3)),
            ("ica", S.ICA(n_components=20, picks="meg", method="infomax", labeler="megnet")),
        ]
    else:
        specific = [
            ("bads", S.BadChannels(method="prep", random_state=42)),
            ("interpolate", S.Interpolate()),
            ("line_noise", S.LineNoise(method="zapline-plus")),
            ("reference", S.Reference(eeg="average")),
            ("ica", S.ICA(n_components=20, picks="eeg", method="infomax", labeler="iclabel")),
        ]
    return Pipeline([*common, *specific])


def _epochs_stage(datatype: str, head_destination: Any) -> Pipeline:
    from .. import steps as S
    from ..core import Pipeline

    epoching = [
        ("epoch", S.Epoch("video on", -0.1, 2.0, baseline=None)),
        ("baseline", S.Baseline((None, 0.0), mode="mean")),
    ]
    filtering = [("lowpass", S.Filter(None, 40.0)), ("resample", S.Resample(200.0))]
    if datatype == "eeg":
        steps = [
            ("drop_mastoids", S.DropChannels(["M1", "M2"], on_missing="ignore")),
            ("reference", S.Reference(eeg="average")),
            *filtering,
            *epoching,
        ]
    else:
        if head_destination is None:
            raise ValueError(
                "The MEG epochs stage aligns every run to a common head position; pass "
                "head_destination=meu.io.average_dev_head_t(runs)."
            )
        steps = [
            *filtering,
            ("align", S.HeadAlign(head_destination, origin=MEG_ORIGIN)),
            *epoching,
        ]
    return Pipeline(steps)
