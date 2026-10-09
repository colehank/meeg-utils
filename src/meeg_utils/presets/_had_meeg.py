"""The HAD-MEEG preprocessing (https://github.com/colehank/HAD-MEEG), corrected."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core import Pipeline

#: Sphere origin (head frame, m) used by HAD-MEEG for MEG bad-channel
#: detection and interpolation.
MEG_ORIGIN = (0.0, 0.0, 0.04)


def had_meeg(*, datatype: str) -> Pipeline:
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

    Cross-run steps of the original epoching stage are not part of this
    per-run preset; head alignment is available as
    :class:`~meeg_utils.steps.HeadAlign` (the original mapped in the
    reverse direction).

    Parameters
    ----------
    datatype : {"meg", "eeg"}
        Which recordings the pipeline is for.

    Returns
    -------
    Pipeline
        The unfitted pipeline.
    """
    from .. import steps as S
    from ..core import Pipeline

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
    elif datatype == "eeg":
        specific = [
            ("bads", S.BadChannels(method="prep", random_state=42)),
            ("interpolate", S.Interpolate()),
            ("line_noise", S.LineNoise(method="zapline-plus")),
            ("reference", S.Reference(eeg="average")),
            ("ica", S.ICA(n_components=20, picks="eeg", method="infomax", labeler="iclabel")),
        ]
    else:
        raise ValueError(f"datatype must be 'meg' or 'eeg', got {datatype!r}")
    return Pipeline([*common, *specific])
