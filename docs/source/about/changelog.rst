Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

Validated on real data: the first two runs of HAD-MEEG sub-01 (OpenNeuro
ds007353; CTF MEG and Neuroscan EEG) through QC, ``had-meeg`` and the
recommended presets. What it revealed is in ``docs/validation/ds007353-sub-01.md``
and fixed below.

Changed
~~~~~~~

* ``qc.Bridging`` / ``steps.BridgedElectrodes`` confirm MNE's candidate
  pairs by their correlation after average reference in 15-30 Hz
  (``min_correlation=0.98``): with small signals, MNE's absolute distance
  cutoff also flags unbridged neighbours.
* ``BadChannels("prep")`` detects after PREP's robust average reference
  (``robust_reference=True``), as the PREP paper does; channels next to the
  recording reference are no longer flagged.
* Maxwell bad-channel detection no longer marks MEG reference sensors bad
  (they cannot be interpolated); noisy ones are reported with a warning.
* ``origin="auto"`` falls back to (0, 0, 0.04) m with a warning when there
  are no head-shape points (bad channels, interpolation, HeadAlign, Maxwell).
* MEG presets resample before ZapLine-plus (memory).
* ``qc.HeartRate`` judges the beat detection by its intervals: when more
  than 10 % are irregular the rate is not reported (``reliable=False``),
  a ``"warn"`` with an ECG channel and ``"info"`` with the synthetic ECG.
* QC checks only give verdicts where the threshold holds for any dataset.
  Measurements whose acceptable range depends on the population, task or
  equipment are reported with the new level ``"info"`` (not a flag) unless
  a threshold is passed: ``HeadMovement(warn_mm)``, ``Muscle(warn_fraction)``,
  ``Blinks(min_rate)`` (a flat EOG is still flagged), ``HeartRate(min_bpm,
  max_bpm)``, ``Impedance(warn_kohm)``, ``ChpiSNR(max_drop_db)``,
  ``EmptyRoom(max_days)`` now default to ``None``. Dataset-level outliers
  (``inspect_dataset``, ``meu report``) cover them.
* ``BadChannels("prep")`` no longer runs PyPREP's band-power criterion
  (``find_bad_by_PSD``), which is not part of PREP and flags frontal
  channels for their blink power (three in MNE's sample EEG); ``psd=True``
  turns it on.
* ``qc.SquidJumps`` requires a lasting level change (200 ms windows,
  larger than three times the local spread): on MNE's sample recording
  the 50 ms windows took noise bursts and 30 ms glitches for jumps (16 in
  7 channels; now 8, all real steps, in MEG 2313).
* ``qc.Amplitude`` counts clipping from ``min_clip_duration`` (10 ms, at
  least 4 samples) instead of 3 samples, which quantized data reach at
  their peaks by chance.
* ``steps.ICA``: ``plot("components")`` failed on Neuromag data (MNE titles
  the components ``ICA000 (mag)``); the number of components is capped at
  the rank in ``info`` as well as the data rank (after SSS, ZapLine-plus
  raises the data rank above the SSS rank).
* ``meu.Dataset`` leaves out empty-room recordings; it and
  ``qc.inspect_dataset`` no longer take the Neuromag cross-talk and
  calibration files (``*_acq-crosstalk_meg.fif``) or files under
  ``derivatives/`` for recordings.
* ``steps.AutoReject`` runs on the MEG/EEG and reference channels only,
  working around an autoreject indexing error when other channels come
  first (CTF), and writes the result back to all channels.

Added
~~~~~

* ``meu run --set STEP__PARAM=VALUE`` changes step parameters.

* ``meu.qc``: acquisition-quality checks (amplitude, bridging, impedance,
  outlying channels, narrowband noise, muscle, blinks, heart rate, head
  movement, digitization, events) with documented thresholds and figures;
  ``inspect`` for one recording, ``inspect_dataset`` for a dataset with
  outlying recordings flagged.
* Second batch of checks: ``SquidJumps``, ``ChpiSNR``, ``EmptyRoom``
  (found in BIDS), ``BidsMetadata`` and ``Photodiode``.
* ``meu.report.build``: HTML reports of QC results and fitted pipelines.
* ``steps.BridgedElectrodes``: repair of bridged EEG electrodes.
* ``BadChannels(cross_talk="auto", calibration="auto")``: Neuromag
  cross-talk and fine-calibration files are found in BIDS.
* The ``meu`` command line: ``meu qc``, ``meu run``, ``meu preset`` (list
  presets, write one as YAML) and ``meu report`` (report of a saved
  derivative, or summary of a derivatives folder with outlying runs:
  ``meu.report.from_derivative``, ``meu.report.summarize``).
* Epoching: ``steps.Epoch``, ``steps.Baseline`` (with a guard against
  z-scoring on short baselines), ``steps.DropChannels``,
  ``steps.AutoReject`` (autoreject), ``steps.FixedLengthEpochs`` and
  ``meu.epochs.combine`` for runs.
* ``meu.Dataset`` and ``meu.process_dataset``: select the recordings of a
  BIDS dataset, preprocess every run, epoch them (MEG runs aligned to their
  average head position) and combine the runs of each session; resumable,
  with a log of every run and session (``meu run --epochs``).
* Recommended presets ``eeg-erp``, ``eeg-rest``, ``meg-erp`` and
  ``meg-rest`` (``stage="preprocessing"`` / ``"epochs"``; MEG by system),
  with the source of every parameter in ``docs/presets/recommended.md``.
* ``Pipeline.preset("had-meeg", stage="epochs")``: the corrected HAD-MEEG
  epoching stage.
* ``steps.Maxwell``: SSS / tSSS for Neuromag data with cross-talk and
  fine-calibration files from BIDS, movement compensation from cHPI (coil
  signals removed), and a destination head position.
* ``steps.HFC`` (OPM homogeneous / harmonic field correction) and
  ``steps.Regression`` (reference-sensor or EOG regression).
* ``steps.BadSegments`` (``BAD_`` annotations by amplitude, flatness or
  muscle activity), ``steps.ASR`` and ``steps.SNS`` (mne-denoise). Steps
  may declare ``adds_annotations``; the contract checks that existing
  annotations are kept.
* ``steps.ByChannelType``: separate steps per channel type (e.g. MEG and
  EEG), with nested parameters (``by_type__eeg__ica__n_components``).
* Figures that exist only for some fits (e.g. head movement) are left out
  of ``plot()`` and explain why when requested.

[0.2.0]
-------

A redesign around scikit-learn style steps and pipelines. This release is
not backwards compatible.

Added
~~~~~

* ``Step`` and ``Pipeline`` (``meu.Pipeline``): ``fit`` / ``transform`` /
  ``fit_transform``, nested ``set_params``, ``clone``, slicing, editing,
  YAML configurations, type checking of the step chain, provenance and QC
  metrics (``qc_``, ``provenance_``).
* Steps (``meu.steps``): ``Filter``, ``Resample``, ``LineNoise``
  (ZapLine-plus, ZapLine, notch; via mne-denoise), ``BadChannels`` (full
  PREP, Maxwell for Neuromag and CTF), ``Interpolate``, ``Reference``,
  ``ICA`` (ICLabel / MEGnet with a probability threshold, ``relabel``),
  ``HeadAlign`` (MEG head-position alignment).
* ``plot()`` on every step and pipeline for visual quality control.
* ``meu.io``: BIDS-aware ``read``, ``detect_system``, ``save_derivative``
  (BIDS derivatives with a provenance sidecar), ``existing_derivatives``,
  ``average_dev_head_t``.
* ``meu.process`` for batch processing, with per-recording error records.
* Presets: ``Pipeline.preset("had-meeg", datatype=...)``, the corrected
  HAD-MEEG preprocessing.
* ``meeg_utils.testing.check_step``: contract checks for steps.

Changed
~~~~~~~

* Requires MNE-Python >= 1.13.2, mne-bids >= 0.20, mne-icalabel >= 0.10,
  pyprep >= 0.9.
* The library no longer configures logging on import; use
  ``meu.setup_logging``.

Removed
~~~~~~~

* ``PreprocessingPipeline`` and ``BatchPreprocessingPipeline`` (use
  ``Pipeline`` / ``Pipeline.preset`` and ``meu.process``).
* The meegkit and pandas dependencies.

[0.1.0] - 2026-03-02
--------------------

Added
~~~~~

* Initial release of meeg-utils
* ``PreprocessingPipeline`` for single file preprocessing
* ``BatchPreprocessingPipeline`` for parallel batch processing
* Support for multiple input formats (string, Path, BIDSPath, Raw objects)
* Filtering and resampling
* Bad channel detection (PREP for EEG, Maxwell for MEG)
* Line noise removal (Zapline/Zapline-iter)
* ICA-based artifact removal (ICLabel for EEG, MEGnet for MEG)
* Comprehensive test suite (52 tests, >80% coverage)
* Full type hints
* NumPy-style docstrings
* GitHub Actions CI/CD
* Pre-commit hooks
* Sphinx documentation with PyData theme

Fixed
~~~~~

* None (initial release)

Changed
~~~~~~~

* None (initial release)

Deprecated
~~~~~~~~~~

* None (initial release)

Removed
~~~~~~~

* None (initial release)

Security
~~~~~~~~

* None (initial release)
