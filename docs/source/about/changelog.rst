Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

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
