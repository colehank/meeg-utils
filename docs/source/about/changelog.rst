Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

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
