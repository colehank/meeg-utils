meeg-utils Documentation
========================

.. image:: https://img.shields.io/badge/python-3.11+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.11+

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/code%20style-ruff-000000.svg
   :target: https://github.com/astral-sh/ruff
   :alt: Code style: ruff

MEG and EEG quality control, preprocessing and epoching on top of
`MNE-Python <https://mne.tools>`_. Every step is a small, scikit-learn style
estimator with explicit parameters; steps compose into pipelines that record
what they did, measure how well it worked, and draw figures to check it.

.. code-block:: python

   import meeg_utils as meu
   from meeg_utils import steps as S

   pipe = meu.Pipeline([
       ("filter", S.Filter(0.1, 100.0)),
       ("bads", S.BadChannels()),
       ("interpolate", S.Interpolate()),
       ("line_noise", S.LineNoise()),
       ("reference", S.Reference("average")),
       ("ica", S.ICA(labeler="iclabel")),
   ])
   clean = pipe.fit_transform("bids/sub-01/eeg/sub-01_task-rest_eeg.vhdr")
   pipe.qc_

Highlights
----------

* **scikit-learn conventions**: ``fit`` / ``transform``, ``set_params``,
  ``clone``, YAML configurations.
* **Data integrity**: no silent channel drops or time shifts; contract
  checks for every step.
* **Acquisition QC** (:doc:`user_guide/qc`): how well a recording was made
  (bridged electrodes, flat channels, head movement, SQUID jumps, empty
  room, BIDS metadata, ...), with dataset-level outlier detection.
* **Processing QC**: metrics and figures for every step.
* **Presets** (:doc:`user_guide/presets`): recommended EEG and MEG
  pipelines with every parameter traced to a published default, and the
  HAD-MEEG pipeline with its known issues fixed.
* **BIDS first**: sidecars are read, whole datasets are processed run by run
  and combined per session, results are written as BIDS derivatives with
  full provenance.
* **Command line**: ``meu qc``, ``meu run``, ``meu preset``, ``meu report``
  (:doc:`user_guide/cli`).

Analysis (ERP/ERF measures, spectra, time-frequency, group statistics) is
planned; the outputs are plain MNE objects, so MNE's own analysis tools
apply directly.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/quickstart
   user_guide/steps
   user_guide/qc
   user_guide/presets
   user_guide/batch_processing
   user_guide/cli

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer/contributing
   developer/testing
   developer/ci_cd
   developer/release

.. toctree::
   :maxdepth: 1
   :caption: About

   about/changelog
   about/license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
