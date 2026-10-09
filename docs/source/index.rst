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

MEG and EEG quality control, preprocessing and analysis on top of
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
* **Quality control**: QC metrics and figures for every step.
* **BIDS first**: sidecars are read, results are written as BIDS
  derivatives with full provenance.
* **Presets** from published pipelines, with every deviation documented.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/quickstart
   user_guide/steps
   user_guide/presets
   user_guide/batch_processing

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
