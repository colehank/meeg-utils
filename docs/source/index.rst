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

A Python-based MEG/EEG processing toolkit built on MNE-Python, providing a high-level,
user-friendly API for preprocessing electrophysiological data.

Features
--------

* **High-level API** - Simple, intuitive interface for complex preprocessing pipelines
* **BIDS Support** - Native support for Brain Imaging Data Structure (BIDS)
* **Advanced Preprocessing**:

  * Automated bad channel detection (PREP pipeline for EEG, Maxwell filtering for MEG)
  * Line noise removal (Zapline/Zapline-iter)
  * ICA-based artifact removal with automatic labeling (ICLabel for EEG, MEGNet for MEG)

* **Batch Processing** - Parallel processing of multiple datasets
* **Type-safe** - Full type hints for better IDE support
* **Well-tested** - Comprehensive test suite with >80% coverage

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install meeg-utils

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from meeg_utils.preprocessing import PreprocessingPipeline
   from mne_bids import BIDSPath

   # Define input
   bids_path = BIDSPath(
       subject="01",
       session="01",
       task="rest",
       datatype="eeg",
       root="/path/to/bids/dataset"
   )

   # Create and run pipeline
   pipeline = PreprocessingPipeline(
       input_path=bids_path,
       output_dir="/path/to/output"
   )

   raw_preprocessed = pipeline.run(
       filter_params={"highpass": 0.1, "lowpass": 100.0, "sfreq": 250.0},
       detect_bad_channels=True,
       remove_line_noise=True,
       apply_ica=True,
   )

   # Save results
   pipeline.save()

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/quickstart
   user_guide/preprocessing
   user_guide/batch_processing

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/preprocessing
   api/pipeline
   api/batch

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
