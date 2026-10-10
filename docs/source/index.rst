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
`MNE-Python <https://mne.tools>`_: scikit-learn style steps with documented
defaults, QC metrics and figures for every step, BIDS in and out.

.. image:: _static/pipeline.png
   :alt: From raw recordings to preprocessed runs and combined epochs
   :width: 100%

.. code-block:: python

   import meeg_utils as meu

   raw = meu.io.read("bids/sub-01/eeg/sub-01_task-oddball_eeg.vhdr")
   meu.qc.inspect(raw)                     # how well was it recorded?
   pipe = meu.Pipeline.preset("eeg-erp")   # every parameter has a cited source
   clean = pipe.fit_transform(raw)
   pipe.plot(inst=raw)                     # figures of every step

New here? Start with the :ref:`tutorials`: they run on real data and show
what every step does.

- :ref:`tutorials`: :ref:`tut-quickstart`, :ref:`tut-qc`,
  :ref:`tut-pipelines`, :ref:`tut-epochs`, :ref:`tut-datasets`
- User guide: :doc:`user_guide/installation`, :doc:`user_guide/steps`,
  :doc:`user_guide/qc`, :doc:`user_guide/presets`,
  :doc:`user_guide/batch_processing`, :doc:`user_guide/cli`
- :doc:`api/index`: every function and parameter
- :doc:`about/changelog`

Analysis (ERP/ERF measures, spectra, time-frequency, group statistics) is
planned; the outputs are plain MNE objects, so MNE's own analysis tools
apply directly.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorials

   auto_tutorials/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   user_guide/installation
   user_guide/steps
   user_guide/qc
   user_guide/presets
   user_guide/batch_processing
   user_guide/cli

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Developer Guide

   developer/contributing
   developer/testing
   developer/ci_cd
   developer/release

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: About

   about/changelog
   about/license
