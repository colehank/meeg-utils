Installation
============

Requirements
------------

* Python 3.11 or higher
* pip or uv package manager

Install from PyPI
-----------------

Using pip (recommended):

.. code-block:: bash

   pip install meeg-utils

Using uv:

.. code-block:: bash

   uv pip install meeg-utils

Install from Source
-------------------

For development or latest features:

.. code-block:: bash

   # Clone repository
   git clone https://github.com/colehank/meeg-utils.git
   cd meeg-utils

   # Install with uv (recommended for development)
   uv sync --all-extras --dev

   # Or with pip
   pip install -e ".[dev]"

Dependencies
------------

Core dependencies are automatically installed:

* **mne** (>=1.11.0) - Core MEG/EEG processing
* **mne-bids** (>=0.18.0) - BIDS support
* **pyprep** (>=0.6.0) - Bad channel detection for EEG
* **meegkit** (>=0.1.9) - Line noise removal
* **mne-icalabel** (>=0.8.1) - Automatic ICA labeling
* **loguru** (>=0.7.3) - Logging

Optional Dependencies
---------------------

For development:

.. code-block:: bash

   pip install meeg-utils[dev]

This installs additional tools:

* pytest - Testing framework
* ruff - Linter and formatter
* mypy - Type checking
* pre-commit - Git hooks

Verify Installation
-------------------

.. code-block:: python

   import meeg_utils
   print(meeg_utils.__version__)

   from meeg_utils.preprocessing import PreprocessingPipeline
   print("Installation successful!")
