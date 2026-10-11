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
   uv sync

Dependencies
------------

Installed automatically:

* **mne** (>=1.13.2) - MEG/EEG data structures and processing
* **mne-bids** (>=0.20) - BIDS reading and writing
* **mne-denoise** (>=0.0.3) - ZapLine and ZapLine-plus line-noise removal
* **pyprep** (>=0.9) - PREP bad-channel detection for EEG
* **mne-icalabel** (>=0.10) and **onnxruntime** - ICLabel / MEGnet component labels
* **autoreject** (>=0.5.1) - epoch rejection and repair
* **scikit-learn** (>=1.5) - estimator conventions (``clone``, parameters)
* **pyyaml**, **joblib**, **loguru**

Verify Installation
-------------------

.. code-block:: python

   import meeg_utils as meu
   print(meu.__version__)
   print(meu.presets.available())

.. code-block:: bash

   meu --help
