Testing
=======

Principles
----------

* **No mocks of the scientific code.** Steps run the real MNE, PyPREP,
  mne-denoise and mne-icalabel code, so a dependency update that changes
  results is caught.
* **Ground truth where possible.** Synthetic data are built so the right
  answer is known: line noise added to a line-free signal, a noisy and a
  flat channel at known positions, dipoles projected at known head
  positions. Tests compare against that truth, not against a previous
  output.
* **Contracts for every step.** Every step passes
  :func:`meeg_utils.testing.check_step`: parameters survive ``clone``,
  ``set_params`` and a YAML round trip; ``transform`` before ``fit`` raises;
  the input is not modified; channels and the time axis are unchanged unless
  declared; every declared figure can be drawn.

Layout
------

.. code-block:: text

   tests/
   ├── conftest.py               # small_raw: tiny EEG with first_samp, meas_date, annotations
   ├── test_batch.py             # meu.process
   ├── test_logging.py
   ├── test_core/                # Step and Pipeline (toy steps in _steps.py)
   ├── test_io/                  # reading, system detection, derivatives
   └── test_steps/
       ├── conftest.py           # make_eeg() simulation, Neuromag test recording
       ├── test_steps.py         # each step: contract + scientific checks
       ├── test_head.py          # HeadAlign against forward-simulated data
       └── test_presets.py       # preset configuration and end-to-end runs

Fixtures
--------

``make_eeg(line=True, blinks=True, bad=False, seed=0)`` (``tests/test_steps/conftest.py``)
   60 s of 32-channel EEG at 250 Hz: spatially smooth 1/f background,
   posterior alpha, frontal blinks with an EOG channel, 50 Hz line noise,
   and optionally a noisy (C4) and a flat (P8) channel.

``neuromag``
   10 s of MNE's Neuromag test recording (306 MEG + 60 EEG), downloaded once
   with a pinned URL and SHA-256 hash into the pooch cache; tests that need
   it are skipped when offline.

Testing a custom step
---------------------

.. code-block:: python

   from meeg_utils.testing import check_step

   def test_my_step_contract(eeg):
       check_step(MyStep(param=1), eeg)

Running
-------

.. code-block:: bash

   uv run pytest                                  # everything (a few minutes)
   uv run pytest tests/test_core tests/test_io    # fast subset
   uv run pytest tests/test_steps/test_head.py -k recovers
   uv run pytest --cov=meeg_utils --cov-report=term-missing

Code quality checks run in CI as well:

.. code-block:: bash

   uv run ruff check src tests
   uv run ruff format --check src tests
   uv run mypy src
   uv run interrogate src
