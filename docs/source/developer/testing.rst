Testing
=======

This document describes the testing strategy and guidelines for meeg-utils.

Test-Driven Development
-----------------------

All development follows strict TDD principles:

RED-GREEN-REFACTOR Cycle
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **RED**: Write a failing test

   .. code-block:: python

      def test_filter_with_valid_params():
          """Test filtering with valid parameters."""
          pipeline = PreprocessingPipeline(input_path=sample_eeg_raw)
          pipeline.raw = sample_eeg_raw

          filtered = pipeline.filter_and_resample(
              highpass=0.1,
              lowpass=100.0,
              sfreq=250.0,
          )

          assert filtered is not None
          assert filtered.info["sfreq"] == 250.0

2. **GREEN**: Write minimal code to pass

   .. code-block:: python

      def filter_and_resample(self, highpass, lowpass, sfreq):
          self.raw.filter(l_freq=highpass, h_freq=lowpass)
          self.raw.resample(sfreq)
          return self.raw

3. **REFACTOR**: Improve code while keeping tests green

Test Organization
-----------------

Directory Structure
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   tests/
   ├── conftest.py                    # Shared fixtures
   ├── test_preprocessing/
   │   ├── test_pipeline.py          # Pipeline tests
   │   └── test_batch.py             # Batch processing tests
   └── ...

Fixtures
~~~~~~~~

Common test data is provided via fixtures in ``conftest.py``:

.. code-block:: python

   @pytest.fixture
   def sample_eeg_raw() -> BaseRaw:
       \"\"\"Create sample EEG data for testing.\"\"\"
       # Creates realistic synthetic EEG data
       ...

   @pytest.fixture
   def sample_meg_raw() -> BaseRaw:
       \"\"\"Create sample MEG data for testing.\"\"\"
       # Creates realistic synthetic MEG data
       ...

   @pytest.fixture
   def mock_bids_path(tmp_path) -> BIDSPath:
       \"\"\"Create a mock BIDS dataset.\"\"\"
       # Creates temporary BIDS structure
       ...

Mock Strategy
~~~~~~~~~~~~~

To keep tests fast, we mock expensive operations:

.. code-block:: python

   @pytest.fixture
   def mock_iclabel(mocker):
       \"\"\"Mock ICLabel to avoid expensive computation.\"\"\"
       def mock_label_components(raw, ica, method="iclabel"):
           n_components = ica.n_components_
           return {
               "labels": ["brain"] * n_components,
               "y_pred_proba": [0.9] * n_components,
           }

       mocker.patch(
           "meeg_utils.preprocessing.ica.label_components",
           side_effect=mock_label_components,
       )
       return mock_label_components

**Mocked components:**

* ICA (ICLabel/MEGNet) - Avoid expensive ICA decomposition
* PREP (NoisyChannels) - Avoid RANSAC detection
* Zapline (DSS) - Avoid DSS iterations

**Philosophy:** Trust library implementations, test our wrapper logic.

Running Tests
-------------

Basic Usage
~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   uv run pytest

   # Run specific test file
   uv run pytest tests/test_preprocessing/test_pipeline.py

   # Run specific test
   uv run pytest tests/test_preprocessing/test_pipeline.py::TestPathInputParsing::test_accepts_plain_string_path

   # Run tests matching pattern
   uv run pytest -k "ica"

With Coverage
~~~~~~~~~~~~~

.. code-block:: bash

   # Generate coverage report
   uv run pytest --cov=src/meeg_utils --cov-report=html

   # Open report
   open htmlcov/index.html

Verbose Output
~~~~~~~~~~~~~~

.. code-block:: bash

   # Show detailed output
   uv run pytest -v

   # Show print statements
   uv run pytest -s

   # Show why tests were skipped
   uv run pytest -rs

Performance
~~~~~~~~~~~

Our optimized tests run in ~35 seconds (total of 52 tests):

.. code-block:: bash

   $ uv run pytest
   ============================= 52 passed in 35.18s ==============================

This is 83% faster than running without mocks (216 seconds).

Test Categories
---------------

Unit Tests
~~~~~~~~~~

Test individual functions in isolation:

.. code-block:: python

   class TestFilteringAndResampling:
       def test_filter_and_resample_with_valid_params(self, sample_eeg_raw):
           pipeline = PreprocessingPipeline(input_path=sample_eeg_raw)
           pipeline.raw = sample_eeg_raw

           filtered = pipeline.filter_and_resample(
               highpass=0.1,
               lowpass=100.0,
               sfreq=250.0,
           )

           assert filtered is not None
           assert filtered.info["sfreq"] == 250.0
           assert filtered.info["highpass"] == 0.1
           assert filtered.info["lowpass"] == 100.0

Integration Tests
~~~~~~~~~~~~~~~~~

Test multiple components working together:

.. code-block:: python

   class TestFullPipeline:
       def test_run_complete_pipeline_default_params(
           self, mock_bids_path, mock_iclabel, mock_zapline, mock_prep, mocker
       ):
           mocker.patch(
               "meeg_utils.preprocessing.bad_channels.NoisyChannels",
               return_value=mock_prep,
           )

           pipeline = PreprocessingPipeline(input_path=mock_bids_path)
           result = pipeline.run(ica_params={"n_components": 2})

           assert result is not None
           assert isinstance(result, mne.io.BaseRaw)

Batch Tests
~~~~~~~~~~~

Test parallel processing:

.. code-block:: python

   class TestBatchProcessing:
       def test_run_batch_parallel_processing(
           self, multiple_bids_paths, temp_output_dir, mocks...
       ):
           batch = BatchPreprocessingPipeline(
               input_paths=multiple_bids_paths,
               output_dir=temp_output_dir,
               n_jobs=2,
           )

           batch.run(ica_params={"n_components": 2})

           output_files = list(temp_output_dir.glob("**/*_preproc_*.fif"))
           assert len(output_files) == 3

Writing Good Tests
------------------

Test Names
~~~~~~~~~~

Use descriptive names following the pattern: ``test_<what>_<condition>``

Good:

.. code-block:: python

   def test_filter_validates_nyquist_frequency(self, sample_eeg_raw):
       ...

   def test_interpolate_bad_channels(self, sample_eeg_raw):
       ...

Bad:

.. code-block:: python

   def test_filter1(self):
       ...

   def test_channels(self):
       ...

Assertions
~~~~~~~~~~

Be specific in assertions:

Good:

.. code-block:: python

   assert result.info["sfreq"] == 250.0
   assert "Fp1" in result.info["bads"]
   assert len(output_files) == 3

Bad:

.. code-block:: python

   assert result
   assert result.info
   assert output_files

Error Testing
~~~~~~~~~~~~~

Test error conditions:

.. code-block:: python

   def test_filter_validates_highpass_lowpass_order(self, sample_eeg_raw):
       pipeline = PreprocessingPipeline(input_path=sample_eeg_raw)
       pipeline.raw = sample_eeg_raw

       with pytest.raises((ValueError, AssertionError)):
           pipeline.filter_and_resample(
               highpass=50.0,
               lowpass=10.0,  # lowpass < highpass: invalid
               sfreq=250.0,
           )

Coverage Goals
--------------

Target: >80% code coverage

Check coverage:

.. code-block:: bash

   uv run pytest --cov=src/meeg_utils --cov-report=term

Current coverage:

* Overall: TBD
* preprocessing.pipeline: TBD
* preprocessing.batch: TBD

Continuous Integration
----------------------

All tests run automatically on GitHub Actions:

* Python 3.11, 3.12
* Ubuntu, macOS, Windows
* Coverage reported to Codecov

See :doc:`ci_cd` for details.

Troubleshooting
---------------

Tests Fail Locally
~~~~~~~~~~~~~~~~~~

1. Ensure dependencies are up to date:

   .. code-block:: bash

      uv sync --dev

2. Clear pytest cache:

   .. code-block:: bash

      rm -rf .pytest_cache
      uv run pytest

3. Run with verbose output:

   .. code-block:: bash

      uv run pytest -vv

Tests Timeout
~~~~~~~~~~~~~

If tests are slow:

1. Check if mocks are being used
2. Reduce test data size
3. Run specific test file instead of all

Import Errors
~~~~~~~~~~~~~

.. code-block:: bash

   # Reinstall package in editable mode
   uv pip install -e .
