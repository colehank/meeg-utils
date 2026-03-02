Quick Start Guide
=================

This guide will help you get started with meeg-utils for preprocessing MEG/EEG data.

Single File Processing
----------------------

Basic Pipeline
~~~~~~~~~~~~~~

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

   # Create pipeline
   pipeline = PreprocessingPipeline(
       input_path=bids_path,
       output_dir="/path/to/output"
   )

   # Run with default settings
   raw_preprocessed = pipeline.run()

   # Save results
   pipeline.save()

Custom Parameters
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Run with custom filtering
   raw_preprocessed = pipeline.run(
       filter_params={
           "highpass": 0.1,
           "lowpass": 100.0,
           "sfreq": 250.0
       },
       detect_bad_channels=True,
       remove_line_noise=True,
       apply_ica=True,
       ica_params={"n_components": 20}
   )

Step-by-Step Processing
~~~~~~~~~~~~~~~~~~~~~~~~

For more control, run individual steps:

.. code-block:: python

   from meeg_utils.preprocessing import PreprocessingPipeline

   pipeline = PreprocessingPipeline(input_path="path/to/data.fif")

   # Load data
   raw = pipeline.load_data()

   # Apply filtering
   raw_filtered = pipeline.filter_and_resample(
       highpass=1.0,
       lowpass=40.0,
       sfreq=200.0
   )

   # Detect and fix bad channels
   raw_clean = pipeline.detect_and_fix_bad_channels(fix=True)

   # Remove line noise
   raw_denoised = pipeline.remove_line_noise(fline=50.0)

   # Apply ICA
   raw_ica = pipeline.apply_ica(n_components=20, regress=True)

   # Save
   pipeline.save()

Batch Processing
----------------

Process multiple files in parallel:

.. code-block:: python

   from meeg_utils.preprocessing import BatchPreprocessingPipeline
   from mne_bids import BIDSPath

   # Define multiple inputs
   bids_paths = [
       BIDSPath(
           subject=f"{i:02d}",
           session="01",
           task="rest",
           datatype="eeg",
           root="/path/to/bids/dataset"
       )
       for i in range(1, 11)  # Process 10 subjects
   ]

   # Create batch pipeline
   batch = BatchPreprocessingPipeline(
       input_paths=bids_paths,
       output_dir="/path/to/output",
       n_jobs=4  # Use 4 parallel workers
   )

   # Run batch preprocessing
   batch.run(
       filter_params={"highpass": 0.1, "lowpass": 100.0},
       detect_bad_channels=True,
       remove_line_noise=True,
       apply_ica=True,
       save_logs=True
   )

Common Use Cases
----------------

EEG Preprocessing
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from meeg_utils.preprocessing import PreprocessingPipeline

   pipeline = PreprocessingPipeline(
       input_path="eeg_data.fif",
       output_dir="output/"
   )

   # Standard EEG preprocessing
   result = pipeline.run(
       filter_params={
           "highpass": 0.5,  # Remove slow drifts
           "lowpass": 50.0,   # Remove high-frequency noise
           "sfreq": 250.0     # Downsample to 250 Hz
       },
       detect_bad_channels=True,  # PREP pipeline
       remove_line_noise=True,     # Zapline-iter for 50/60 Hz
       apply_ica=True,             # ICLabel for artifacts
       ica_params={"n_components": 20}
   )

   pipeline.save()

MEG Preprocessing
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from meeg_utils.preprocessing import PreprocessingPipeline

   pipeline = PreprocessingPipeline(
       input_path="meg_data.fif",
       output_dir="output/"
   )

   # Standard MEG preprocessing
   result = pipeline.run(
       filter_params={
           "highpass": 0.1,
           "lowpass": 100.0,
           "sfreq": 500.0
       },
       detect_bad_channels=True,  # Maxwell filtering
       remove_line_noise=True,     # Zapline for MEG
       apply_ica=True,             # MEGnet for artifacts
       ica_params={"n_components": 40}
   )

   pipeline.save()

Next Steps
----------

* Learn about :doc:`preprocessing` options in detail
* Explore :doc:`batch_processing` for large datasets
* Check the :doc:`../api/preprocessing` for all available methods
