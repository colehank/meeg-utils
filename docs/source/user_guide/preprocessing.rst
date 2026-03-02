Preprocessing Guide
===================

This guide covers all preprocessing steps available in meeg-utils.

Filtering and Resampling
-------------------------

High-pass and low-pass filtering removes unwanted frequency components:

.. code-block:: python

   pipeline.filter_and_resample(
       highpass=0.1,   # Remove slow drifts below 0.1 Hz
       lowpass=100.0,  # Remove noise above 100 Hz
       sfreq=250.0     # Resample to 250 Hz
   )

**Guidelines:**

* **EEG**: highpass=0.5-1.0 Hz, lowpass=40-50 Hz
* **MEG**: highpass=0.1 Hz, lowpass=100-150 Hz
* Resampling reduces data size and computational cost

Bad Channel Detection
---------------------

Automatically detect and fix bad channels:

EEG (PREP Pipeline)
~~~~~~~~~~~~~~~~~~~

Uses the Preprocessing Pipeline (PREP) with multiple detection methods:

.. code-block:: python

   # Detect only
   pipeline.detect_and_fix_bad_channels(fix=False)
   print(f"Bad channels: {pipeline.raw.info['bads']}")

   # Detect and interpolate
   pipeline.detect_and_fix_bad_channels(fix=True)

**Methods used:**

* Correlation-based detection
* Deviation-based detection
* RANSAC-based detection

MEG (Maxwell Filtering)
~~~~~~~~~~~~~~~~~~~~~~~~

Uses Maxwell filtering for MEG data:

.. code-block:: python

   pipeline.detect_and_fix_bad_channels(fix=True)

**Note:** Requires properly calibrated MEG system with head position information.

Line Noise Removal
------------------

Remove power line noise (50 Hz or 60 Hz):

.. code-block:: python

   # For 50 Hz power line
   pipeline.remove_line_noise(fline=50.0)

   # For 60 Hz power line
   pipeline.remove_line_noise(fline=60.0)

**Methods:**

* **EEG**: Zapline-iter (iterative Zapline for multiple harmonics)
* **MEG**: Zapline (standard Zapline)

ICA-Based Artifact Removal
---------------------------

Independent Component Analysis (ICA) separates brain activity from artifacts:

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   # Automatic artifact detection and removal
   pipeline.apply_ica(
       n_components=20,
       method="infomax",
       regress=True
   )

**Parameters:**

* ``n_components``: Number of ICA components (default: 20 for EEG, 40 for MEG)
* ``method``: ICA algorithm ("infomax", "fastica", "picard")
* ``regress``: Whether to remove artifacts (True) or just fit ICA (False)

Automatic Labeling
~~~~~~~~~~~~~~~~~~~

Components are automatically labeled using:

* **EEG**: ICLabel (trained on EEG data)
* **MEG**: MEGnet (trained on MEG data)

Labeled categories:

* Brain activity
* Eye blinks
* Eye movements
* Heart beats
* Muscle artifacts
* Line noise
* Channel noise

Manual Labeling
~~~~~~~~~~~~~~~

Override automatic labels if needed:

.. code-block:: python

   # First run ICA without regression
   pipeline.apply_ica(n_components=5, regress=False)

   # Manually specify labels
   manual_labels = ["brain", "eye blink", "brain", "heart beat", "brain"]

   # Apply with manual labels
   pipeline.apply_ica(
       n_components=5,
       manual_labels=manual_labels,
       regress=True
   )

Complete Pipeline
-----------------

Run all steps in sequence:

.. code-block:: python

   from meeg_utils.preprocessing import PreprocessingPipeline

   pipeline = PreprocessingPipeline(
       input_path="data.fif",
       output_dir="output/"
   )

   # Run complete pipeline
   result = pipeline.run(
       filter_params={
           "highpass": 0.1,
           "lowpass": 100.0,
           "sfreq": 250.0
       },
       detect_bad_channels=True,
       remove_line_noise=True,
       apply_ica=True,
       ica_params={
           "n_components": 20,
           "method": "infomax"
       },
       save_intermediate=True  # Save intermediate results
   )

   # Save final result
   pipeline.save()

Customization Options
---------------------

Skip Steps
~~~~~~~~~~

.. code-block:: python

   # Skip ICA
   result = pipeline.run(
       detect_bad_channels=True,
       remove_line_noise=True,
       apply_ica=False
   )

Custom Order
~~~~~~~~~~~~

Run steps individually in custom order:

.. code-block:: python

   pipeline.load_data()
   pipeline.remove_line_noise()  # Remove line noise first
   pipeline.detect_and_fix_bad_channels()  # Then detect bad channels
   pipeline.filter_and_resample()  # Then filter
   pipeline.apply_ica()  # Finally ICA
   pipeline.save()

Save Intermediate Results
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   pipeline.run(save_intermediate=True)

   # Saves:
   # - bad_channels.tsv
   # - bad_channels.json
   # - ica_components.tsv
   # - preprocessing_info.json

Best Practices
--------------

1. **Always inspect your data visually** before and after preprocessing
2. **Use appropriate filter settings** for your experiment
3. **Check bad channel detection results** - manual inspection recommended
4. **Verify ICA components** - automatic labeling is not perfect
5. **Save intermediate results** for debugging and quality control
6. **Document your preprocessing steps** for reproducibility

Troubleshooting
---------------

Bad Channel Detection Fails
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If PREP fails:

.. code-block:: python

   # Try with fix=False to see what's detected
   pipeline.detect_and_fix_bad_channels(fix=False)
   print(pipeline.raw.info['bads'])

   # Manually mark bad channels if needed
   pipeline.raw.info['bads'] = ['Fp1', 'O2']

   # Then interpolate
   pipeline.detect_and_fix_bad_channels(fix=True)

ICA Produces Strange Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Check if data is properly filtered (1 Hz highpass recommended)
* Reduce n_components if you have few channels
* Try different ICA methods
* Manually inspect and label components

Line Noise Persists
~~~~~~~~~~~~~~~~~~~~

* Verify correct power line frequency (50 or 60 Hz)
* Check if line noise is present before removal
* Try increasing data length (Zapline works better with longer data)
