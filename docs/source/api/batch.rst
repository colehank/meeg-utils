BatchPreprocessingPipeline
===========================

Batch preprocessing pipeline for processing multiple files in parallel.

.. currentmodule:: meeg_utils.preprocessing

.. autoclass:: BatchPreprocessingPipeline
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

Example Usage
-------------

.. code-block:: python

   from meeg_utils.preprocessing import BatchPreprocessingPipeline
   from mne_bids import BIDSPath

   # Define inputs
   bids_paths = [
       BIDSPath(subject=f"{i:02d}", session="01", task="rest",
                datatype="eeg", root="/data/bids")
       for i in range(1, 11)
   ]

   # Create batch pipeline
   batch = BatchPreprocessingPipeline(
       input_paths=bids_paths,
       output_dir="/data/output",
       n_jobs=4,
       random_state=42
   )

   # Run batch processing
   batch.run(
       filter_params={"highpass": 0.1, "lowpass": 100.0},
       detect_bad_channels=True,
       remove_line_noise=True,
       apply_ica=True,
       save_logs=True
   )
