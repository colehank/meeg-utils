PreprocessingPipeline
=====================

Main preprocessing pipeline for single file processing.

.. currentmodule:: meeg_utils.preprocessing

.. autoclass:: PreprocessingPipeline
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. automethod:: __init__

Example Usage
-------------

.. code-block:: python

   from meeg_utils.preprocessing import PreprocessingPipeline
   from mne_bids import BIDSPath

   # Create pipeline
   pipeline = PreprocessingPipeline(
       input_path=BIDSPath(
           subject="01",
           session="01",
           task="rest",
           datatype="eeg",
           root="/data/bids"
       ),
       output_dir="/data/output",
       n_jobs=1,
       use_cuda=False,
       random_state=42
   )

   # Run preprocessing
   result = pipeline.run(
       filter_params={"highpass": 0.1, "lowpass": 100.0, "sfreq": 250.0},
       detect_bad_channels=True,
       remove_line_noise=True,
       apply_ica=True
   )

   # Save results
   pipeline.save()
