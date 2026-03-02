Batch Processing
================

Process multiple datasets efficiently using parallel processing.

Basic Usage
-----------

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
       for i in range(1, 21)  # Process 20 subjects
   ]

   # Create batch pipeline
   batch = BatchPreprocessingPipeline(
       input_paths=bids_paths,
       output_dir="/path/to/output",
       n_jobs=4  # Use 4 parallel workers
   )

   # Run batch processing
   batch.run()

Parallel Processing
-------------------

Control the number of parallel jobs:

.. code-block:: python

   # Sequential processing (n_jobs=1)
   batch = BatchPreprocessingPipeline(
       input_paths=bids_paths,
       output_dir="/path/to/output",
       n_jobs=1
   )

   # Parallel processing
   batch = BatchPreprocessingPipeline(
       input_paths=bids_paths,
       output_dir="/path/to/output",
       n_jobs=4  # 4 parallel workers
   )

   # Use all available CPUs
   import multiprocessing
   batch = BatchPreprocessingPipeline(
       input_paths=bids_paths,
       output_dir="/path/to/output",
       n_jobs=multiprocessing.cpu_count()
   )

**Note:** Each worker uses 1 CPU core. Don't set n_jobs higher than available cores.

Custom Parameters
-----------------

Apply same preprocessing to all files:

.. code-block:: python

   batch.run(
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

Skip Existing Files
-------------------

Resume interrupted batch processing:

.. code-block:: python

   batch.run(skip_existing=True)

This skips files that have already been processed, allowing you to:

* Resume after interruption
* Add new files to existing batch
* Re-process only failed files

Logging
-------

Enable logging to file:

.. code-block:: python

   batch.run(
       save_logs=True,
       logging_level="INFO"  # or "DEBUG", "WARNING", "ERROR"
   )

This creates a log file: ``output_dir/batch_preprocessing.log``

Control logging verbosity:

* ``"DEBUG"``: Detailed information
* ``"INFO"``: General progress (default)
* ``"WARNING"``: Only warnings and errors
* ``"ERROR"``: Only errors

Input Path Types
-----------------

Multiple path types are supported:

BIDS Paths
~~~~~~~~~~

.. code-block:: python

   from mne_bids import BIDSPath

   bids_paths = [
       BIDSPath(subject="01", session="01", task="rest",
                datatype="eeg", root="/data/bids"),
       BIDSPath(subject="02", session="01", task="rest",
                datatype="eeg", root="/data/bids"),
   ]

String Paths
~~~~~~~~~~~~

.. code-block:: python

   string_paths = [
       "/data/subject01_eeg.fif",
       "/data/subject02_eeg.fif",
   ]

Path Objects
~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path

   path_objects = [
       Path("/data/subject01_eeg.fif"),
       Path("/data/subject02_eeg.fif"),
   ]

Mixed Types
~~~~~~~~~~~

.. code-block:: python

   mixed_paths = [
       bids_paths[0],           # BIDSPath
       "/data/subject02.fif",   # string
       Path("/data/subject03.fif"),  # Path
   ]

   batch = BatchPreprocessingPipeline(input_paths=mixed_paths)

Error Handling
--------------

Batch processing continues even if individual files fail:

.. code-block:: python

   # Include an invalid path
   paths_with_invalid = bids_paths + [
       BIDSPath(subject="99", session="99", task="nonexistent",
                datatype="eeg", root="/nonexistent")
   ]

   batch = BatchPreprocessingPipeline(
       input_paths=paths_with_invalid,
       output_dir="/path/to/output"
   )

   # Processing continues for valid files
   batch.run()

Check logs to see which files failed and why.

Output Organization
-------------------

BIDS Structure
~~~~~~~~~~~~~~

Output preserves BIDS directory structure:

.. code-block:: text

   output_dir/
   ├── sub-01/
   │   └── ses-01/
   │       └── eeg/
   │           ├── sub-01_ses-01_task-rest_preproc_eeg.fif
   │           └── sub-01_ses-01_task-rest_bad_channels.tsv
   ├── sub-02/
   │   └── ses-01/
   │       └── eeg/
   │           ├── sub-02_ses-01_task-rest_preproc_eeg.fif
   │           └── sub-02_ses-01_task-rest_bad_channels.tsv
   └── batch_preprocessing.log

Custom Output
~~~~~~~~~~~~~

For non-BIDS paths:

.. code-block:: text

   output_dir/
   ├── subject01_preproc_eeg.fif
   ├── subject02_preproc_eeg.fif
   └── batch_preprocessing.log

Performance Tips
----------------

Optimal n_jobs
~~~~~~~~~~~~~~

.. code-block:: python

   import multiprocessing

   # Leave 1 core free for system
   n_jobs = max(1, multiprocessing.cpu_count() - 1)

   batch = BatchPreprocessingPipeline(
       input_paths=bids_paths,
       output_dir="/path/to/output",
       n_jobs=n_jobs
   )

Memory Considerations
~~~~~~~~~~~~~~~~~~~~~

Each worker loads one dataset into memory. If datasets are large:

.. code-block:: python

   # Reduce n_jobs to avoid memory issues
   batch = BatchPreprocessingPipeline(
       input_paths=bids_paths,
       output_dir="/path/to/output",
       n_jobs=2  # Use fewer workers for large datasets
   )

Progress Monitoring
~~~~~~~~~~~~~~~~~~~

Monitor progress in real-time:

.. code-block:: python

   batch.run(
       save_logs=True,
       logging_level="INFO"
   )

   # Watch log file in another terminal
   # tail -f output_dir/batch_preprocessing.log

Example: Large Study
--------------------

Complete example for processing a large study:

.. code-block:: python

   from meeg_utils.preprocessing import BatchPreprocessingPipeline
   from mne_bids import BIDSPath
   import multiprocessing

   # Define study parameters
   bids_root = "/data/my_study/bids"
   output_dir = "/data/my_study/derivatives/meeg-utils"
   subjects = [f"{i:02d}" for i in range(1, 101)]  # 100 subjects

   # Create BIDSPaths for all subjects
   bids_paths = [
       BIDSPath(
           subject=sub,
           session="01",
           task="task",
           datatype="eeg",
           root=bids_root
       )
       for sub in subjects
   ]

   # Configure batch processing
   n_jobs = max(1, multiprocessing.cpu_count() - 1)

   batch = BatchPreprocessingPipeline(
       input_paths=bids_paths,
       output_dir=output_dir,
       n_jobs=n_jobs,
       random_state=42  # For reproducibility
   )

   # Run with standard parameters
   batch.run(
       filter_params={
           "highpass": 0.5,
           "lowpass": 50.0,
           "sfreq": 250.0
       },
       detect_bad_channels=True,
       remove_line_noise=True,
       apply_ica=True,
       ica_params={"n_components": 20},
       save_intermediate=True,
       skip_existing=True,  # Resume if interrupted
       save_logs=True,
       logging_level="INFO"
   )

   print("Batch processing complete!")
   print(f"Results saved to: {output_dir}")
   print(f"Log file: {output_dir}/batch_preprocessing.log")
