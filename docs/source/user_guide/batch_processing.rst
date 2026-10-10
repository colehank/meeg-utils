Batch processing
================

.. seealso::

   :ref:`tut-datasets` processes a small BIDS dataset end to end.


:func:`meeg_utils.process` runs one pipeline over many recordings. Each
recording gets its own clone of the pipeline, fitted on that recording only,
and the result is saved with :func:`meeg_utils.io.save_derivative`.

.. code-block:: python

   import meeg_utils as meu
   from mne_bids import find_matching_paths

   pipe = meu.Pipeline.preset("had-meeg", datatype="eeg")
   recordings = find_matching_paths("bids", datatypes="eeg", extensions=".vhdr")

   records = meu.process(pipe, recordings, "bids/derivatives/meu", n_jobs=8)

``sources`` can mix :class:`~mne_bids.BIDSPath` objects and plain paths;
paths inside a BIDS dataset are named like BIDS paths
(``sub-01/eeg/sub-01_task-rest_desc-preproc_eeg.fif``), other files are
written to the root as ``<name>_desc-preproc_<datatype>.fif``.

Whole datasets
--------------

:class:`meeg_utils.Dataset` selects the recordings of a BIDS dataset and
groups the runs of each session and task; :func:`meeg_utils.process_dataset`
takes them through every stage:

.. code-block:: python

   ds = meu.Dataset("bids", datatype="meg", tasks=["faces"], exclude=["sub-07"])
   result = meu.process_dataset(
       ds, "bids/derivatives/meu",
       preprocessing=meu.Pipeline.preset("meg-erp", system="neuromag"),
       epochs=meu.Pipeline.preset("meg-erp", stage="epochs", event_id=["face", "house"]),
       n_jobs=8,
   )
   result.to_csv("bids/derivatives/meu/batch.csv")
   result.failed          # one record per failed run or session

1. every run is preprocessed with its own copy of the pipeline (``desc-preproc``);
2. per session and task, MEG runs are mapped to their average head
   position (``align="average"``), and each run is epoched (``desc-epochs``,
   with the run entity);
3. the runs are combined (:func:`meeg_utils.epochs.combine`) into one file
   without the run entity, e.g. ``sub-01_task-faces_desc-epochs_epo.fif``,
   whose sidecar lists the runs.

Outputs that exist are skipped by default (``skip_existing=True``), so
calling it again resumes an interrupted batch and retries what failed.
From the command line: ``meu run --preset meg-erp --epochs --option
"event_id=[face, house]" --sources bids --out bids/derivatives/meu``.

Errors
------

A failing recording never stops the others. Once all are done,
``on_error="raise"`` (default) raises a :class:`RuntimeError` listing every
failure, and ``on_error="warn"`` emits a warning instead. Either way the
function returns one record per recording:

.. code-block:: python

   records = meu.process(pipe, recordings, root, on_error="warn")
   failed = [r for r in records if r["status"] == "failed"]
   # {"source", "status": "ok" | "skipped" | "failed", "output", "error", "duration_s"}

Existing outputs
----------------

By default an existing output is an error. ``skip_existing=True`` skips
recordings that already have an output (to resume an interrupted run);
``overwrite=True`` replaces them. :func:`meeg_utils.io.existing_derivatives`
lists the outputs of one recording.

Logging
-------

.. code-block:: python

   meu.setup_logging("INFO", log_file="bids/derivatives/meu/logs/process.log")

Failures are logged with their full traceback.
