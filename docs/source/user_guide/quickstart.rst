Quickstart
==========

.. code-block:: python

   import meeg_utils as meu
   from meeg_utils import steps as S

Read a recording
----------------

:func:`meeg_utils.io.read` reads any format MNE supports. BIDS recordings
(a :class:`~mne_bids.BIDSPath`, or a plain path inside a BIDS dataset) are
read with their sidecars, so bad channels, channel types and the power-line
frequency come from ``channels.tsv`` and ``*_eeg.json`` / ``*_meg.json``.

.. code-block:: python

   raw = meu.io.read("bids/sub-01/eeg/sub-01_task-rest_eeg.vhdr")
   meu.io.detect_system(raw)   # "eeg", "neuromag", "ctf", "kit", ...

Build a pipeline
----------------

A pipeline is a list of named steps. Parameters are set when a step is
created and can be changed later with ``set_params``:

.. code-block:: python

   pipe = meu.Pipeline([
       ("filter", S.Filter(0.1, 100.0)),
       ("resample", S.Resample(250.0)),
       ("bads", S.BadChannels(method="prep")),
       ("interpolate", S.Interpolate()),
       ("line_noise", S.LineNoise()),
       ("reference", S.Reference("average")),
       ("ica", S.ICA(n_components=20, labeler="iclabel")),
   ])
   pipe.set_params(ica__threshold=0.9, filter__h_freq=40.0)

Or start from a preset (see :doc:`presets`):

.. code-block:: python

   pipe = meu.Pipeline.preset("had-meeg", datatype="eeg")

Process
-------

``fit`` learns what each step needs from the data (bad channels, ICA
decomposition, ...); ``transform`` applies it; ``fit_transform`` does both.
The input is never modified.

.. code-block:: python

   clean = pipe.fit_transform(raw)

Check the result
----------------

Every fitted step has ``qc_`` metrics, collected in ``pipe.qc_``, and can
draw figures:

.. code-block:: python

   pipe.qc_["bads"]          # detected channels, per criterion
   pipe.qc_["line_noise"]    # suppression, distortion, over-/under-cleaning
   pipe.qc_["ica"]           # labels, probabilities, excluded components

   pipe["ica"].plot_kinds    # {"components": False, "labels": False, "properties": True, ...}
   figs = pipe.plot(inst=raw)        # {step name: {kind: Figure}}
   figs["line_noise"]["psd"].savefig("line_noise.png")

Correct the ICA labels after looking at the components, then transform again:

.. code-block:: python

   pipe["ica"].relabel({3: "eye blink", 7: "brain"})
   clean = pipe.transform(raw)

Save
----

.. code-block:: python

   meu.io.save_derivative(clean, "bids/sub-01/eeg/sub-01_task-rest_eeg.vhdr",
                          "bids/derivatives/meu", pipeline=pipe)

This writes ``sub-01_task-rest_desc-preproc_eeg.fif`` in BIDS derivative
layout, with a JSON sidecar holding the pipeline configuration, software
versions, the duration and warnings of every step, and all QC metrics. The
configuration alone can be shared as YAML:

.. code-block:: python

   pipe.to_yaml("pipeline.yaml")
   pipe = meu.Pipeline.from_yaml("pipeline.yaml")

Logging
-------

meeg-utils is silent by default and never configures logging on import.
Show its messages with :func:`meeg_utils.setup_logging`:

.. code-block:: python

   meu.setup_logging("INFO", log_file="logs/preproc.log")
