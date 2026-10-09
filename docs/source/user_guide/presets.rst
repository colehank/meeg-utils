Presets
=======

Presets are ready-made pipelines. They are ordinary
:class:`~meeg_utils.Pipeline` objects: change any parameter with
``set_params``, edit steps, or export them to YAML.

.. code-block:: python

   import meeg_utils as meu

   meu.presets.available()
   pipe = meu.Pipeline.preset("had-meeg", datatype="meg")

Dataset presets reproduce the preprocessing of a published dataset with its
known problems fixed. Every difference from the original code is documented
with its reason.

``had-meeg``
------------

The per-run preprocessing of `HAD-MEEG <https://github.com/colehank/HAD-MEEG>`_
for CTF MEG (``datatype="meg"``) or EEG (``datatype="eeg"``): 0.1-100 Hz,
250 Hz, bad channels (Maxwell / PREP), interpolation, ZapLine-plus, CTF
grade-3 compensation or average reference, and ICA labelled with MEGnet /
ICLabel.

The list of corrections, including those for the original epoching stage
(head alignment, baseline, mastoid reference), is in
`docs/presets/had-meeg.md <https://github.com/colehank/meeg-utils/blob/main/docs/presets/had-meeg.md>`_.
