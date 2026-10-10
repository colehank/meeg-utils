Presets
=======

.. seealso::

   :ref:`tut-quickstart` runs the ``eeg-erp`` preset from raw data to epochs.


Presets are ready-made pipelines. They are ordinary
:class:`~meeg_utils.Pipeline` objects: change any parameter with
``set_params``, edit steps, or export them to YAML.

.. code-block:: python

   import meeg_utils as meu

   meu.presets.available()
   pipe = meu.Pipeline.preset("had-meeg", datatype="meg")

Recommended presets
-------------------

``eeg-erp``, ``eeg-rest``, ``meg-erp`` and ``meg-rest`` are the pipelines
this library recommends. Every parameter comes from a published default
(MNE-BIDS-Pipeline, the requirements of ICLabel and MEGnet, PREP,
ZapLine-plus, autoreject, Tanner et al. 2015 for the high-pass); parameters
without an accepted default must be given. The sources are listed step by
step in `docs/presets/recommended.md
<https://github.com/colehank/meeg-utils/blob/main/docs/presets/recommended.md>`_.

Each has two stages, applied per run:

.. code-block:: python

   pre = meu.Pipeline.preset("meg-erp", system="neuromag")   # or "ctf", "kit"
   clean = pre.fit_transform(run)

   ep = meu.Pipeline.preset("meg-erp", stage="epochs", event_id=["face", "house"],
                            head_destination=meu.io.average_dev_head_t(runs))
   epochs = ep.fit_transform(clean)
   # several runs: meu.epochs.combine([...])

- **preprocessing** -- EEG: bridged electrodes, 0.1 Hz high-pass,
  ZapLine-plus, PREP, interpolation, average reference, ICA with ICLabel.
  MEG: system-specific noise reduction (Neuromag: SSS; CTF: grade-3
  gradiometers; KIT: reference regression), 0.1 Hz high-pass, ZapLine-plus,
  250 Hz, 20-component ICA with MEGnet.
- **epochs** -- ERP: 40 Hz low-pass, -0.2 to 0.5 s around ``event_id``,
  pre-stimulus baseline. Rest: consecutive windows of ``epoch_duration``.
  Both end with local autoreject.

Dataset presets
---------------

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
