# meeg-utils

[![CI](https://github.com/colehank/meeg-utils/workflows/CI/badge.svg)](https://github.com/colehank/meeg-utils/actions)
[![Documentation](https://github.com/colehank/meeg-utils/workflows/Documentation/badge.svg)](https://colehank.github.io/meeg-utils/)
[![PyPI version](https://badge.fury.io/py/meeg-utils.svg)](https://badge.fury.io/py/meeg-utils)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/gh/colehank/meeg-utils/branch/main/graph/badge.svg)](https://codecov.io/gh/colehank/meeg-utils)

MEG and EEG quality control, preprocessing and analysis on top of
[MNE-Python](https://mne.tools). Every step is a small, scikit-learn style
estimator with explicit parameters; steps compose into pipelines that record
what they did, measure how well it worked, and draw figures to check it.

```python
import meeg_utils as meu
from meeg_utils import steps as S

pipe = meu.Pipeline([
    ("filter", S.Filter(0.1, 100.0)),
    ("bads", S.BadChannels()),          # PREP for EEG, Maxwell for Neuromag/CTF MEG
    ("interpolate", S.Interpolate()),
    ("line_noise", S.LineNoise()),      # ZapLine-plus
    ("reference", S.Reference("average")),
    ("ica", S.ICA(labeler="iclabel")),  # removes components with artifact probability >= 0.8
])

source = "bids/sub-01/eeg/sub-01_task-rest_eeg.vhdr"
raw = meu.io.read(source)       # BIDS sidecars (bad channels, line frequency, ...) are used
clean = pipe.fit_transform(raw)
pipe.qc_                        # quality metrics of every step
figs = pipe.plot(inst=raw)      # {step: {figure kind: matplotlib Figure}}
meu.io.save_derivative(clean, source, "bids/derivatives/meu", pipeline=pipe)
# -> data + JSON sidecar with the configuration, provenance and QC metrics
```

## Features

- **Steps** (`meu.steps`): `Filter`, `Resample`, `LineNoise` (ZapLine,
  ZapLine-plus, notch), `BadChannels` (PREP, Maxwell), `Interpolate`,
  `Reference`, `ICA` (ICLabel / MEGnet labelling, manual relabelling),
  `HeadAlign` (map MEG runs to a common head position), `BridgedElectrodes`;
  epoching with `Epoch`, `Baseline`, `AutoReject`, and `meu.epochs.combine`
  for runs.
- **Pipelines** follow scikit-learn: parameters are set in the constructor,
  `fit` / `transform` / `fit_transform` are the only entry points,
  `set_params(ica__threshold=0.9)`, `clone`, slicing, YAML configurations.
- **Data integrity**: steps never drop channels or shift the time axis
  silently; `first_samp`, `meas_date` and annotations are preserved. Every
  step passes the contract checks in `meeg_utils.testing.check_step`.
- **Acquisition quality control** (`meu.qc`): bridged electrodes,
  impedances, flat/clipped channels, outlying channels, line and other
  narrowband noise, muscle, blink and heart rate (is the EOG/ECG working?),
  MEG head movement and HPI coils, digitization, event counts. Thresholds
  are documented with their source; `meu.qc.inspect_dataset` flags
  outlying recordings across a dataset; `meu.report.build` writes HTML
  reports.
- **Processing quality control**: `qc_` metrics on every step and `plot()`
  figures (spectra before/after, bad-channel scores, sensor maps, ICA
  components, head positions, ...).
- **Command line**: `meu qc /data/bids --out qc/` and
  `meu run --preset had-meeg --datatype eeg --sources /data/bids --out derivatives/`.
- **Presets**: `meu.Pipeline.preset("had-meeg", datatype="meg")` reproduces
  the [HAD-MEEG](https://github.com/colehank/HAD-MEEG) preprocessing with its
  known issues fixed; see [docs/presets/had-meeg.md](docs/presets/had-meeg.md).
- **Batch processing**: `meu.process(pipe, recordings, root, n_jobs=8)` fits
  one copy of the pipeline per recording and saves BIDS derivatives.
- **Any system MNE reads**: the acquisition system (Neuromag/MEGIN, CTF, KIT,
  BTi, Artemis123, OPM, EEG) is detected from the data; steps with
  system-specific methods refuse systems they have not been validated for
  instead of guessing (e.g. Maxwell bad-channel detection: Neuromag and CTF).

## Installation

```bash
pip install meeg-utils
```

Python 3.11+. Version 0.2.0 replaced the 0.1 `PreprocessingPipeline` and
`BatchPreprocessingPipeline` classes with the step/pipeline API above.

## Documentation

- [User guide and API reference](https://colehank.github.io/meeg-utils/)
- [Design notes](docs/design/architecture.md) (in Chinese): scope, constraints,
  roadmap.

## Development

```bash
git clone https://github.com/colehank/meeg-utils.git
cd meeg-utils
uv sync
uv run pytest
uv run ruff check src tests && uv run mypy src
```

See the [Contributing Guide](https://colehank.github.io/meeg-utils/developer/contributing.html)
for details.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- [Documentation](https://colehank.github.io/meeg-utils/)
- [Issue Tracker](https://github.com/colehank/meeg-utils/issues)
- [Discussions](https://github.com/colehank/meeg-utils/discussions)
