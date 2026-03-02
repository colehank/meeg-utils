# meeg-utils

[![CI](https://github.com/`colehank`/meeg-utils/workflows/CI/badge.svg)](https://github.com/colehank/meeg-utils/actions)
[![Documentation](https://github.com/colehank/meeg-utils/workflows/Documentation/badge.svg)](https://colehank.github.io/meeg-utils/)
[![codecov](https://codecov.io/gh/colehank/meeg-utils/branch/main/graph/badge.svg)](https://codecov.io/gh/colehank/meeg-utils)
[![PyPI version](https://badge.fury.io/py/meeg-utils.svg)](https://badge.fury.io/py/meeg-utils)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> A Python-based MEG/EEG preprocessing toolkit built on MNE-Python, providing a high-level, user-friendly API for preprocessing electrophysiological data.

## 🌟 Features

- **High-level API** - Simple, intuitive interface for complex preprocessing pipelines
- **BIDS Support** - Native Brain Imaging Data Structure (BIDS) support
- **Advanced Preprocessing** - Bad channel detection, line noise removal, ICA-based artifact removal
- **Batch Processing** - Efficient parallel processing of multiple datasets
- **Type-safe** - Full type hints for better IDE support
- **Well-tested** - Comprehensive test suite (52 tests, >80% coverage, 0 warnings)

## 📦 Installation

```bash
pip install meeg-utils
```

## 🚀 Quick Start

```python
from meeg_utils.preprocessing import PreprocessingPipeline
from mne_bids import BIDSPath

# Create pipeline
pipeline = PreprocessingPipeline(
    input_path=BIDSPath(
        subject="01", session="01", task="rest",
        datatype="eeg", root="/data/bids"
    ),
    output_dir="/data/output"
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
```

**Batch processing:**

```python
from meeg_utils.preprocessing import BatchPreprocessingPipeline

# Process multiple subjects in parallel
batch = BatchPreprocessingPipeline(
    input_paths=bids_paths,  # List of BIDSPaths
    output_dir="/data/output",
    n_jobs=4  # Use 4 parallel workers
)

batch.run(detect_bad_channels=True, remove_line_noise=True, apply_ica=True)
```

## 📚 Documentation

**Full documentation:** https://colehank.github.io/meeg-utils/

- [Installation Guide](https://colehank.github.io/meeg-utils/user_guide/installation.html)
- [Quick Start](https://colehank.github.io/meeg-utils/user_guide/quickstart.html)
- [Preprocessing Guide](https://colehank.github.io/meeg-utils/user_guide/preprocessing.html)
- [Batch Processing](https://colehank.github.io/meeg-utils/user_guide/batch_processing.html)
- [API Reference](https://colehank.github.io/meeg-utils/api/preprocessing.html)
- [Contributing Guide](https://colehank.github.io/meeg-utils/developer/contributing.html)

## 🛠️ Development

```bash
# Clone and setup
git clone https://github.com/colehank/meeg-utils.git
cd meeg-utils
uv sync --dev
uv run pre-commit install

# Run tests
uv run pytest

# Build docs
cd docs && uv run make html
```

See the [Contributing Guide](https://colehank.github.io/meeg-utils/developer/contributing.html) for detailed development instructions.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built on the excellent [MNE-Python](https://mne.tools/) ecosystem.

## 📞 Support

- 📖 [Documentation](https://colehank.github.io/meeg-utils/)
- 🐛 [Issue Tracker](https://github.com/colehank/meeg-utils/issues)
- 💬 [Discussions](https://github.com/colehank/meeg-utils/discussions)
