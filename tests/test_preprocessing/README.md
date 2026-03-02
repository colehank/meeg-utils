# meeg-utils Preprocessing Module - TDD Status

## 🔴 RED Phase - Tests Written, Implementation Pending

Following strict **Test-Driven Development (TDD)** principles:

### Current Status

✅ **Tests Written** - Comprehensive test suite for preprocessing module
❌ **Implementation** - Intentionally NOT implemented yet
⏳ **Next Step** - Verify tests fail correctly, then implement

### Test Coverage

#### Core Functionality Tests (`tests/test_preprocessing/test_pipeline.py`)

1. **Path Input Parsing** (6 tests)
   - Plain string paths
   - Pathlib.Path objects
   - BIDS string format
   - BIDSPath objects
   - Error handling for invalid paths

2. **Data Loading** (4 tests)
   - Load from BIDSPath
   - Load from string path
   - Automatic datatype inference
   - Data caching

3. **Filtering and Resampling** (4 tests)
   - Valid filter parameters
   - Nyquist frequency validation
   - Highpass/lowpass order validation
   - Default parameters

4. **Bad Channel Detection** (5 tests)
   - EEG bad channel detection (PREP)
   - MEG bad channel detection (Maxwell)
   - Channel interpolation
   - Skip interpolation option

5. **Line Noise Removal** (3 tests)
   - Default frequency (50 Hz)
   - Custom frequency (60 Hz)
   - Different methods for EEG vs MEG

6. **ICA Processing** (6 tests)
   - Automatic component labeling
   - Custom n_components
   - Default components for EEG/MEG
   - Artifact regression
   - Manual labels

7. **Full Pipeline** (6 tests)
   - End-to-end with defaults
   - Custom filter parameters
   - Selective step execution
   - Intermediate file saving
   - Output directory creation

8. **Output Management** (3 tests)
   - Save preprocessed data
   - Automatic BIDS-compliant naming
   - Derivative metadata files

9. **Configuration** (3 tests)
   - CUDA acceleration
   - Parallel processing (n_jobs)
   - Random state reproducibility

#### Batch Processing Tests (`tests/test_preprocessing/test_batch.py`)

1. **Batch Input Parsing** (4 tests)
   - List of BIDSPath objects
   - List of string paths
   - Mixed path types
   - Empty list validation

2. **Batch Processing** (5 tests)
   - Default parameters
   - Parallel processing
   - Custom parameters
   - Skip existing files
   - Error handling

3. **Batch Logging** (2 tests)
   - Log file creation
   - Logging level configuration

4. **Output Organization** (2 tests)
   - BIDS structure preservation
   - Derivatives directory creation

5. **Progress Tracking** (2 tests)
   - Progress reporting
   - Partial completion handling

### Test Infrastructure

- **Fixtures** (`tests/conftest.py`)
  - `sample_eeg_raw`: Synthetic EEG data
  - `sample_meg_raw`: MNE sample MEG data
  - `temp_output_dir`: Temporary output directory
  - `mock_bids_path`: Mock BIDS dataset
  - `mock_bids_path_string`: String version of BIDS path

### Total Test Count

- **Pipeline Tests**: 40 tests
- **Batch Tests**: 15 tests
- **Total**: 55 comprehensive tests

## TDD Workflow

### ✅ Step 1: RED - Write Failing Tests (CURRENT)

```bash
# Run tests to verify they fail correctly
uv run pytest tests/test_preprocessing/ -v
```

Expected: All tests should fail with `NotImplementedError`

### ⏳ Step 2: GREEN - Minimal Implementation

After user confirms tests fail correctly:
1. Implement minimal code to make each test pass
2. Follow reference code patterns
3. Use proper type hints and logging

### ⏳ Step 3: REFACTOR - Clean Up

After all tests pass:
1. Remove duplication
2. Improve code organization
3. Enhance documentation

## Design Principles

Based on reference code analysis:

1. **Input Flexibility**: Support string, Path, BIDSPath, BIDS string
2. **BIDS Compliance**: Follow BIDS derivatives structure
3. **Modular Design**: Separate concerns (bad channels, ICA, etc.)
4. **Logging**: Use loguru for detailed logging
5. **Type Safety**: Complete type hints throughout
6. **Error Handling**: Graceful error handling with clear messages

## Key APIs (Designed from Tests)

```python
# Single file preprocessing
pipeline = PreprocessingPipeline(
    input_path="path/to/data",  # str | Path | BIDSPath
    output_dir="path/to/output",
    n_jobs=1,
    use_cuda=False,
    random_state=42,
)

result = pipeline.run(
    filter_params={"highpass": 0.1, "lowpass": 100.0, "sfreq": 250.0},
    detect_bad_channels=True,
    remove_line_noise=True,
    apply_ica=True,
    save_intermediate=True,
)

# Batch processing
batch = BatchPreprocessingPipeline(
    input_paths=[path1, path2, path3],
    output_dir="path/to/output",
    n_jobs=4,
)

batch.run(skip_existing=True, logging_level="INFO")
```

## Next Steps

1. **User reviews tests** and confirms they cover requirements
2. **Run tests to verify RED phase** (`pytest` should show failures)
3. **User approves** ("通过") to proceed to GREEN phase
4. **Implement minimal code** to pass tests one by one
5. **Refactor** once all tests pass

---

**Following TDD Iron Law**: NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST ✅
