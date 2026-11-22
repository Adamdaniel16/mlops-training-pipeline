# AI 53 Lab 1 Student Report
**Author : Adam Daniel BIN ZULKORNAIN**

## Pre-requisites and Workstation Preparation
>uv --version

uv 0.12.13 (0ebbd9274 2026-09-10 aarch64-pc-windows-msvc)

uv have been installed successfully

>uv run --locked pytest -q tests/test_model.py
```bash
Using CPython 3.12.14
Creating virtual environment at: .venv
    Built antlr4-python3-runtime==4.9.3
Installed 162 packages in 6.28s
.   [100%]
1 passed in 10.82s
```
## Milestone 1 - Make training repeatable
>pyproject.toml

project and dependency declaration

>uv.lock

resolved dependency graph

>.venv/

virtual environment used to use specific version of dependencies (to avoid version conflict)

### 1.1 Run the starter tests

starter test run result (```uv run --locked pytest -q```)

>11 failed, 8 passed in 7.36s

Some tests fail because of TODOs that are needed to be completed. Leave the tests unchanged and only work in source code.

### 1.2 Set the random seeds

- Completed the function set_reproducibility(seed) so that Python, NumPy and PyTorch are controlled. 
- Improved CUDA/cuDNN repeatability when CUDA is available.
- Implemented seed_worker(seed) to seed a DataLoader worker from PyTorch's worker seed.
- Implemented correctly make_generator(seed) to return a seeded ```torch.Generator``` used for deterministic shuffling.

Test run result (```uv run --locked pytest -q tests/test_reproducibility.py```)
>3 passed in 1.91s  # reproducibility test

>8 failed, 11 passed in 6.93s # starter test (global)

**Engineering checkpoint 2.** Does random_state=42 guarantee bit-for-bit identical results on every CPU/GPU/software stack?

>No, the same random state doesn't guarantee bit-for-bit identical results as hardware (kernel) level technologies still differs between machines. Kernel-level differences (CPU vs GPU execution paths, Parallel thread scheduling, lib version, …) could still differs it.

## Milestone 2 - Extract and track the images
### 2.1 Get the Beans images

**Option A : Use the included archives.** (```uv run --locked python scripts/download_dataset.py```)
```
--archive-dir dataset-archives
train.zip: SHA-256 OK
validation.zip: SHA-256 OK
test.zip: SHA-256 OK
train     : 1034 images OK
validation:  133 images OK
test      :  128 images OK
Dataset ready in data/raw/.
```

Checking the content of data/raw (```tree .\data\raw\```)
```
C:\xxx\AI53_PHASE1_LAB\AI53_MLOPS_PROJECT_PHASE1_STARTER\DATA\RAW
├───test
│   ├───angular_leaf_spot
│   ├───bean_rust
│   └───healthy
├───train
│   ├───angular_leaf_spot
│   ├───bean_rust
│   └───healthy
└───validation
    ├───angular_leaf_spot
    ├───bean_rust
    └───healthy
```

### 2.2 Track the images with DVC

Data Versioning Control (DVC) allows Content-Based Versioning. Each version of data is fingerprinted (based on its content) so that we can keep the same file name.

Initialized DVC, track the images and set default DVC remote. TO verify :```uv run --locked dvc remote list```

```lab-data        C:\xxx\AI53_Phase1_Lab\ai53-phase1-dvc-storage       (default)```

```data/raw``` file is tracked by git and contains values for outs -> md5, size, nfiles, hash, path

**Engineering checkpoint 3.** In one sentence each, state what the Git remote stores, what the DVC remote stores, and whether git push transfers the DVC-managed raw images.
- Git remote stores remote commits of files such as model implementation + dvc/raw (points to dataset used at the specific commit) excluding raw dataset files
- DVC remote stores dataset files identified by its version that are pointable by git
- No, git push did not transfers the DVC-managed raw images as its included as ignored file in .gitignore to avoid repository bloat, expensive clones, inefficient differentiation, and collaboration friction 

push dvc remote `uv run --locked dvc push .\data\raw.dvc`
```
Collecting                                                                                     |1.31k [00:00, 4.20kentry/s]
Pushing
1297 files pushed
```

## Milestone 3 - Implement a Core data-quality gate
### Implemented `validate_data.py` to validate raw dataset and output deterministic quality report.

The implementation will verify:
- required splits exist;
- exactly the expected classes exist in every split;
- expected image counts are present;
- every image can be fully decoded by Pillow (`Image.load()` or equivalent);
- no split or class is empty.

data validation test (`uv run --locked pytest -q tests/test_data_validation.py tests/test_validation_contract.py`)
```
.......     [100%] 
7 passed in 0.21s
```

```json
{
  "classes": [
    "angular_leaf_spot",
    "bean_rust",
    "healthy"
  ],
  "splits": {
    "test": {
      "class_counts": {
        "angular_leaf_spot": 43,
        "bean_rust": 43,
        "healthy": 42
      },
      "images": 128
    },
    "train": {
      "class_counts": {
        "angular_leaf_spot": 345,
        "bean_rust": 348,
        "healthy": 341
      },
      "images": 1034
    },
    "validation": {
      "class_counts": {
        "angular_leaf_spot": 44,
        "bean_rust": 45,
        "healthy": 44
      },
      "images": 133
    }
  },
  "status": "PASS",
  "total_images": 1295
}
```

