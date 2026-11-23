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

**Engineering checkpoint 1.** Why is a working `model.pt` insuﬀicient evidence that an experiment is reproducible?

A `model.pt` file only stores the final trained weights. It provides no record of how those weights were generated, preventing someone else from re-running the experiment to verify the results. To prove reproducibility, you must also capture the full provenance of the training run:
- **Code**: The exact version of the training scripts and model architecture (via a Git commit SHA)
- **Data**: The specific dataset version, preprocessing steps, and train/validation splits (tracked via DVC)
- **Configuration & Seeds**: The exact hyperparameters (e.g., learning rate, batch size, epochs) and random seeds used to control randomness
- **Software Environment**: The locked dependency and runtime versions (e.g., Python, PyTorch, OS) 

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

## Milestone 4 - Design the DVC pipeline from scratch
Created `dvc.yaml` with four stages: `validate`, `prepare`, `train`, and `evaluate`.

### Pipeline design

- `validate` checks `data/raw` and writes `reports/data_quality.json`
- `prepare` depends on the raw data and quality report. It uses `data.image_size` and writes `data/processed` plus `reports/data_summary.json`
- `train` uses prepared train/validation images and manifests. It tracks the local checkpoint and training artifacts with DVC, and records the numeric training summary as a metric.
- `evaluate` uses prepared test images and the selected MLflow Run ID. It writes the test metrics and confusion-matrix image
- TRAIN and EVALUATE are separate branches after PREPARE. EVALUATE does not depend on TRAIN or `models/model.pt`; the selected model remains in MLflow
- Python stages list `pyproject.toml` and `uv.lock` as dependencies. Each stage tracks only the parameters it uses

### Pipeline validation

Commands:

```
uv run --locked dvc stage list
uv run --locked dvc dag
```

Results:
```
validate  Reports reports/data_quality.json
prepare   Outputs data/processed; Reports reports/data_summary.json
train     Outputs models/model.pt, reports/training_curves.png, reports/run_manifest.json…
evaluate  Outputs reports/test_confusion_matrix.png; Reports reports/test_metrics.json
```
```
        +--------------+          
        | data\raw.dvc |          
        +--------------+          
           **        **           
         **            **         
        *                **       
+----------+               *      
| validate |             **       
+----------+           **         
           **        **           
             **    **             
               *  *               
           +---------+            
           | prepare |            
           +---------+            
           **        **           
         **            **         
        *                *        
  +-------+         +----------+  
  | train |         | evaluate |  
  +-------+         +----------+ 
```

Confirmed that all four stages appear and that EVALUATE does not depend on TRAIN.

### What the pipeline output shows
DVC recognizes all four stages and their declared outputs. The `data\raw.dvc` node in the DAG represents the DVC-tracked raw dataset input; it is not an additional pipeline stage.

The raw dataset feeds `validate`, and validation’s quality report is a dependency of `prepare`, so preparation follows the quality gate. PREPARE then supplies shared processed data to TRAIN and EVALUATE. These stages are separate branches: TRAIN creates the candidate model and training artifacts, while EVALUATE uses the prepared test data and the selected MLflow Run. EVALUATE does not consume `models/model.pt` or depend on TRAIN; the selected model remains in MLflow.

## Milestone 5 - Record what each MLflow Run used

### 5.1 Run provenance and complete `src/provenance.py`

Implemented `build_run_manifest()` in `src/provenance.py`. Each Run’s `reports/run_manifest.json` records its Git revision and working-tree status, training/validation data identity, environment versions, CUDA runtime, training configuration, device, random seed, and MLflow Run ID.

Provenance test (`uv run --locked pytest -q .\tests\test_provenance.py`)
```
.       [100%]
1 passed in 1.88s
```

### 5.2 MLflow Run logging

Updated `src/train.py` to create one MLflow Run per training execution. At the start of the Run, it writes the provenance manifest and logs the required parameters, including `training_data_identity`.

The Run also logs the `bean-training` dataset from `data/processed/manifests/train.csv`, with `label` as its target. During training it logs train/validation loss, validation accuracy, and validation macro-F1 at each epoch, plus total training time.

Logged artifacts:
- `reports/training_curves.png`
- `reports/train_summary.json`
- `reports/run_manifest.json`

Logged the final PyTorch model as `bean-cnn`, with a signature and input example
from the validation set, and metadata for class names and image size. The
model is exported on CPU; the test set is not used for model logging.

### 5.3 Tests and commit

Full test suite (`uv run --locked pytest -q`):
```
...................     [100%]
19 passed in 6.74s
```

**Engineering checkpoint 4**. In one sentence per tool, explain what Git, DVC and MLflow each record about a training Run:
- **Git** records the committed code version and whether the working tree has uncommitted changes.
- **DVC** tracks dataset and pipeline file dependencies and outputs by version.
- **MLflow** records each training Run’s parameters, dataset input, metrics, artifacts, and model.

## Milestone 6 - Predict which stages need to run again

### 6.1 Predict before running DVC

| Change | VALIDATE | PREPARE | TRAIN | EVALUATE |
|---|---|---|---|---|
| Change `train.learning_rate` | REUSED | REUSED | RERUN | REUSED |
| Change `data.image_size` | REUSED | RERUN | RERUN | NEEDS UPDATE |
| Edit `src/validate_data.py`; quality report stays the same | RERUN | REUSED | REUSED | REUSED |
| Edit only `README.md` | REUSED | REUSED | REUSED | REUSED |

**Engineering checkpoint 5.** Explain your labels for at least two rows: one setting change and one file edit. For each stage in those rows, name the input that changes or explain why its inputs stay the same. Use your `dvc.yaml` for this explanation.
- only **TRAIN** lists `train.learning_rate` as a parameter. With the selected Run ID fixed, **EVALUATE**’s inputs haven’t changed.
- Validator edit, same report: **VALIDATE** depends on the source file, so it reruns. If its report content remains unchanged, **PREPARE**’s report dependency is unchanged, so downstream stages can reuse their outputs.

### 6.2 Start MLflow

MLFlow page opened succesfully via (`uv run --locked mlflow server --host 127.0.0.1 --port 5000 --workers 1 --backend-store-uri sqlite:///mlflow.db`)

### 6.3 Run and check the training stages

1. Run A (baseline)
```
MLflow Run ID : 519d86318a0d431fb3f9fc8f97786329
```

2. Re-run for verification (unchanged)
```
'data\raw.dvc' didn't change, skipping                                                                     
Stage 'validate' didn't change, skipping                                                                   
Stage 'prepare' didn't change, skipping                                                                    
Stage 'train' didn't change, skipping                                                                      
Data and pipelines are up to date.
```

3. after add comment to src/validate_data.py
```
MLflow Run ID : b961546f67de49e795b49088170cab40
> it validates reruns, the rest are reused
```

### 6.4 Verify the learning-rate change

After set train.learning_rate: 0.0003 in params.yaml
```
(Run B)
MLflow Run ID : f57fed6025364ee5bc746d680b822ee1
```

Check | VALIDATE | PREPARE | TRAIN | EVALUATE | Terminal lines
|---|---|---|---|---|---|
Rerun with no change | REUSED | REUSED | REUSED | NOT RUN | Data and pipelines are up to date
Add comment to `src/validate_data.py` | RERUN | REUSED | REUSED | NOT RUN | Running stage 'validate'
Learning rate set to 0.0003 (Run B) | REUSED | REUSED | RERUN | NOT RUN | Running stage 'train'

## Milestone 7 - Compare three controlled runs

### 7.1 Create Run C

1. Set the `learning rate` back to 0.001
2. Change `train.dropout` to 0.5

>**Initial expecation:**
Run C expectation (dropout 0.30 → 0.50): The CNN is small (the classifier is a single Linear(64→3) layer), so it is more likely to underfit than overfit. I expect higher train loss and a slightly slower improvement in val loss. Val accuracy and macro-F1 should be similar to A or slightly lower. Higher dropout would only help if A shows a large train/val gap, and I don't expect one.

Run | Controlled change | Val. Accuracy | Val. Macro-F1 | Training time | MLflow Run ID
|---|---|---|---|---|---|
A | Baseline (lr 0.001, dropout 0.30) | 0.654 | 0.642 | 12.355s | 519d86318a0d431fb3f9fc8f97786329
B | learning_rate 0.0003 | 0.692 | 0.677 | 11.688s | f57fed6025364ee5bc746d680b822ee1
C | dropout 0.50 | 0.617 | 0.607 | 13.386s | 37f19b77a90b4b24a0adb729b76f86db

### 7.2 Compare the validation results

- **B vs A:** Macro-F1 rose from 0.642 to 0.677 (+0.035) and accuracy from 0.654 to 0.692. This does not support my prediction that a lower learning rate would converge more slowly and score lower. A plausible reason is that lr=0.001 with Adam was noisy or overshooting at the last epoch, while 0.0003 gave a more stable final model.

- **C vs A:** Macro-F1 fell from 0.642 to 0.607 (−0.035). This supports my prediction that dropout 0.50 would not help a small, probably underfitting model. The drop was a bit larger than "slightly."

- **Training time:** The differences (11.7 s to 13.4 s) are small and likely CPU timing noise. The settings changed should not affect cost.

>**Caveat:** One seed, one run per setting, 133 validation images. A 0.035 gap is about 5 images, so the ranking is suggestive, not proven. All scores are low (about 0.61 to 0.68), so every candidate is probably under-trained.

**Engineering checkpoint 6.** Justify the primary metric you chose for this three-class problem. Explain why you use the test set only after choosing a model.

>Primary metric: val macro-F1. It averages F1 across the three classes (angular_leaf_spot, bean_rust, healthy) with equal weight, so a model can't score well by favoring one class, and it accounts for both precision and recall. Accuracy hides per-class failures. Ties are broken by lower val_loss, then shorter training time.

>Test set only after selection: If the test set guided the choice, the final score would be optimistically biased because we'd have picked the model that happens to fit it. Keeping it untouched until one Run is frozen gives an unbiased estimate of how the chosen model performs on new data.

## Milestone 8 - Select and test one model

### Selected Run: `Run B, f57fed6025364ee5bc746d680b822ee1`.
- **Val macro-F1**: 0.677 (A: 0.642, C: 0.607)
- **Other evidence**: best val accuracy (0.692) and shortest training time (11.7 s)
- **Limitation**: one seed and 133 validation images, so a 5-image margin may be noise. Absolute performance is low, so a longer schedule or tuning would likely improve all runs.

