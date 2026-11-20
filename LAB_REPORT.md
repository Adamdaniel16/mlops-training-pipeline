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
## Milestone 1
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

Test run result
>3 passed in 1.91s  # reproducibility test

>8 failed, 11 passed in 6.93s # starter test (global)

**Engineering checkpoint 2.** Does random_state=42 guarantee bit-for-bit identical results on every CPU/GPU/software stack?

>No, the same random state doesn't guarantee bit-for-bit identical results as hardware (kernel) level technologies still differs between machines. Kernel-level differences (CPU vs GPU execution paths, Parallel thread scheduling, lib version, …) could still differs it.

