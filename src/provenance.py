from __future__ import annotations

import hashlib
import json
from pathlib import Path
import importlib.metadata
import platform
import subprocess
import torch


def training_data_identity(
    train_manifest: Path = Path("data/processed/manifests/train.csv"),
    validation_manifest: Path = Path("data/processed/manifests/validation.csv"),
) -> str:
    """Return a deterministic identity for the training + validation manifests.

    This helper is provided because the final evaluator uses the same identity
    to verify that the selected MLflow Run matches the current prepared data.
    Do not hard-code the value.
    """
    paths = (train_manifest, validation_manifest)
    if any(not path.exists() for path in paths):
        return "unavailable"

    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_run_manifest(params: dict, device: str, mlflow_run_id: str) -> dict:
    """Build experiment provenance metadata.

    TODO: collect at least:
      - Git commit SHA and whether the working tree is dirty;
      - identity of the prepared training + validation manifests;
      - Python, PyTorch, torchvision, DVC and MLflow versions;
      - OS/platform and CUDA runtime (when available);
      - compute device; random seed; image size; training parameters;
      - MLflow Run ID.

    Hints: subprocess, platform, importlib.metadata, torch,
    training_data_identity().
    """
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )

    if git_commit.returncode == 0:
        git_status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        commit_sha = git_commit.stdout.strip()
        working_tree_dirty = bool(git_status.stdout.strip())
    else:
        # Tests may call this function from a temporary directory outside a Git repo
        commit_sha = None
        working_tree_dirty = None

    train = params["train"]
    augmentation = params["augmentation"]

    return {
        "source": {
            "git_commit": commit_sha,
            "working_tree_dirty": working_tree_dirty,
        },
        "data": {
            "training_data_identity": training_data_identity(),
        },
        "environment": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "torch": importlib.metadata.version("torch"),
            "torchvision": importlib.metadata.version("torchvision"),
            "dvc": importlib.metadata.version("dvc"),
            "mlflow": importlib.metadata.version("mlflow"),
            "cuda": torch.version.cuda,
        },
        "configuration": {
            "image_size": int(params["data"]["image_size"]),
            "batch_size": int(train["batch_size"]),
            "epochs": int(train["epochs"]),
            "learning_rate": float(train["learning_rate"]),
            "dropout": float(train["dropout"]),
            "augmentation_enabled": bool(augmentation["enabled"]),
            "horizontal_flip": float(augmentation["horizontal_flip"]),
            "rotation_degrees": float(augmentation["rotation_degrees"]),
        },
        "execution": {
            "device": device,
            "random_state": int(train["random_state"]),
            "mlflow_run_id": mlflow_run_id,
        },
    }


def write_run_manifest(params: dict, device: str, mlflow_run_id: str, path: Path) -> dict:
    manifest = build_run_manifest(params, device, mlflow_run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
