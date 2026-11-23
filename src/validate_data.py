from __future__ import annotations

import json
from pathlib import Path
from PIL import Image

EXPECTED_SPLITS = {"train": 1034, "validation": 133, "test": 128}
EXPECTED_CLASSES = {"angular_leaf_spot", "bean_rust", "healthy"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_dataset(raw_root: Path) -> dict:
    """Validate the raw dataset and return a deterministic quality report.

    Required Core checks:
      1. required splits exist;
      2. exactly the expected classes exist in every split;
      3. expected image counts are present;
      4. every image can be fully decoded by Pillow (`Image.load()` or equivalent);
      5. no split or class is empty.

    TODO: implement the checks and return a JSON-serializable report.
    The report must not contain timestamps because it is a DVC output.
    """
    classes = sorted(EXPECTED_CLASSES)
    split_reports = {}

    for split, expected_count in sorted(EXPECTED_SPLITS.items()):
        split_dir = raw_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing dataset split: {split_dir}")

        present_classes = {path.name for path in split_dir.iterdir() if path.is_dir()}
        if present_classes != EXPECTED_CLASSES:
            raise RuntimeError(
                f"Class mismatch in {split_dir}: expected {classes}, "
                f"found {sorted(present_classes)}"
            )

        class_counts = {}
        for class_name in classes:
            class_dir = split_dir / class_name
            images = sorted(
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not images:
                raise RuntimeError(f"No images found in {class_dir}")

            for image_path in images:
                with Image.open(image_path) as image:
                    image.load()    # decoded by Pillow
            class_counts[class_name] = len(images)

        image_count = sum(class_counts.values())
        if image_count != expected_count:   # exactly the expected classes exist
            raise RuntimeError(
                f"Image count mismatch in {split_dir}: expected "
                f"{expected_count}, found {image_count}"
            )

        split_reports[split] = {
            "images": image_count,
            "class_counts": class_counts,
        }

    return {
        "status": "PASS",
        "classes": classes,
        "splits": split_reports,
        "total_images": sum(report["images"] for report in split_reports.values()),
    }


def main() -> None:
    report = validate_dataset(Path("data/raw"))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/data_quality.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

# verification