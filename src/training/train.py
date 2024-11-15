"""
    - Chỉnh sửa dòng 692 file ultralytics/ultralytics/utils/check.py từ:
    ```
assert amp_allclose(YOLO("yolo11n.pt"), im)
    ```
    thành:
    ```
assert amp_allclose(YOLO("yolov8m-ghost-p2.yaml"), im)
    ```
"""

import os
import glob
import shutil
from pathlib import Path
from ultralytics import YOLO
from distillation import DistillationTrainer


def patch_ultralytics():
    """
    Patch Ultralytics package with custom modifications from patches directory.
    Only runs when executing locally, not in Kaggle environment.
    """
    try:
        # Find ultralytics package location
        import ultralytics

        ultralytics_path = Path(ultralytics.__file__).parent

        # Define patch mappings
        patches = {
            "check.py": ultralytics_path / "utils" / "check.py",
            "task.py": ultralytics_path / "nn" / "tasks.py",
            "modules_init.py": ultralytics_path / "nn" / "modules" / "__init__.py",
            "conv.py": ultralytics_path / "nn" / "modules" / "conv.py",
        }

        # Get patches directory path relative to train.py
        patches_dir = Path(__file__).parent.parent.parent / "patches"

        # Apply patches
        print("Applying patches to Ultralytics package...")
        for patch_file, target_path in patches.items():
            patch_path = patches_dir / patch_file
            if patch_path.exists():
                print(f"Patching {target_path}")
                shutil.copy2(patch_path, target_path)
            else:
                print(f"Warning: Patch file {patch_path} not found")

        print("Patches applied successfully")

    except ImportError:
        print("Warning: Ultralytics package not found")
    except Exception as e:
        print(f"Error applying patches: {e}")


def read_augmentation_parameters(file_path):
    params = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                key_value = line.strip().split(maxsplit=1)
                if len(key_value) == 2:
                    key, value = key_value
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                    params[key] = value
    return params


if __name__ == "__main__":
    # Apply patches when running locally
    patch_ultralytics()

    # Paths
    model_name = "./models/custom-yolov8m-ghost-p2.yaml"
    data_dir = "./data/soict-hackathon-2024_dataset"
    train_project = "./runs"
    train_name = "distillation_custom-yolov8m-ghost-p2"

    # Read augmentation parameters from the text file
    augmentation_params = read_augmentation_parameters("./runs/mosaic_erasing.txt")

    # Initialize the student model
    model = YOLO(model_name)

    # Training parameters
    train_params = {
        "data": f"{data_dir}/data.yaml",
        "epochs": 600,
        "time": 0.5,
        "batch": -1,
        "cache": "disk",
        "device": 0,
        "project": train_project,
        "name": train_name,
        "exist_ok": True,
        "optimizer": "auto",
        "seed": 42,
        "cos_lr": True,
        "fraction": 0.1,
        "multi_scale": True,
        "augment": True,
        "label_smoothing": 0.1,
        **augmentation_params,
    }

    # Start training with the custom distillation trainer
    model.train(trainer=DistillationTrainer, **train_params)
