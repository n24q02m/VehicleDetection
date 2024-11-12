"""
    - Chỉnh sửa dòng 696 file /envs/vehicle-detection/Lib/site-packages/ultralytics/utils/check.py từ:
    ```
assert amp_allclose(YOLO("yolo11n.pt"), im)
    ```
    thành:
    ```
assert amp_allclose(YOLO("yolov8m-ghost-p2.yaml"), im)
    ```
"""

from ultralytics import YOLO
from distillation import DistillationTrainer


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
    # Paths
    model_name = "./models/custom-yolov8m-ghost-p2.yaml"
    data_dir = "./data/soict-hackathon-2024_dataset"
    train_project = "./runs"
    train_name = "distillation_custom-yolov8m-ghost-p2"

    # Read augmentation parameters from the text file
    augmentation_params = read_augmentation_parameters(
        "./runs/augmentation_parameters.txt"
    )

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
        "show": True,
        "label_smoothing": 0.1,
        **augmentation_params,
    }

    # Start training with the custom distillation trainer
    model.train(trainer=DistillationTrainer, **train_params)
