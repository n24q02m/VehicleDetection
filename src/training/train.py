"""
    - Chỉnh sửa dòng 696 file /envs/vehicle-detection/Lib/site-packages/ultralytics/utils/check.py từ:
    ```
assert amp_allclose(YOLO("yolo11n.pt"), im)
    ```
    thành:
    ```
assert amp_allclose(YOLO("yolov8s-ghost.yaml"), im)
    ```
    
    - Chỉnh sửa dòng 6 file /envs/vehicle-detection/Lib/site-packages/ultralytics/cfg/models/v8/yolov8-ghost.yaml từ:
    ```
nc: 80
    ```
    thành:
    ```
nc: 4
    ```
"""

import os
import shutil
from ultralytics import YOLO

if __name__ == "__main__":
    model_name = "yolov8s-ghost.yaml"
    model_dir = "./models"
    model_save_dir = f"{model_dir}/{model_name}"
    data_dir = "./data/soict-hackathon-2024_dataset"
    train_project = "./runs"
    train_name = "train-yolov8s-ghost-edit"

    if os.path.exists(model_save_dir):
        shutil.move(model_save_dir, model_name)

    model = YOLO(model_name)

    model.train(
        data=f"{data_dir}/data.yaml",
        epochs=20,
        batch=16,
        lr0=1e-2,
        device=0,
        project=train_project,
        name=train_name,
        exist_ok=True,
        pretrained=False,
        optimizer="SGD",
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        amp=True,
        patience=10,
        cos_lr=True,
        plots=True,
        show=True,
    )

    if os.path.exists(model_name):
        shutil.move(model_name, model_save_dir)
