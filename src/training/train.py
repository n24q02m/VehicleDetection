"""
    - Chỉnh sửa dòng 696 file /envs/vehicle-detection/Lib/site-packages/ultralytics/utils/check.py từ:
    ```
assert amp_allclose(YOLO("yolo11n.pt"), im)
    ```
    thành:
    ```
assert amp_allclose(YOLO("yolov8m-ghost-p2.yaml"), im)
    ```
    
    - Chỉnh sửa dòng 6 file /envs/vehicle-detection/Lib/site-packages/ultralytics/cfg/models/v8/yolov8-ghost-p2.yaml từ:
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
    model_name = "yolov8m-ghost-p2.yaml"
    model_dir = "./models"
    model_save_dir = f"{model_dir}/{model_name}"
    data_dir = "./data/soict-hackathon-2024_dataset"
    train_project = "./runs"
    train_name = "train-yolov8m-ghost-p2"

    if os.path.exists(model_save_dir):
        shutil.move(model_save_dir, model_name)

    model = YOLO(model_name)

    model.train(
        data=f"{data_dir}/data.yaml",
        fraction=1.0,  # 0.1 là sử dụng 10% dữ liệu
        cache="disk",  # True nếu cache ram
        epochs=100,  # 300 cho train from scratch, 100 cho finetune
        time=10,
        batch=-1, # -1 là tự động chọn batch size
        patience=10, # 30 cho train from scratch, 10 cho finetune
        lr0=1e-3,  # 1e-3 đối với finetune
        device=0,
        project=train_project,
        name=train_name,
        exist_ok=True,
        pretrained=False,  # True nếu finetune
        optimizer="SGD", # AdamW với finetune
        weight_decay=5e-4,  # 1e-5 đối với finetune
        warmup_epochs=5,
        augment=True,
        multi_scale=True,
        cos_lr=True,
        plots=True,
        show=True,
    )

    if os.path.exists(model_name):
        shutil.move(model_name, model_save_dir)
