import os
import shutil
from ultralytics import YOLO

if __name__ == "__main__":
    data_dir = "./data/soict-hackathon-2024_dataset"
    train_project = "./src/training/runs"
    train_name = "train"

    if os.path.exists("./models/yolov8s.pt"):
        shutil.move("./models/yolov8s.pt", "./yolov8s.pt")

    model = YOLO("yolov8s.pt")

    model.train(
        data=f"{data_dir}/data.yaml",
        epochs=10,
        device=0,
        project=train_project,
        name=train_name,
        exist_ok=True,
        cos_lr=True,
        plots=True,
    )

    if os.path.exists("./yolov8s.pt"):
        shutil.move("./yolov8s.pt", "./models/yolov8s.pt")
