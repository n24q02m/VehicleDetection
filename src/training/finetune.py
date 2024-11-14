from ultralytics import YOLO
from train import read_augmentation_parameters


if __name__ == "__main__":
    # Paths
    model_name = "./models/yolo11x.pt"
    data_dir = "./data/soict-hackathon-2024_dataset"
    train_project = "./runs"
    train_name = "finetune_yolo11x"

    # Read augmentation parameters from the text file
    augmentation_params = read_augmentation_parameters("./runs/just_mosaic.txt")

    # Initialize the model
    model = YOLO(model_name)

    # Training parameters
    train_params = {
        "data": f"{data_dir}/data.yaml",
        "epochs": 600,
        "time": 10,
        "batch": 0.8,
        "cache": "disk",
        "device": 0,
        "project": train_project,
        "name": train_name,
        "exist_ok": True,
        "optimizer": "auto",
        "seed": 42,
        "cos_lr": True,
        "fraction": 1.0,
        "multi_scale": True,
        "augment": True,
        "show": True,
        "label_smoothing": 0.1,
        **augmentation_params,
    }

    # Start training
    model.train(**train_params)
