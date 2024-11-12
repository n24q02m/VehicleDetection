from ultralytics import YOLO
from train import read_augmentation_parameters


if __name__ == "__main__":
    # Paths
    model_name = "./models/yolov8m.pt"
    data_dir = "./data/soict-hackathon-2024_dataset"
    train_project = "./runs"
    train_name = "finetune_yolov8m"

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
    model.train(**train_params)
