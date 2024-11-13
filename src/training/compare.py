from ultralytics import YOLO
import os
import traceback
from train import read_augmentation_parameters
from distillation import DistillationTrainer

if __name__ == "__main__":
    # Đường dẫn
    data_dir = "./data/soict-hackathon-2024_dataset"
    train_project = "./runs"
    augmentation_params = read_augmentation_parameters(
        "./runs/augmentation_parameters.txt"
    )

    # Danh sách cấu hình mô hình
    model_configs = [
        {"name": "yolov8l-adsc.yaml", "train_name": "train_yolov8-adsc"},
        {"name": "yolov8l-ghost.yaml", "train_name": "train_yolov8-ghost"},
    ]

    # Tham số huấn luyện
    train_params = {
        "data": f"{data_dir}/data.yaml",
        "epochs": 5,
        "batch": -1,
        "cache": "disk",
        "device": 0,
        "project": train_project,
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

    # Huấn luyện từng mô hình
    for config in model_configs:
        print(f"\nBắt đầu huấn luyện cho mô hình {config['name']}...")
        try:
            model_path = os.path.join("./models", config["name"])
            train_name = config["train_name"]

            # Khởi tạo mô hình
            model = YOLO(model_path)

            # Cập nhật tham số huấn luyện với tên chạy cụ thể
            train_params_updated = train_params.copy()
            train_params_updated.update(
                {
                    "name": train_name,
                }
            )

            model.train(trainer=DistillationTrainer, **train_params_updated)
            print(f"Hoàn thành huấn luyện cho mô hình {config['name']}.")
        except Exception as e:
            print(f"Đã xảy ra lỗi khi huấn luyện mô hình {config['name']}: {e}")
            traceback.print_exc()
            print("Bỏ qua và chuyển sang mô hình tiếp theo...\n")
            continue
