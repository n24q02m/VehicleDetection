import os

# Cài đặt các thư viện cần thiết
os.system("pip install -U ultralytics kaggle tqdm albumentations")

# Kiểm tra môi trường và thiết lập thư mục dataset
if "KAGGLE_URL_BASE" in os.environ:
    # Trên Kaggle
    os.system("git clone https://github.com/n24q02m/VehicleDetection.git")
    os.chdir("VehicleDetection")
    dataset_dir = "/kaggle/working/VehicleDetection/data/soict-hackathon-2024_dataset"
else:
    # Trên máy ảo
    dataset_dir = "C:\\Users\\Administrator\\Desktop\\VehicleDetection\\src\\data\\soict-hackathon-2024_dataset"

# Set YOLO dataset directory
os.system(f"yolo settings datasets_dir={dataset_dir}")

# Chạy hàm main từ train.py
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.training.train import main as train_main

train_main()

from src.utils.gitaction import git_push_changes

git_push_changes("refactor: Update after run")