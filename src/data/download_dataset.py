import os
import kaggle
import zipfile
import shutil

def download_file(dataset_name, destination_dir):
    """Download dataset from Kaggle."""
    if not os.path.exists(destination_dir):
        print(f"Downloading dataset {dataset_name}...")
        try:
            kaggle.api.dataset_download_files(dataset_name, path=destination_dir, unzip=False)
            print(f"Downloaded dataset to {destination_dir}")
            return True
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False
    else:
        print(f"Directory {destination_dir} already exists.")
        return True

def create_folders():
    base_dir = "./data/soict-hackathon-2024_dataset"
    folders = [
        os.path.join(base_dir, "images", "train"),
        os.path.join(base_dir, "images", "val"),
        os.path.join(base_dir, "labels", "train"),
        os.path.join(base_dir, "labels", "val"),
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print("Đã tạo các thư mục cần thiết.")

def create_yaml():
    yaml_content = """
path: ./
train: ./images/train
val: ./images/val

names:
  0: motorbike
  1: car
  2: bus
  3: truck
"""
    yaml_path = "./data/soict-hackathon-2024_dataset/data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"Đã tạo tệp {yaml_path}")

def extract_zip(zip_path, extract_to):
    print(f"Đang giải nén {zip_path} vào {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Đã giải nén xong.")

def main():
    os.makedirs("./data", exist_ok=True)
    dataset_name = "n24q02m/raw-vehicle-detection-dataset"
    extract_dir = "./data/extracted_data"
    os.makedirs(extract_dir, exist_ok=True)

    # Download dataset from Kaggle
    if download_file(dataset_name, "./data"):
        zip_files = [f for f in os.listdir("./data") if f.endswith('.zip')]
        for zip_file in zip_files:
            extract_zip(os.path.join("./data", zip_file), extract_dir)
    
    create_folders()
    create_yaml()

if __name__ == "__main__":
    main()
