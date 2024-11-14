import os
import requests
import zipfile
import shutil

def download_file(url, destination):
    if not os.path.exists(destination):
        print(f"Đang tải tệp từ {url}...")
        response = requests.get(url, stream=True)
        with open(destination, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        print(f"Đã tải tệp về {destination}")
    else:
        print(f"Tệp {destination} đã tồn tại.")

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
    zip_url = "https://drive.google.com/uc?id=19ceZ6wTnXnNGc3WtVTWabUjLVIpnY-WY&export=download"
    zip_path = "./data/train_20241023.zip"
    extract_dir = "./data/extracted_data"
    os.makedirs(extract_dir, exist_ok=True)

    if not os.path.exists(zip_path):
        download_file(zip_url, zip_path)
    else:
        print(f"Tệp {zip_path} đã tồn tại.")

    create_folders()
    create_yaml()
    extract_zip(zip_path, extract_dir)

if __name__ == "__main__":
    main()
