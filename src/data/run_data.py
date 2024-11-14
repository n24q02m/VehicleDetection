import os
import zipfile
import shutil
import requests
from download_dataset import main as download_dataset_main
from preprocess_data import main as preprocess_data_main
from explore_dataset import main as explore_dataset_main
from augment_data import main as augment_data_main

def download_file(url, destination):
    """Tải tệp từ URL đến đường dẫn đích."""
    print(f"Đang tải tệp từ {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(destination, "wb") as f:
            shutil.copyfileobj(response.raw, f)
        print(f"Đã tải tệp về {destination}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi tải tệp: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """Giải nén tệp zip."""
    print(f"Đang giải nén {zip_path} vào {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print("Đã giải nén xong.")


def zip_dataset(folder_path, output_path):
    """Nén thư mục dataset thành tệp zip."""
    print(f"Đang nén {folder_path} vào {output_path}...")
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(folder_path))
                zipf.write(file_path, arcname)
    print("Đã nén dataset.")


if __name__ == "__main__":
    dataset_zip = "soict-hackathon-2024_dataset.zip"
    data_dir = "./data"
    dataset_dir = os.path.join(data_dir, "soict-hackathon-2024_dataset")

    # Tạo thư mục data nếu chưa tồn tại
    os.makedirs(data_dir, exist_ok=True)

    # Kiểm tra nếu tệp zip đã tồn tại
    zip_paths = [os.path.join(".", dataset_zip), os.path.join(data_dir, dataset_zip)]
    zip_exists = any(os.path.exists(path) for path in zip_paths)

    if zip_exists:
        # Sử dụng tệp zip hiện có
        zip_path = next(path for path in zip_paths if os.path.exists(path))
        print(f"Tìm thấy tệp {zip_path}.")
        extract_zip(zip_path, data_dir)
        print("Dataset đã sẵn sàng.")
    else:
        # Tải tệp zip từ Google Drive
        print("Không tìm thấy tệp dataset zip. Đang tải từ Google Drive...")
        google_drive_url = "https://drive.google.com/uc?id=19fm1TDHeRypdpqNdj0EyXAppPeGBV5Wh&export=download"
        zip_path = os.path.join(data_dir, dataset_zip)
        download_success = download_file(google_drive_url, zip_path)
        if download_success and os.path.exists(zip_path):
            extract_zip(zip_path, data_dir)
            print("Dataset đã sẵn sàng.")
        else:
            print("Không thể tải tệp zip từ Google Drive.")
            print("Tiếp tục chạy các script xử lý dữ liệu...")

    # Kiểm tra nếu thư mục dataset đã tồn tại
    if os.path.exists(dataset_dir):
        print(
            "Thư mục dataset đã tồn tại. Bỏ qua các bước xử lý và tăng cường dữ liệu."
        )
    else:
        # Chạy các script xử lý dữ liệu
        print("Bắt đầu quá trình xử lý và tăng cường dữ liệu...")
        download_dataset_main()
        explore_dataset_main()
        preprocess_data_main()
        augment_data_main()
        print("Hoàn thành quá trình xử lý và tăng cường dữ liệu.")

    # Nén dataset sau khi hoàn thành
    if not zip_exists:
        zip_dataset(dataset_dir, os.path.join(data_dir, dataset_zip))
        print("Dataset đã được nén lại.")
