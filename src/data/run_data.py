import os
import gdown
import zipfile
import shutil
import multiprocessing
from zipfile import ZipFile
from download_dataset import main as download_dataset_main
from preprocess_data import main as preprocess_data_main
from explore_dataset import main as explore_dataset_main
from augment_data import main as augment_data_main

def download_file(url, destination):
    """Download file from URL to the specified destination using gdown."""
    print(f"Downloading file from {url}...")
    try:
        gdown.download(url, destination, quiet=False)
        print(f"Downloaded file to {destination}")
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """Extract zip file using multiprocessing for speed."""
    print(f"Extracting {zip_path} into {extract_to}...")
    def extract_member(member):
        zip_ref.extract(member, extract_to)

    with ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.infolist()
        with multiprocessing.Pool() as pool:
            pool.map(extract_member, members)
    print("Extraction completed.")


def zip_dataset(folder_path, output_path):
    """Compress the dataset folder into a zip file."""
    print(f"Compressing {folder_path} into {output_path}...")
    # Ensure the output path does not have the .zip extension for make_archive
    base_name = os.path.splitext(output_path)[0]
    shutil.make_archive(base_name=base_name, format='zip', root_dir=folder_path)
    print("Dataset compressed.")


if __name__ == "__main__":
    dataset_zip = "soict-hackathon-2024_dataset.zip"
    data_dir = "./data"
    dataset_dir = os.path.join(data_dir, "soict-hackathon-2024_dataset")

    # Tạo thư mục data nếu chưa tồn tại
    os.makedirs(data_dir, exist_ok=True)

    # Kiểm tra nếu thư mục dataset đã tồn tại
    if os.path.exists(dataset_dir):
        print("Thư mục dataset đã tồn tại. Bỏ qua các bước xử lý và tăng cường dữ liệu.")
    else:
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
            google_drive_url = (
                "https://drive.google.com/uc?id=19fm1TDHeRypdpqNdj0EyXAppPeGBV5Wh"
            )
            zip_path = os.path.join(data_dir, dataset_zip)
            download_success = download_file(google_drive_url, zip_path)

            if download_success and os.path.exists(zip_path):
                extract_zip(zip_path, data_dir)
                print("Dataset đã sẵn sàng.")
            else:
                print("Không thể tải tệp zip từ Google Drive.")
                print("Tiếp tục chạy các script xử lý dữ liệu...")

                # Chạy các script xử lý dữ liệu
                print("Bắt đầu quá trình xử lý và tăng cường dữ liệu...")
                download_dataset_main()
                explore_dataset_main()
                preprocess_data_main()
                augment_data_main()
                print("Hoàn thành quá trình xử lý và tăng cường dữ liệu.")

                # Nén dataset sau khi hoàn thành xử lý
                zip_dataset(dataset_dir, os.path.join(data_dir, dataset_zip))
                print("Dataset đã được nén lại.")
