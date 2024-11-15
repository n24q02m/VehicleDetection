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
from tqdm import tqdm

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


def extract_member(member, zip_path, extract_to):
    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extract(member, extract_to)

def extract_zip(zip_path, extract_to):
    """Extract zip file using multiprocessing for speed."""
    print(f"Extracting {zip_path} into {extract_to}...")

    with ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.namelist()

    # Chuẩn bị danh sách tham số cho multiprocessing
    args = [(member, zip_path, extract_to) for member in members]

    # Sử dụng multiprocessing Pool để tăng tốc độ giải nén
    with multiprocessing.Pool() as pool:
        list(tqdm(
            pool.starmap(extract_member, args),
            total=len(members),
            desc="Extracting"
        ))
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

    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Check if dataset directory already exists
    if os.path.exists(dataset_dir):
        print("Thư mục dataset đã tồn tại. Bỏ qua các bước xử lý và tăng cường dữ liệu.")
    else:
        # Check if zip file exists 
        zip_paths = [os.path.join(".", dataset_zip), os.path.join(data_dir, dataset_zip)]
        zip_exists = any(os.path.exists(path) and os.path.getsize(path) > 0 for path in zip_paths)

        if zip_exists:
            # Use existing zip file after validating
            zip_path = next(path for path in zip_paths if os.path.exists(path))
            print(f"Tìm thấy tệp {zip_path}.")
            
            # Validate zip file
            try:
                with ZipFile(zip_path, 'r') as zf:
                    # Test zip file validity
                    bad_file = zf.testzip()
                    if bad_file:
                        raise zipfile.BadZipFile(f"Bad file found: {bad_file}")
                    
                # If valid, extract it
                extract_zip(zip_path, data_dir)
                print("Dataset đã sẵn sàng.")
                
            except zipfile.BadZipFile:
                print(f"Tệp {zip_path} không phải là file zip hợp lệ hoặc bị hỏng.")
                print("Tiếp tục chạy các script xử lý dữ liệu...")
                # Run data processing scripts
                print("Bắt đầu quá trình xử lý và tăng cường dữ liệu...")
                download_dataset_main()
                explore_dataset_main() 
                preprocess_data_main()
                augment_data_main()
                print("Hoàn thành quá trình xử lý và tăng cường dữ liệu.")
                
                # Compress dataset after processing
                zip_dataset(dataset_dir, os.path.join(data_dir, dataset_zip))
                print("Dataset đã được nén lại.")
        else:
            # If Google Drive URL is empty, skip downloading and proceed with processing
            print("Bỏ qua bước tải dataset từ Google Drive.")
            print("Tiếp tục chạy các script xử lý dữ liệu...")
            
            # Run data processing scripts
            print("Bắt đầu quá trình xử lý và tăng cường dữ liệu...")
            download_dataset_main()
            explore_dataset_main()
            preprocess_data_main()
            augment_data_main()
            print("Hoàn thành quá trình xử lý và tăng cường dữ liệu.")

            # Compress dataset after processing
            zip_dataset(dataset_dir, os.path.join(data_dir, dataset_zip))
            print("Dataset đã được nén lại.")
