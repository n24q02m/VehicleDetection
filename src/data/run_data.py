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


if __name__ == "__main__":
    dataset_zip = "soict-hackathon-2024_dataset.zip"
    data_dir = "./data"
    dataset_dir = os.path.join(data_dir, "soict-hackathon-2024_dataset")

    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Check if dataset directory already exists
    if os.path.exists(dataset_dir):
        print("Dataset directory already exists. Skipping download and extraction.")
    else:
        if os.path.exists(os.path.join(data_dir, dataset_zip)):
            print(f"Found {dataset_zip}. Extracting...")
            extract_zip(os.path.join(data_dir, dataset_zip), data_dir)
        else:
            zip_url = "https://drive.google.com/uc?id=19gL2L2LUjX8A0uxKvYJ5zsYD7gRK02QO"
            success = download_file(zip_url, os.path.join(data_dir, dataset_zip))
            if success:
                extract_zip(os.path.join(data_dir, dataset_zip), data_dir)
            else:
                print("Download failed. Running data processing scripts...")
                download_dataset_main()
                explore_dataset_main()
                preprocess_data_main()
                augment_data_main()

        # Verify extraction
        if not os.path.exists(dataset_dir):
            print("Extraction failed or dataset directory missing. Running data processing scripts...")
            download_dataset_main()
            explore_dataset_main()
            preprocess_data_main()
            augment_data_main()
        else:
            print("Dataset ready.")

    # Compress dataset after processing
    zip_dataset(dataset_dir, os.path.join(data_dir, dataset_zip))
    print("Dataset compressed.")
