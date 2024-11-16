import os
import kaggle
import json
import zipfile
import shutil
import multiprocessing
from zipfile import ZipFile
from download_dataset import main as download_dataset_main
from preprocess_data import main as preprocess_data_main
from explore_dataset import main as explore_dataset_main
from augment_data import main as augment_data_main
from tqdm import tqdm

def download_file(dataset_name, destination):
    """Download dataset from Kaggle."""
    print(f"Downloading dataset {dataset_name}...")
    try:
        kaggle.api.dataset_download_files(dataset_name, path=destination, unzip=False)
        print(f"Downloaded dataset to {destination}")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def extract_member(member, zip_path, extract_to):
    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extract(member, extract_to)

def extract_zip(zip_path, extract_to):
    """Extract zip file using multiprocessing for speed."""
    print(f"Extracting {zip_path} into {extract_to}...")

    with ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.namelist()

    args = [(member, zip_path, extract_to) for member in members]

    with multiprocessing.Pool() as pool:
        list(tqdm(
            pool.starmap(extract_member, args),
            total=len(members),
            desc="Extracting"
        ))
    print("Extraction completed.")

def update_kaggle_dataset(dataset_name, folder_path):
    """Update Kaggle dataset with new version."""
    print(f"Updating Kaggle dataset {dataset_name}...")
    try:
        metadata = {
            "title": "Augmented Vehicle Detection Dataset",
            "id": f"{dataset_name}",
            "licenses": [{"name": "CC0-1.0"}]
        }
        with open(os.path.join(folder_path, "dataset-metadata.json"), "w") as f:
            json.dump(metadata, f)

        kaggle.api.dataset_create_version(folder_path, version_notes="Updated dataset")
        print("Dataset updated successfully")
        return True
    except Exception as e:
        print(f"Error updating dataset: {e}")
        return False

if __name__ == "__main__":
    dataset_dir = "./data/soict-hackathon-2024_dataset"
    dataset_name = "n24q02m/augmented-vehicle-detection-dataset"

    os.makedirs(dataset_dir, exist_ok=True)

    dataset_updated = False

    if os.path.exists(dataset_dir):
        print("Dataset directory already exists. Skipping download and extraction.")
    else:
        success = download_file(dataset_name, dataset_dir)
        if success:
            zip_files = [f for f in os.listdir(dataset_dir) if f.endswith(".zip")]
            for zip_file in zip_files:
                extract_zip(os.path.join(dataset_dir, zip_file), dataset_dir)
        else:
            print("Download failed. Running data processing scripts...")
            download_dataset_main()
            explore_dataset_main()
            preprocess_data_main()
            augment_data_main()
            dataset_updated = True

        if not os.path.exists(dataset_dir):
            print("Extraction failed or dataset directory missing. Running data processing scripts...")
            download_dataset_main()
            explore_dataset_main()
            preprocess_data_main()
            augment_data_main()
            dataset_updated = True
        else:
            print("Dataset ready.")

    # Update Kaggle dataset only if data was processed
    if dataset_updated:
        update_kaggle_dataset(dataset_name, dataset_dir)
    else:
        print("No changes made to dataset. Skipping Kaggle update.")
