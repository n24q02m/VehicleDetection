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
        list(
            tqdm(
                pool.starmap(extract_member, args),
                total=len(members),
                desc="Extracting",
            )
        )
        list(
            tqdm(
                pool.starmap(extract_member, args),
                total=len(members),
                desc="Extracting",
            )
        )
    print("Extraction completed.")


def update_kaggle_dataset(dataset_name, folder_path):
    """Update Kaggle dataset with new version."""
    print(f"Updating Kaggle dataset {dataset_name}...")
    try:
        metadata = {
            "title": "Augmented Vehicle Detection Dataset",
            "id": f"{dataset_name}",
            "licenses": [{"name": "CC0-1.0"}],
            "licenses": [{"name": "CC0-1.0"}],
        }
        with open(os.path.join(folder_path, "dataset-metadata.json"), "w") as f:
            json.dump(metadata, f)

        kaggle.api.dataset_create_version(folder_path, version_notes="Updated dataset")
        print("Dataset updated successfully")
        return True
    except Exception as e:
        print(f"Error updating dataset: {e}")
        return False


def clean_directories():
    """Clean dataset and explore directories."""
    dirs_to_clean = [
        "./data/soict-hackathon-2024_dataset",
        "./data/extracted_data",
        "./runs/explore_dataset",
    ]
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"Removed {dir_path}")


def main(use_existing_dataset=True):
    """
    Main function for dataset processing pipeline.

    Args:
        use_existing_dataset (bool): Whether to use existing dataset (True) or create new one (False)
    """
    dataset_dir = "./data/soict-hackathon-2024_dataset"
    dataset_name = "n24q02m/augmented-vehicle-detection-dataset"
    explore_dir = "./runs/explore_dataset"

    if not use_existing_dataset:
        # Remove existing dataset and explore directories
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
        if os.path.exists(explore_dir):
            shutil.rmtree(explore_dir)

    if use_existing_dataset:
        if os.path.exists(dataset_dir):
            print("Dataset directory already exists. Using existing dataset.")
        else:
            print("Dataset directory not found. Attempting to download from Kaggle...")
            os.makedirs(dataset_dir, exist_ok=True)
            success = download_file(dataset_name, dataset_dir)

            if success:
                zip_files = [f for f in os.listdir(dataset_dir) if f.endswith(".zip")]
                for zip_file in zip_files:
                    extract_zip(os.path.join(dataset_dir, zip_file), dataset_dir)
                print("Dataset downloaded and extracted successfully.")
            else:
                print("Error: Failed to download dataset from Kaggle.")
                exit(1)
    else:
        print("Creating new dataset...")
        # Clean existing directories
        clean_directories()

        # Create new dataset
        os.makedirs(dataset_dir, exist_ok=True)

        # Run data processing pipeline
        print("Running data processing pipeline...")
        download_dataset_main()  # Download raw data
        explore_dataset_main()  # Explore dataset
        preprocess_data_main()  # Preprocess data
        augment_data_main()  # Augment data

        # Update dataset on Kaggle
        print("Uploading new dataset to Kaggle...")
        update_kaggle_dataset(dataset_name, dataset_dir)


if __name__ == "__main__":
    # Set configuration
    use_existing_dataset = False  # Set to False to create new dataset
    main(use_existing_dataset)
