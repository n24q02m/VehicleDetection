import os
import gdown
import zipfile
import json
import tempfile
from pathlib import Path
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi


class DatasetDownloader:
    def __init__(self):
        self.datasets = {
            "raw": {
                "gdrive_url": "https://drive.google.com/uc?id=1SjMOqzzKDtmkqmiesIyDy2zkEN7xGjbE",
                "kaggle_dataset": "n24q02m/raw-vehicle-detection-dataset",
                "title": "Raw Vehicle Detection Dataset",
                "local_dir": "raw_vehicle_detection",
            },
            "test": {
                "gdrive_url": "https://drive.google.com/uc?id=1BQvwhSoeDm-caCImtlbcAMzhI8MDsrCZ",
                "kaggle_dataset": "n24q02m/public-test-vehicle-detection-data",
                "title": "Public Test Vehicle Detection Data",
                "local_dir": "public_test",
            },
        }

        # Initialize Kaggle API
        self.kaggle_api = KaggleApi()
        self.kaggle_api.authenticate()

    def download_from_drive(self, dataset_key):
        """Download dataset from Google Drive"""
        dataset = self.datasets[dataset_key]

        # Create temporary directory for download
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, f"{dataset['local_dir']}.zip")

            print(f"Downloading {dataset['title']} from Google Drive...")
            try:
                gdown.download(dataset["gdrive_url"], zip_path, quiet=False)

                # Create extraction directory
                extract_dir = os.path.join(temp_dir, dataset["local_dir"])
                os.makedirs(extract_dir, exist_ok=True)

                # Extract the zip file
                print(f"Extracting {zip_path}...")
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)

                print("Extraction completed.")
                return extract_dir

            except Exception as e:
                print(f"Error downloading/extracting dataset: {e}")
                return None

    def upload_to_kaggle(self, dataset_key, local_path):
        """Upload dataset to Kaggle"""
        dataset = self.datasets[dataset_key]

        try:
            # Create dataset metadata
            metadata = {
                "title": dataset["title"],
                "id": dataset["kaggle_dataset"],
                "licenses": [{"name": "CC0-1.0"}],
            }

            # Write metadata file
            metadata_path = os.path.join(local_path, "dataset-metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            print(f"Uploading {dataset['title']} to Kaggle...")

            # Create new dataset version
            self.kaggle_api.dataset_create_version(
                folder=local_path,
                version_notes=f"Update {dataset['title']}",
                quiet=False,
                dir_mode="zip",
            )

            print(f"Successfully uploaded to Kaggle as {dataset['kaggle_dataset']}")
            return True

        except Exception as e:
            print(f"Error uploading to Kaggle: {e}")
            return False

    def process_dataset(self, dataset_key):
        """Process a single dataset - download and upload"""
        print(f"\nProcessing {self.datasets[dataset_key]['title']}...")

        # Download and extract
        local_path = self.download_from_drive(dataset_key)
        if not local_path:
            return False

        # Upload to Kaggle
        success = self.upload_to_kaggle(dataset_key, local_path)
        return success


def main():
    downloader = DatasetDownloader()

    # Process both datasets
    for dataset_key in ["raw", "test"]:
        success = downloader.process_dataset(dataset_key)
        if not success:
            print(f"Failed to process {downloader.datasets[dataset_key]['title']}")


if __name__ == "__main__":
    main()
