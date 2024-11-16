import os
import json
from pathlib import Path


def update_model(
    model_name,
    model_dir,
    title,
    license_name="CC0-1.0",
):
    """
    Update model on Kaggle.
    """
    try:
        import kaggle

        print(f"Updating model {model_name}...")

        # Ensure model_dir exists and is absolute
        model_dir = Path(model_dir).absolute()
        if not model_dir.exists():
            raise ValueError(f"Model directory {model_dir} does not exist")

        # Create metadata file
        metadata = {
            "title": title,
            "id": model_name,
            "licenses": [{"name": license_name}],
        }

        metadata_path = model_dir / "dataset-metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        try:
            # Create new version
            kaggle.api.dataset_create_version(
                folder=str(model_dir),
                version_notes="Updated model",
            )
            print("Model updated successfully")
            return True
        except Exception as e:
            print(f"Error during Kaggle API call: {e}")
            return False

    except Exception as e:
        print(f"Error updating model: {e}")
        return False
