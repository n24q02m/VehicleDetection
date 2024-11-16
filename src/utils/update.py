import os
import json
import kaggle


def update_model(
    model_name,
    model_dir,
    version_notes="Updated model",
    title=None,
    license_name="CC0-1.0",
):
    """
    Update model on Kaggle.

    Args:
        model_name (str): Model name in format "username/model-slug"
        model_dir (str): Directory containing model files
        version_notes (str): Notes for this version
        title (str, optional): Model title. If None, will use last part of model_name
        license_name (str): License name. Default is "CC0-1.0"
    """
    try:
        print(f"Updating model {model_name}...")

        # Create metadata file
        metadata = {
            "title": title or model_name.split("/")[-1].replace("-", " ").title(),
            "id": model_name,
            "licenses": [{"name": license_name}],
        }

        metadata_path = os.path.join(model_dir, "dataset-metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Update model on Kaggle
        kaggle.api.dataset_create_version(model_dir, version_notes=version_notes)
        print("Model updated successfully")
        return True

    except Exception as e:
        print(f"Error updating model: {e}")
        return False
