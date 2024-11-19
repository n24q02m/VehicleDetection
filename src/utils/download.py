import os
from pathlib import Path


def download_dataset(
    dataset_name="n24q02m/augmented-vehicle-detection-dataset",
    dataset_dir="./data/soict-hackathon-2024_dataset",
):
    """
    Download dataset if not exists locally.

    Args:
        dataset_name (str): Kaggle dataset name in format username/dataset-slug
        dataset_dir (str): Local directory to save dataset
    """
    import kaggle

    if not os.path.exists(dataset_dir):
        print(f"Downloading dataset {dataset_name}...")
        os.makedirs(dataset_dir, exist_ok=True)
        try:
            kaggle.api.dataset_download_files(
                dataset_name, path=dataset_dir, unzip=True
            )
            print("Dataset downloaded and extracted.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False
    else:
        print("Dataset directory already exists. Skipping download.")
    return True


def download_model(
    model_name="n24q02m/finetuned-vehicle-detection-model",
    model_dir="./model",
    best_model_filename="finetuned_best.pt",
    last_model_filename="finetuned_last.pt",
):
    """
    Download pre-trained model if it does not exist locally.

    Args:
        model_name (str): Kaggle model name in format username/model-slug
        model_dir (str): Local directory to save model
        best_model_filename (str): Filename for the best model
        last_model_filename (str): Filename for the last model
    """


    print(f"Downloading model {model_name}...")
    os.makedirs(model_dir, exist_ok=True)
    try:
        import kaggle

        kaggle.api.dataset_download_files(model_name, path=model_dir, unzip=True)

        # Đổi tên các tệp mô hình
        os.rename(
            os.path.join(model_dir, "best.pt"),
            os.path.join(model_dir, best_model_filename),
        )
        os.rename(
            os.path.join(model_dir, "last.pt"),
            os.path.join(model_dir, last_model_filename),
        )
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False
    return True
