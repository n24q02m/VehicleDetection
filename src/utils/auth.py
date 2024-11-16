import os
import json


def setup_kaggle_auth():
    """Set up Kaggle authentication using environment variables."""
    try:
        # Check if we're running on Kaggle
        on_kaggle = "KAGGLE_URL_BASE" in os.environ

        if on_kaggle:
            # Verify that secrets are available
            if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
                print(
                    "Kaggle secrets not found. Please configure them in the notebook settings."
                )
                return False

            # Set up authentication
            kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
            os.makedirs(kaggle_dir, exist_ok=True)
            kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")

            kaggle_creds = {
                "username": os.environ["KAGGLE_USERNAME"],
                "key": os.environ["KAGGLE_KEY"],
            }

            with open(kaggle_json_path, "w") as f:
                json.dump(kaggle_creds, f)
            os.chmod(kaggle_json_path, 0o600)

            return True
        else:
            # Running locally - use existing kaggle.json
            kaggle_json_path = os.path.join(
                os.path.expanduser("~"), ".kaggle", "kaggle.json"
            )
            if os.path.exists(kaggle_json_path):
                os.chmod(kaggle_json_path, 0o600)
                return True
            else:
                print("kaggle.json not found in ~/.kaggle/")
                return False

    except Exception as e:
        print(f"Error setting up Kaggle authentication: {e}")
        return False
