import os
import json
from pathlib import Path


def setup_kaggle_auth():
    """
    Set up Kaggle authentication for both local and Kaggle environments.
    For local: Uses environment variables or existing kaggle.json
    For Kaggle: Uses Kaggle Secrets
    """
    try:
        kaggle_dir = os.path.expanduser("~/.kaggle")
        kaggle_json = os.path.join(kaggle_dir, "kaggle.json")

        # Check if running on Kaggle
        if "KAGGLE_URL_BASE" in os.environ:
            print("Running on Kaggle - Using Kaggle Secrets")
            from kaggle_secrets import UserSecretsClient

            user_secrets = UserSecretsClient()
            kaggle_username = user_secrets.get_secret("kaggle_username")
            kaggle_key = user_secrets.get_secret("kaggle_key")

            # Save to /root/.config/kaggle/kaggle.json for Kaggle environment
            root_config = "/root/.config/kaggle"
            os.makedirs(root_config, exist_ok=True)
            with open(os.path.join(root_config, "kaggle.json"), "w") as f:
                json.dump({"username": kaggle_username, "key": kaggle_key}, f)
            os.chmod(os.path.join(root_config, "kaggle.json"), 0o600)

        else:
            print("Running locally - Checking Kaggle credentials")
            # Check if credentials exist in environment variables
            if "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ:
                print("Using Kaggle credentials from environment variables")
                kaggle_username = os.environ["KAGGLE_USERNAME"]
                kaggle_key = os.environ["KAGGLE_KEY"]

                # Create kaggle.json if it doesn't exist
                os.makedirs(kaggle_dir, exist_ok=True)
                if not os.path.exists(kaggle_json):
                    with open(kaggle_json, "w") as f:
                        json.dump({"username": kaggle_username, "key": kaggle_key}, f)
                    os.chmod(kaggle_json, 0o600)

            # Check if kaggle.json exists
            elif os.path.exists(kaggle_json):
                print("Using existing Kaggle credentials from kaggle.json")
            else:
                raise Exception(
                    "No Kaggle credentials found. Please set KAGGLE_USERNAME and KAGGLE_KEY "
                    "environment variables or place kaggle.json in ~/.kaggle/"
                )

        return True

    except Exception as e:
        print(f"Error setting up Kaggle authentication: {e}")
        return False
