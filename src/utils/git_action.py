import os
import json
from pathlib import Path


def setup_git_credentials():
    """
    Set up Git credentials based on environment (Kaggle vs local).
    Returns credentials dict with username, email and token.
    """

    # Check if running on Kaggle
    is_kaggle = "KAGGLE_URL_BASE" in os.environ

    if is_kaggle:
        try:
            from kaggle_secrets import UserSecretsClient

            user_secrets = UserSecretsClient()

            credentials = {
                "username": user_secrets.get_secret("username"),
                "email": user_secrets.get_secret("email"),
                "token": user_secrets.get_secret("github-token"),
            }

            if not all(credentials.values()):
                raise ValueError("Could not get all required secrets from Kaggle")

        except Exception as e:
            print(f"Error getting Kaggle secrets: {e}")
            return None

    else:
        # Local environment - read from config file
        config_path = Path.home() / "githubconfig.json"

        try:
            if not config_path.exists():
                raise FileNotFoundError(f"GitHub config not found at {config_path}")

            with open(config_path) as f:
                credentials = json.load(f)

            required = ["username", "email", "token"]
            if not all(k in credentials for k in required):
                raise ValueError(f"GitHub config must contain: {required}")

        except Exception as e:
            print(f"Error reading GitHub config: {e}")
            return None

    return credentials


def git_push_changes(message="Update repository"):
    """
    Configure Git credentials and push changes to remote.
    """
    credentials = setup_git_credentials()
    if not credentials:
        print("Failed to set up Git credentials")
        return False

    try:
        # Configure Git
        os.system(f'git config --global user.name "{credentials["username"]}"')
        os.system(f'git config --global user.email "{credentials["email"]}"')

        # Update remote URL with token
        remote_url = f'https://{credentials["username"]}:{credentials["token"]}@github.com/{credentials["username"]}/VehicleDetection.git'
        os.system(f"git remote remove origin 2>/dev/null || true")
        os.system(f"git remote add origin {remote_url}")

        # Add, commit and push
        os.system("git add .")
        os.system(f'git commit -m "{message}"')
        os.system("git push origin main")

        return True

    except Exception as e:
        print(f"Error pushing to GitHub: {e}")
        return False
