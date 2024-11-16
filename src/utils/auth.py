import os
import shutil


def setup_kaggle_auth():
    kaggle_json_src = "/kaggle/input/kaggle.json"
    kaggle_json_dest = "/root/.kaggle/kaggle.json"

    if os.path.exists(kaggle_json_src):
        os.makedirs("/root/.kaggle/", exist_ok=True)
        shutil.copyfile(kaggle_json_src, kaggle_json_dest)
        os.chmod(kaggle_json_dest, 0o600)
        return True
    else:
        print("kaggle.json not found. Please add it to the notebook inputs.")
        return False
