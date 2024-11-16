import os
import json


def setup_kaggle_auth():
    on_kaggle = "KAGGLE_URL_BASE" in os.environ

    if on_kaggle:
        print("Đang chạy trên Kaggle. Sử dụng biến môi trường cho xác thực.")

        username = os.environ.get("KAGGLE_USERNAME")
        key = os.environ.get("KAGGLE_KEY")

        if username and key:
            kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
            os.makedirs(kaggle_dir, exist_ok=True)
            kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")

            kaggle_credentials = {"username": username, "key": key}
            with open(kaggle_json_path, "w") as f:
                json.dump(kaggle_credentials, f)

            os.chmod(kaggle_json_path, 0o600)
            return True
        else:
            print("Không tìm thấy biến môi trường KAGGLE_USERNAME và KAGGLE_KEY.")
            return False
    else:
        # Handle local execution
        kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
        kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")

        if os.path.exists(kaggle_json_path):
            print("Đã tìm thấy kaggle.json trong thư mục mặc định.")
            os.chmod(kaggle_json_path, 0o600)
            return True
        else:
            print("Không tìm thấy kaggle.json. Vui lòng thêm nó vào thư mục ~/.kaggle/")
            return False
