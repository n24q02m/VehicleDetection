import os
import json
from pathlib import Path


def setup_kaggle_auth():
    # Kiểm tra xem có đang chạy trên Kaggle không
    on_kaggle = "KAGGLE_URL_BASE" in os.environ

    if on_kaggle:
        print("Đang chạy trên Kaggle. Sử dụng Kaggle Secrets cho xác thực.")
        try:
            from kaggle_secrets import UserSecretsClient

            user_secrets = UserSecretsClient()

            # Lấy thông tin xác thực từ secret "n24q02m"
            secret_value = user_secrets.get_secret("n24q02m")
            print(f"Secret value: {secret_value}")  # Debug: In giá trị của secret_value

            credentials = json.loads(secret_value)
            print(f"Credentials: {credentials}")  # Debug: In giá trị của credentials

            username = credentials.get("username")
            key = credentials.get("key")

            if username and key:
                print(f"Username: {username}")  # Debug: In giá trị của username
                print(f"Key: {key}")  # Debug: In giá trị của key

                # Tạo thư mục .kaggle
                kaggle_dir = Path.home() / ".kaggle"
                kaggle_dir.mkdir(exist_ok=True)
                kaggle_json_path = kaggle_dir / "kaggle.json"

                # Ghi tệp kaggle.json
                kaggle_credentials = {"username": username, "key": key}
                with open(kaggle_json_path, "w") as f:
                    json.dump(kaggle_credentials, f)
                os.chmod(kaggle_json_path, 0o600)
                return True
            else:
                print("Không thể lấy thông tin xác thực từ Kaggle Secrets.")
                return False

        except Exception as e:
            print(f"Lỗi khi thiết lập xác thực Kaggle: {e}")
            return False
    else:
        # Chạy trên máy local
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_json_path = kaggle_dir / "kaggle.json"

        if kaggle_json_path.exists():
            print("Đã tìm thấy kaggle.json trong thư mục mặc định.")
            kaggle_json_path.chmod(0o600)
            return True
        else:
            print(
                "Không tìm thấy kaggle.json. Vui lòng đặt nó trong thư mục ~/.kaggle/"
            )
            return False
