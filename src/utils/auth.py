import os
from pathlib import Path


def setup_kaggle_auth():
    # Kiểm tra xem có đang chạy trên Kaggle không
    on_kaggle = "KAGGLE_URL_BASE" in os.environ

    if on_kaggle:
        print("Đang chạy trên Kaggle. Sử dụng Kaggle Secrets cho xác thực.")
        try:
            from kaggle_secrets import UserSecretsClient

            user_secrets = UserSecretsClient()

            # Lấy username và API key từ Kaggle Secrets
            username = user_secrets.get_secret("username")
            api_key = user_secrets.get_secret("kaggle-api-key")

            if username and api_key:
                # Thiết lập biến môi trường cho Kaggle API
                os.environ["KAGGLE_USERNAME"] = username
                os.environ["KAGGLE_KEY"] = api_key

                return True
            else:
                print("Không thể lấy username hoặc API key từ Kaggle Secrets.")
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
