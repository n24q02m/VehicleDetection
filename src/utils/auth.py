import os
import json


def setup_kaggle_auth():
    # Kiểm tra xem có đang chạy trên Kaggle không
    on_kaggle = "KAGGLE_URL_BASE" in os.environ

    if on_kaggle:
        print("Đang chạy trên Kaggle. Sử dụng biến môi trường cho xác thực.")

        # Lấy username và key từ biến môi trường
        username = os.environ.get("KAGGLE_USERNAME")
        key = os.environ.get("KAGGLE_KEY")

        if username and key:
            # Tạo thư mục .kaggle
            kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
            os.makedirs(kaggle_dir, exist_ok=True)
            kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")

            # Ghi tệp kaggle.json
            kaggle_credentials = {"username": username, "key": key}
            with open(kaggle_json_path, "w") as f:
                json.dump(kaggle_credentials, f)
            os.chmod(kaggle_json_path, 0o600)
            return True
        else:
            print("Không tìm thấy biến môi trường KAGGLE_USERNAME và KAGGLE_KEY.")
            return False
    else:
        # Chạy trên máy local
        kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
        kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")

        if os.path.exists(kaggle_json_path):
            print("Đã tìm thấy kaggle.json trong thư mục mặc định.")
            # Đảm bảo quyền được thiết lập đúng
            os.chmod(kaggle_json_path, 0o600)
            return True
        else:
            print(
                "Không tìm thấy kaggle.json. Vui lòng đặt nó trong thư mục ~/.kaggle/"
            )
            return False
