# Hướng dẫn chuẩn bị dữ liệu

## Các bước thực hiện

1. Tải xuống 2 file zip dữ liệu:
   - [train_old_20241016.zip](https://drive.google.com/file/d/1yMfAmhUdHHyvhtc4T_H-3NlHkzdouHZq/view?usp=sharing)
   - [train_20241023.zip](https://drive.google.com/file/d/1Z7Gb_Jv51yDHiN-jHHSsND3TnAIbdqpT/view?usp=sharing)

2. Đặt 2 file zip vừa tải vào thư mục `./data`

3. Chạy script Python để xử lý dữ liệu:

```bash
python soict-hackathon-2024_dataset.py
```

Script sẽ thực hiện các công việc sau:

- Giải nén 2 file zip
- Đổi tên các file trong tập dữ liệu cũ với hậu tố "(old)"
- Tạo cấu trúc thư mục chuẩn cho YOLOv8
- Xử lý lại nhãn trong các file txt (chuyển nhãn 4,5,6,7 thành 0,1,2,3)
- Tách tập validation (20% dữ liệu)
- Tạo file data.yaml
- Dọn dẹp các thư mục tạm

## Cấu trúc thư mục sau khi chạy xong

```text
soict-hackathon-2024_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

## Các lớp trong tập dữ liệu

- 0: motorbike
- 1: car  
- 2: bus
- 3: truck
