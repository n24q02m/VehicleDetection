# Hướng dẫn thiết lập cho cộng tác viên

## Yêu cầu hệ thống

- Python 3.10
- CUDA Toolkit 12.4, cuDNN 8.9.7 (optional)
- Git

## Clone dự án

```bash
git clone git@github.com:n24q02m/VehicleDetection.git
cd VehicleDetection
git checkout develop
```

## Thiết lập môi trường

```bash
conda create -n vehicle-detection python=3.10
conda activate vehicle-detection
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c conda-forge ultralytics scikit-learn
```

## [Chuẩn bị dữ liệu](./prepare_dataset.md)

1. Tải 2 file zip dữ liệu và đặt vào thư mục `./data`

2. Chạy script xử lý:

```bash
python src/data/soict-hackathon-2024_dataset.py
```

## Quy trình làm việc

1. Luôn pull code mới nhất:

```bash
git pull origin develop
```

2. Commit và push:

```bash
git add .
git commit -m "Mô tả chi tiết thay đổi" # Dùng extension Conventional Commits để thêm feat, refactor trước mỗi mô tả thay đổi
git push origin develop
```

## Lưu ý

- Đọc kỹ file tasks.md để nắm yêu cầu dự án
- Báo cáo công việc hàng ngày lúc 17h
- Ghi chép đầy đủ thử nghiệm trong docs
- Không commit trực tiếp vào branch main
