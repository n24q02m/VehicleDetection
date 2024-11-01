# Nhiệm vụ Vehicle Detection

## 1. Mục tiêu chính

- Cải thiện mô hình phát hiện phương tiện hiện có của viện
- Tối ưu hóa cho thiết bị biên (camera)
- Đảm bảo khả năng triển khai thực tế

## 2. Yêu cầu kỹ thuật

### 2.1. Mô hình

- Sử dụng các mô hình YOLO
- Kích thước tối đa YOLOv8m (ưu tiên bản s trở xuống)
- Hạn chế tối đa việc sử dụng attention
- Có thể điều chỉnh các layer để:
  - Cải thiện độ chính xác
  - Tăng tốc độ inference

### 2.2. Dữ liệu

- Sử dụng dữ liệu từ cuộc thi SoICT Hackathon 2024
- Được phép:
  - Tăng cường dữ liệu (data augmentation)
  - Tạo dữ liệu tổng hợp (synthetic data)
- Không được sử dụng dữ liệu ngoài

### 2.3. Quản lý mã nguồn

#### Git/GitHub

- Tổ chức repository với các branch:
  - main: code ổn định
  - develop: code đang phát triển
  - feature/*: các tính năng mới
- Tạo .gitignore phù hợp
- Code review giữa các thành viên

#### Docker

- Dockerfile cho môi trường phát triển
- docker-compose.yml cho các service
- requirements.txt cho dependencies
- README.md với hướng dẫn cài đặt và sử dụng

## 3. Quy trình làm việc

### 3.1. Báo cáo

- Báo cáo hàng ngày lúc 17h
- Ghi chép đầy đủ trong docs:
  - Kết quả thử nghiệm
  - Phân tích lỗi
  - Cải tiến mô hình
  - Vấn đề gặp phải
- Có thể có họp/báo cáo trực tiếp hàng tuần

### 3.2. Làm việc nhóm

- Phối hợp chặt chẽ giữa các thành viên
- Phân chia công việc rõ ràng
- Trao đổi thường xuyên
- Code review

## 4. Đánh giá

### 4.1. Metric

- MAP50 cho các lớp:
  - 0: xe máy
  - 1: xe ô tô con
  - 2: xe khách
  - 3: xe container

### 4.2. Tiêu chí

- So sánh với baseline model của viện
- Đánh giá tốc độ inference
- Kiểm thử trong các điều kiện:
  - Ban ngày
  - Ban đêm
  - Điều kiện khó (mưa, loá đèn)

## 5. Timeline

- Đăng ký cuộc thi: trước 23h59 15/11/2024
- Nộp danh sách pre-trained models: 17/11/2024
- Vòng sơ khảo: đến 22/11/2024
- Vòng chung khảo online: 01/12/2024
- Chung kết: 21/12/2024

## 6. Lưu ý quan trọng

- Ưu tiên yêu cầu của viện hơn yêu cầu cuộc thi
- Tập trung vào khả năng triển khai thực tế
- Đảm bảo mô hình có thể chạy hiệu quả trên camera
- Liên hệ ngay khi cần thông tin hoặc hỗ trợ
