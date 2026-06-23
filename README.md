# RTSP Live Stream FaceID & Domain Calibration System

Hệ thống FaceID công nghiệp tích hợp cổng kiểm soát chất lượng ảnh khuôn mặt (**Face Quality Gate**), cơ chế so khớp thích ứng (**Adaptive Threshold**) và hiệu chỉnh dịch chuyển miền đặc trưng (**Domain Calibration**) từ luồng camera RTSP trực tiếp hoặc phân cụm offline.

---

## 📌 Cấu trúc Thư mục Dự án

```text
├── data/                       # Chứa ảnh chuẩn (reference), ảnh matches và cache dữ liệu (đã ignore)
│   ├── _face_ID/               # Thư mục lưu ảnh mẫu chuẩn của nhân sự (ví dụ: duong_cropped.jpg)
│   └── stream_matches/         # Lưu ảnh gốc và ảnh cropped khuôn mặt khi so khớp thành công trực tiếp
├── docs/                       # Tài liệu thiết kế và kế hoạch nghiên cứu
├── models/                     # Thư mục chứa mô hình ONNX của InsightFace (buffalo_m) (đã ignore)
├── src/                        # Thư mục mã nguồn chính
│   ├── clustering.py           # Module phân cụm ảnh khuôn mặt (HDBSCAN)
│   ├── face_id_verifier.py     # Module core xác thực FaceID và tính toán hiệu chỉnh miền đặc trưng
│   ├── main.py                 # Pipeline offline (MinIO -> Quality Gate -> Clustering -> Verification -> Report)
│   ├── minio_face_client.py    # Client tương tác dữ liệu ảnh với MinIO
│   ├── qdrant_face_client.py   # Client truy vấn vector đặc trưng với Qdrant
│   ├── quality_gate.py         # Bộ lọc chất lượng khuôn mặt (Blur, Light, Pitch, Yaw, Roll)
│   ├── redis_client.py         # Client quản lý RAM cache cho embedding chuẩn
│   └── stream_verifier.py      # Script nhận diện thời gian thực từ camera RTSP (hỗ trợ Substream)
├── utils/                      # Thư mục tiện ích
│   └── color_normalizer.py     # Module chuẩn hóa màu sắc và phân phối ánh sáng (Reinhard, CLAHE)
├── .env                        # File cấu hình môi trường dự án (đã ignore)
└── .gitignore                  # File cấu hình Git ignore
```

---

## 🛠 Hướng dẫn Cài đặt & Chuẩn bị

### 1. Cài đặt các thư viện cần thiết
Đảm bảo bạn đang sử dụng Python 3.8+ và cài đặt các dependencies:
```bash
pip install opencv-python insightface onnxruntime numpy redis qdrant-client minio scikit-image python-dotenv hdbscan
```

### 2. Chuẩn bị Mô hình
* Tải xuống bộ mô hình `buffalo_m` từ InsightFace.
* Giải nén và đặt các tệp tin `.onnx` vào đường dẫn sau:
  `models/buffalo_m/det_2.5g.onnx`
  `models/buffalo_m/w600k_r50.onnx`

### 3. Cấu hình Ảnh chuẩn của Nhân sự
* Đặt ảnh khuôn mặt thẳng, sạch, rõ nét của nhân sự cần nhận diện vào thư mục:
  `data/_face_ID/duong_cropped.jpg` (Được cấu hình thông qua biến `FACE_REF_IMAGE_PATH`).

---

## ⚙ Cấu hình Hệ thống (`.env`)

Tạo tệp `.env` tại thư mục gốc của dự án với các thông số cấu hình dưới đây:

```ini
# Cấu hình Redis Cache
REDIS_HOST=192.168.x.x
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

# Cấu hình MinIO (Lưu trữ ảnh thô)
MINIO_ENDPOINT=192.168.x.x:9000
MINIO_EXTERNAL_ENDPOINT=192.168.x.x:9001
MINIO_ACCESS_KEY=admin
MINIO_SECRET_KEY=admin123
MINIO_BUCKET_NAME=detected-faces
MINIO_SECURE=0

# Cấu hình Vector Database Qdrant
QDRANT_HOST=192.168.x.x
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=faces

# Cấu hình Phân cụm HDBSCAN (Cho Offline Pipeline)
HDBSCAN_MIN_CLUSTER_SIZE=2
HDBSCAN_MIN_SAMPLES=10
HDBSCAN_METRIC=euclidean

# Cấu hình Chuẩn hóa màu sắc
FACE_NORMALIZE_ENABLED=1
FACE_NORMALIZE_METHOD=reinhard
FACE_REF_IMAGE_PATH=data/_face_ID/duong_cropped.jpg
FACE_NORMALIZE_OVERWRITE=1

# Cấu hình kiểm định chất lượng (Quality Gate)
QUALITY_GATE_ENABLED=1
QUALITY_MIN_SCORE=0.4
QUALITY_GOOD_SCORE=0.7

# Cấu hình ngưỡng xác thực FaceID động (Adaptive Threshold)
FACE_VERIFY_ENABLED=1
FACE_VERIFY_THRESHOLD=0.65            # Ngưỡng tĩnh mặc định
FACE_VERIFY_THRESHOLD_HIGH=0.70       # Ngưỡng nghiêm ngặt cho ảnh chất lượng trung bình
FACE_VERIFY_THRESHOLD_STANDARD=0.60   # Ngưỡng tiêu chuẩn cho ảnh chất lượng tốt

# Cấu hình luồng Live Stream RTSP
RTSP_URL=rtsp://admin:admin_123@192.168.x.x:1904/stream2  # Luồng substream tối ưu băng thông
DOWNLOAD_IMAGES=1
```

---

## 🚀 Hướng dẫn Chạy Chương trình

### Luồng 1: Xử lý và Phân cụm Offline (`src/main.py`)
Dùng để tải ảnh hàng loạt từ MinIO, lọc chất lượng, phân cụm, tính toán dịch chuyển miền đặc trưng (Domain Calibration) và xuất báo cáo markdown thống kê kết quả.
```bash
python3 src/main.py
```
*Kết quả:* Báo cáo thống kê chi tiết được xuất tại `data/verification_report_<timestamp>.md`.

### Luồng 2: Nhận diện trực tiếp từ Camera RTSP (`src/stream_verifier.py`)
Dùng để đọc trực tiếp luồng video từ camera, phát hiện và verify nhân sự thời gian thực:
```bash
python3 src/stream_verifier.py
```

💡 **Các tính năng nổi bật của luồng RTSP Live:**
1. **Skip Frames**: Chỉ xử lý 1 trong mỗi 2 frames giúp duy trì tốc độ xử lý thời gian thực, mượt mà trên CPU.
2. **Auto-Headless Mode**: Tự động phát hiện nếu môi trường terminal không hỗ trợ kết xuất màn hình (ví dụ chạy qua SSH/Term nền), hệ thống sẽ tự động tắt cửa sổ OpenCV hiển thị và ghi log chi tiết trực tiếp ra console, tránh crash ứng dụng.
3. **Double Saving**: Khi nhận diện khớp nhân sự mục tiêu, hệ thống tự động lưu đồng thời:
   * Ảnh full frame: `data/stream_matches/duong_<timestamp>.jpg`
   * Ảnh crop khuôn mặt: `data/stream_matches/cropped/duong_<timestamp>_crop.jpg`
4. **Adaptive Threshold**: Tự động áp dụng ngưỡng động tùy thuộc vào điểm chất lượng khuôn mặt tức thời được trả về từ Quality Gate.
