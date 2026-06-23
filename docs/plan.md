# Kế hoạch triển khai đợt này (2 tuần)

## 1) Mục tiêu
- Giảm sai số FaceID trong điều kiện ánh sáng phức tạp (trưa/chiều/tối, ngược sáng, đèn mạnh).
- Giữ độ trễ pipeline ở mức chấp nhận được.

## 2) Phạm vi
- Thêm `quality gate` trước bước verify.
- Thêm `adaptive threshold` theo chất lượng ảnh.
- POC CORAL trên embedding (không thay thế hoàn toàn bước color).
- Đánh giá A/B theo camera và khung giờ.

## 3) Kế hoạch theo tuần

### Tuần 1 — Baseline + Quality Gate
- Định nghĩa `quality_score` từ các tín hiệu:
  - Landmark hợp lệ (mắt, mũi, miệng + alignment thành công)
  - Độ nét (blur/Laplacian variance)
  - Độ sáng và tương phản
  - Pose (yaw/pitch)
  - Kích thước khuôn mặt (bbox)
- Rule quyết định:
  - `quality_score < Q_min` → Reject/Recapture
  - `Q_min <= quality_score < Q_good` → Verify với ngưỡng cao hơn
  - `quality_score >= Q_good` → Verify với ngưỡng chuẩn
- Ghi log đầy đủ: `quality_score`, lý do reject, similarity, quyết định cuối.

### Tuần 2 — CORAL POC + Calibration
- Tạo thống kê domain theo camera/time-slot: `mu`, `cov`.
- Thêm nhánh verify có CORAL và fallback khi thiếu mẫu.
- Chạy A/B:
  - Baseline
  - Quality gate
  - Quality gate + CORAL
- Chốt ngưỡng theo mục tiêu FAR/FRR.

## 4) KPI chấp nhận
- Giảm FRR/False Reject trong điều kiện thiếu sáng.
- Giảm False Match trên ảnh chất lượng thấp.
- Tỷ lệ reject do quality nằm trong ngưỡng vận hành cho phép.
- Latency tăng không quá mức mục tiêu (ví dụ < 15%).

## 5) Rủi ro và đối sách
- Ít dữ liệu theo camera/khung giờ → bật fallback về pipeline hiện tại.
- Domain drift theo thời gian → cập nhật thống kê định kỳ.
- Gate quá chặt gây reject nhiều → tinh chỉnh `Q_min`, `Q_good` theo dữ liệu thực tế.

## 6) Đầu ra cuối đợt
- Báo cáo A/B theo camera và khung giờ.
- Bộ ngưỡng vận hành đề xuất (`Q_min`, `Q_good`, `verify_threshold` theo nhóm chất lượng).
- Checklist triển khai production và kế hoạch theo dõi drift.
