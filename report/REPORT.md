# Báo cáo Day 6: Độ nhạy calibration LiDAR-camera

- **Họ tên:** Nguyễn Khánh Đỗ (theo tên repo, đang chờ xác nhận)
- **MSSV:** 2A202602687
- **Lớp:** Track 4 (đang chờ tên lớp cụ thể)
- **Link repo:** https://github.com/Diohus/NguyenKhanhDo-2A202602687-Track4-Day21
- **Topic:** A — LiDAR-camera projection QA
- **Dataset:** data/synthetic, data/kitti_mini, data/nuscenes_mini_subset
- **Các frame đã dùng:** synthetic 000000–000004; toàn bộ 20 frame KITTI và 80 frame nuScenes.

## 1. Claim
Giả thuyết CP1: yaw 3° làm giảm tỉ lệ điểm thuộc box 3D chiếu đúng vào box 2D so với yaw 0° trên cả hai dataset thật; score box phát hiện được drift nhưng có thể bỏ sót khi vật lớn hoặc ít điểm.

## 2. Evidence
CP0: hai dataset đã qua verify_data; data_health.csv có đủ 5 frame synthetic.

## 3. Failure case
Sẽ chọn từ benchmark chạy thật.

## 4. Khuyến nghị nếu triển khai thật
Use-case: giám sát calibration trong ADAS.

## 5. Cách chạy lại
```bash
python tools/verify_data.py --data-root data/kitti_mini
python tools/verify_data.py --data-root data/nuscenes_mini_subset
python -m starter.data_health --data-root data/synthetic
```

## 6. Khai báo sử dụng AI
Codex hỗ trợ đọc yêu cầu, cài đặt, thiết kế thí nghiệm và báo cáo; chỉ dùng số liệu chạy thật. Người học cần tự đọc lại và kiểm chứng trước vấn đáp.