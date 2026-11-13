# data/

Thư mục dữ liệu (`raw/`, `processed/*`, `interim/*` không đưa lên Git, xem `.gitignore`).

| Thư mục | Vai trò | Ai ghi vào |
|---------|---------|------------|
| `raw/` | Dữ liệu gốc, **không chỉnh sửa**. `train/` và `test/`, mỗi loại trái cây một thư mục con (`freshapples`, `rottenbanana`, ...) | Bạn đặt ảnh vào |
| `interim/` | Dữ liệu trung gian: `raw_index.csv` (danh mục toàn bộ ảnh quét từ `raw/`, kèm nhãn và split) | `src.data_loader.load_splits()` |
| `processed/` | Dữ liệu sẵn sàng cho mô hình: `train.csv`, `val.csv`, `test.csv` (bản chia cố định, dùng chung cho mọi thí nghiệm) | `src.data_loader.load_splits()` |

Hai thư mục `interim/` và `processed/` được tạo tự động ở lần chạy đầu. Khi thay đổi ảnh trong `raw/`,
chạy `load_splits(refresh=True)` (hoặc xóa các file csv) để tạo lại.

Đường dẫn trong các file csv là đường dẫn tương đối so với thư mục gốc dự án, nên chuyển dự án sang máy khác vẫn dùng được.
