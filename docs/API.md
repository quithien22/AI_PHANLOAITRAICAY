# Tài liệu API – Fruit Ripeness Classifier

Base URL (local): `http://localhost:8000`
Swagger tự động: `http://localhost:8000/docs`

## Endpoints

| Method | Path       | Mô tả                                      |
|--------|------------|--------------------------------------------|
| GET    | `/health`  | Kiểm tra API sống và model đã nạp          |
| GET    | `/classes` | Danh sách nhãn của model                   |
| POST   | `/predict` | Upload 1 ảnh, trả về nhãn + độ tin cậy     |

### GET /health
```json
{ "status": "ok", "model_loaded": true }
```

### GET /classes
```json
{ "classes": ["fresh", "rotten"] }
```

### POST /predict
- Body: `multipart/form-data`, key **`file`** (kiểu File)
- Định dạng cho phép: JPG, PNG, WEBP, BMP; tối đa 10 MB

Thành công (200):
```json
{
  "label": "fresh",
  "confidence": 97.43,
  "probabilities": { "fresh": 97.43, "rotten": 2.57 }
}
```

Lỗi:

| Mã  | Khi nào                                  |
|-----|------------------------------------------|
| 400 | File rỗng hoặc không phải ảnh hợp lệ     |
| 413 | Ảnh lớn hơn 10 MB                        |
| 415 | Định dạng file không được hỗ trợ         |
| 422 | Thiếu field `file`                       |
| 500 | Lỗi nội bộ khi chạy model                |

Ví dụ lỗi: `{ "detail": "Ảnh quá lớn (tối đa 10 MB)." }`

## Test bằng Postman

1. Chạy API (xem README): `uvicorn api.main:app --port 8000`
2. Postman → **Import** 2 file trong `docs/postman/`:
   - `Fruit_API.postman_collection.json`
   - `Local.postman_environment.json`
3. Góc phải trên chọn environment **Fruit API - Local**.
4. Chạy lần lượt các request. Với request 3 và 4, vào tab **Body** → dòng `file` → **Select Files** để chọn ảnh (request 3) hoặc file không phải ảnh (request 4).
5. Mỗi request đã có sẵn test tự động, xem kết quả ở tab **Test Results**.

Đổi địa chỉ server (VD: chạy Docker trên máy khác): sửa biến `base_url` trong environment.

## Tích hợp khi cần

Gọi từ code bất kỳ, ví dụ Python:
```python
import requests

with open("anh.jpg", "rb") as f:
    r = requests.post("http://localhost:8000/predict", files={"file": ("anh.jpg", f, "image/jpeg")})
print(r.json())
```

Hoặc curl:
```bash
curl -F "file=@anh.jpg" http://localhost:8000/predict
```

Nếu gọi API từ trình duyệt ở domain khác, đặt biến môi trường `ALLOW_ORIGINS` (phân tách bằng dấu phẩy), ví dụ:
`ALLOW_ORIGINS=https://app.example.com,http://localhost:3000`
