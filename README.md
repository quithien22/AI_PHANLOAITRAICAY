# AI Agent – Phân loại trạng thái trái cây (fresh / rotten) bằng học máy

Dự án gồm 2 phần: **thí nghiệm & huấn luyện** (notebooks + `src/`) và **REST API** (`api/`) để tích hợp khi cần. Không có giao diện web; test API bằng Postman.

Cách làm là **học máy cổ điển (scikit-learn)**, không dùng mạng nơ-ron:
ảnh → trích xuất 50 đặc trưng thủ công (màu HSV, thống kê màu, tỉ lệ vùng tối/nâu, texture) → chuẩn hoá → thuật toán phân loại.

## Cấu trúc thư mục

```
AI Agent/
├── data/
│   ├── raw/             train/, test/ – dữ liệu gốc, không chỉnh sửa
│   ├── processed/       train.csv, val.csv, test.csv (bản chia cố định) và features_*.npz (đặc trưng đã trích xuất, tự sinh)
│   └── interim/         raw_index.csv – danh mục ảnh quét từ raw (tự sinh)
├── notebooks/
│   ├── 01-exploratory-data-analysis.ipynb   khám phá dữ liệu
│   ├── 02-modeling-baseline.ipynb           baseline Logistic Regression
│   └── 03-experiments-and-selection.ipynb   so sánh 6 thuật toán, chốt mô hình tốt nhất
├── models/              best_model.joblib, class_names.json
├── src/
│   ├── data_loader.py   đường dẫn, đọc dữ liệu, chia train/val
│   ├── features.py      trích xuất đặc trưng từ ảnh
│   ├── train.py         các thuật toán, build/fit model (python -m src.train)
│   ├── evaluate.py      đánh giá, confusion matrix, độ quan trọng đặc trưng
│   └── experiments.py   chạy nhiều thuật toán, chọn best (python -m src.experiments)
├── api/                 main.py – FastAPI
├── docs/                API.md, postman/ (collection + environment)
├── reports/             biểu đồ, experiments.csv, best_experiment.json
├── tests/               unit test (pytest)
├── requirements.txt
├── setup.py
├── Dockerfile
└── README.md
```

## Cài đặt
```bash
pip install -r requirements.txt
# hoặc cài như package:  pip install -e ".[train,dev]"
```

## Đặc trưng ảnh (50 đặc trưng)

| Nhóm | Số lượng | Ý nghĩa |
|------|----------|---------|
| Histogram HSV (H 16 bin, S 8 bin, V 8 bin) | 32 | Phân bố màu: trái hỏng thường nâu, tối, xỉn |
| Mean và std của R,G,B,H,S,V | 12 | Màu trung bình và độ đồng đều của màu |
| Tỉ lệ điểm ảnh tối, tỉ lệ điểm ảnh nâu | 2 | Vết thâm, vùng hỏng |
| Gradient trung bình/độ lệch chuẩn theo x, y | 4 | Texture bề mặt |

## Quy trình làm việc

1. **Khám phá dữ liệu:** mở `notebooks/01-exploratory-data-analysis.ipynb`.
2. **Thí nghiệm & chốt mô hình:** chạy `notebooks/03-experiments-and-selection.ipynb`
   (hoặc `python -m src.experiments`). Có 6 thí nghiệm:

   | Tên | Thuật toán |
   |-----|-----------|
   | E1_logistic_regression | Logistic Regression (baseline) |
   | E2_knn_k5 | K-Nearest Neighbors, k = 5 |
   | E3_naive_bayes | Gaussian Naive Bayes |
   | E4_decision_tree_depth10 | Decision Tree, max_depth = 10 |
   | E5_random_forest_200 | Random Forest, 200 cây |
   | E6_svm_rbf_C10 | SVM kernel RBF, C = 10 |

   Thí nghiệm thắng được chọn theo **val_accuracy** (hòa thì val_loss thấp hơn), chỉ dùng tập validation.
   Tập test chỉ dùng một lần để báo cáo kết quả cuối. Kết quả nằm ở `reports/experiments.csv`,
   `reports/best_experiment.json`; model thắng được lưu vào `models/best_model.joblib`.
3. **(Tuỳ chọn) Huấn luyện lại** bằng cấu hình thắng: `python -m src.train`
4. **Chạy API** (tại thư mục gốc):
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```
   Swagger: http://localhost:8000/docs · Test nhanh: `curl -F "file=@anh.jpg" http://localhost:8000/predict`

> Lần chạy đầu tiên cần vài phút để trích xuất đặc trưng; kết quả được lưu ở `data/processed/features_*.npz` cho các lần sau.
> Các file model/báo cáo từ bản deep learning cũ (`best_model.h5`, `training_history.png`...) không còn được dùng và có thể xoá.

## Test
```bash
python -m pytest
```

## Docker
```bash
docker build -t fruit-api .
docker run -p 8000:8000 fruit-api
```
Image dùng `models/best_model.joblib` nên cần có model trước khi build.

## Test API bằng Postman
Import 2 file trong `docs/postman/` rồi làm theo [docs/API.md](docs/API.md).
