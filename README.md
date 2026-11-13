# AI Agent – Phân loại trạng thái trái cây (fresh / rotten)

Dự án gồm 2 phần: **thí nghiệm & huấn luyện** (notebooks + `src/`) và **REST API** (`api/`) để tích hợp khi cần. Không có giao diện web; test API bằng Postman.

## Cấu trúc thư mục

```
AI Agent/
├── data/
│   ├── raw/             train/, test/ – dữ liệu gốc, không chỉnh sửa
│   ├── processed/       train.csv, val.csv, test.csv –dữ liệu đã qua làm sạch bản chia cố định (tự sinh)
│   └── interim/         raw_index.csv – dữ liệu trung gian trong quá trình xử lý danh mục ảnh quét từ raw (tự sinh)
├── notebooks/
│   ├── 01-exploratory-data-analysis.ipynb   khám phá dữ liệu
│   ├── 02-modeling-baseline.ipynb           mô hình baseline
│   └── 03-experiments-and-selection.ipynb   chạy thí nghiệm, chốt mô hình tốt nhất
├── models/              best_model.h5, class_names.json
├── src/
│   ├── data_loader.py   đường dẫn, đọc dữ liệu, chia train/val
│   ├── features.py      tiền xử lý ảnh + augmentation
│   ├── train.py         xây model, huấn luyện (python -m src.train)
│   ├── evaluate.py      đánh giá, confusion matrix
│   └── experiments.py   chạy nhiều thí nghiệm, chọn best (python -m src.experiments)
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

## Quy trình làm việc

1. **Khám phá dữ liệu:** mở `notebooks/01-exploratory-data-analysis.ipynb`.
2. **Thí nghiệm & chốt mô hình:** chạy `notebooks/03-experiments-and-selection.ipynb`
   (hoặc `python -m src.experiments`). Có 5 thí nghiệm:

   | Tên | Nội dung |
   |-----|----------|
   | E1_cnn_scratch | CNN nhỏ tự xây, không pretrained |
   | E2_mnv2_frozen_lr1e-3 | MobileNetV2 đóng băng, lr 1e-3 (baseline) |
   | E3_mnv2_frozen_lr1e-4 | như E2, lr 1e-4 |
   | E4_mnv2_finetune_last30 | E2 + mở khoá 30 lớp cuối, fine-tune lr 1e-5 |
   | E5_mnv2_strong_aug_dropout0.5 | như E2 nhưng augmentation mạnh, dropout 0.5 |

   Thí nghiệm thắng được chọn theo **val_accuracy** (hòa thì val_loss thấp hơn), chỉ dùng tập validation.
   Tập test chỉ dùng một lần để báo cáo kết quả cuối. Kết quả nằm ở `reports/experiments.csv`,
   `reports/best_experiment.json`; model thắng được lưu vào `models/best_model.h5`.
3. **(Tuỳ chọn) Huấn luyện lại đầy đủ** bằng cấu hình thắng: `python -m src.train`
4. **Chạy API** (tại thư mục gốc):
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```
   Swagger: http://localhost:8000/docs · Test nhanh: `curl -F "file=@anh.jpg" http://localhost:8000/predict`

## Test
```bash
python -m pytest
```

## Docker
```bash
docker build -t fruit-api .
docker run -p 8000:8000 fruit-api
```
Image dùng `models/best_model.h5` nên cần có model trước khi build.

## Test API bằng Postman
Import 2 file trong `docs/postman/` rồi làm theo [docs/API.md](docs/API.md).
