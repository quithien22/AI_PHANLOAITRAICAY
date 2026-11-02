# AI Agent – Phân loại trạng thái trái cây (Xanh / Chín / Hư hỏng)

```
AI Agent/
├── model/      train_model.py, requirements.txt
├── backend/    main.py, requirements.txt
└── frontend/   FruitClassifier.jsx, FruitClassifier.css
```

## 1. Huấn luyện
```bash
cd model
pip install -r requirements.txt
# Chuẩn bị dataset/unripe, dataset/ripe, dataset/rotten (mỗi thư mục chứa ảnh)
python train_model.py
```
Kết quả: `fruit_model.h5`, `class_names.json`, `training_history.png`, `confusion_matrix.png`.

## 2. Backend
Copy `fruit_model.h5` và `class_names.json` vào thư mục `backend/`.
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Test: `curl -F "file=@anh.jpg" http://localhost:8000/predict`
Docs tự động: http://localhost:8000/docs

## 3. Frontend
**Next.js:** đặt 2 file vào `components/`, rồi `import FruitClassifier from "@/components/FruitClassifier"` trong `app/page.jsx`.
**Vite:** đặt vào `src/`, import trong `App.jsx`.
```bash
npm run dev   # mở http://localhost:3000 (Next.js)
```
