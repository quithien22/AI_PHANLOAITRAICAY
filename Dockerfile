FROM python:3.10-slim

WORKDIR /app

# Chỉ cài thư viện cần cho API (không cần pandas/matplotlib/seaborn của phần train)
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    pillow \
    numpy \
    scikit-learn \
    joblib

COPY api ./api
COPY src ./src
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
