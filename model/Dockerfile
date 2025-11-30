# Sử dụng image hỗ trợ tốt cho AI/Torch
# Nếu dùng GPU NVIDIA, nên dùng: FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
# Ở đây dùng python slim cơ bản, nếu có GPU thì cần cài thêm torch version cuda
FROM python:3.11-slim

WORKDIR /app

# Cài đặt git (để load model từ huggingface hoặc cài package từ source)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements và cài đặt
COPY requirements.txt .
# Lưu ý: requirements.txt của worker nên chứa torch, transformers, accelerate, qwen-vl-utils, ...
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Biến môi trường để cache model huggingface vào thư mục mà chúng ta sẽ mount volume
ENV HF_HOME=/app/hf_cache

# Lệnh chạy Celery Worker (Trong Docker Linux không cần --pool=solo)
CMD ["celery", "-A", "tasks", "worker", "--loglevel=info"]