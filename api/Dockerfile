# Sử dụng Python nhẹ cho API
FROM python:3.11-slim

WORKDIR /app

# Cài đặt các dependencies cần thiết cho OS (nếu cần)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port cho FastAPI
EXPOSE 8000

# Chạy server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]