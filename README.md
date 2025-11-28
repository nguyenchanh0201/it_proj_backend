Mermaid Diagram Generator with Qwen3-VL (Microservices)

Dự án này sử dụng kiến trúc Microservices để tách biệt phần API Gateway (FastAPI) nhẹ và phần AI Worker (Celery + Qwen3-VL) nặng. Hệ thống sử dụng Redis làm Message Broker để giao tiếp.

🚀 Bước 1: Khởi động Redis (Message Broker)

Hệ thống cần Redis để truyền tải task giữa API và Worker.

Cách 1: Dùng Docker (Khuyên dùng)

docker run -d -p 6379:6379 --name redis-broker redis

Cách 2: Dùng Redis cài trực tiếp trên máy
Đảm bảo Redis server đang chạy ở cổng 6379.

🧠 Bước 2: Cài đặt và Chạy AI Worker (model/)

Service này chịu trách nhiệm tải Model Qwen3-VL (4GB~) và xử lý tạo code Mermaid.

1. Tạo môi trường và cài đặt thư viện:

# Di chuyển vào thư mục model

cd model

# Tạo venv

python -m venv venv

# Kích hoạt venv (Windows)

.\venv\Scripts\activate

# Cài đặt các thư viện nặng (Torch, Transformers, Qwen...)

pip install -r requirements.txt

# Lưu ý: Nếu dùng GPU NVIDIA, hãy đảm bảo cài torch bản CUDA.

2. Chạy Worker:
   Mở một cửa sổ Terminal riêng (Terminal A), chạy lệnh sau:

# Chạy Celery Worker (Pool=solo là bắt buộc trên Windows để tránh lỗi)

python -m celery -A tasks worker --loglevel=info --pool=solo

Lần chạy đầu tiên sẽ mất vài phút để tải Model từ HuggingFace.

🌐 Bước 3: Cài đặt và Chạy API Gateway (api/)

Service này nhận request từ người dùng và đẩy vào hàng đợi Redis.

1. Tạo môi trường và cài đặt thư viện:

# Mở một Terminal MỚI (Terminal B). Di chuyển vào thư mục api

cd api

# Tạo venv

python -m venv venv

# Kích hoạt venv (Windows)

.\venv\Scripts\activate

# Cài đặt thư viện nhẹ

pip install -r requirements.txt

2. Chạy API Server:

uvicorn main:app --reload --port 8000

⚡ Bước 4: Kiểm thử (Testing)

Bạn có thể dùng Postman hoặc cURL để gửi yêu cầu.

1. Gửi yêu cầu tạo sơ đồ (POST)

URL: http://127.0.0.1:8000/predict
Body (JSON):

{
"text": "Tạo sơ đồ luồng đăng nhập bao gồm: Người dùng nhập user/pass, gửi đến API, API check Database. Nếu đúng trả về Token, sai trả về lỗi."
}

Response:

{
"message": "Đã gửi yêu cầu tạo sơ đồ",
"task_id": "d853d254-018a-4d0e-b0e2-8c36bf1066da",
...
}

2. Lấy kết quả (GET)

Lấy task_id từ bước trên để kiểm tra kết quả.

URL: http://127.0.0.1:8000/results/<TASK_ID>

Response (Khi hoàn thành):

{
"task_id": "...",
"status": "SUCCESS",
"data": {
"status": "completed",
"mermaid_code": "sequenceDiagram\n participant User..."
}
}

⚠️ Các lỗi thường gặp

Lỗi clocks are out of sync (Lệch giờ):

Đảm bảo code celery ở cả 2 file main.py và tasks.py đã có cấu hình timezone='Asia/Ho_Chi_Minh'.

Nếu vẫn bị, hãy restart lại Redis: docker restart redis-broker.

Lỗi Model chưa sẵn sàng:

Worker cần thời gian để tải model vào VRAM/RAM. Hãy nhìn vào Terminal A (Worker) xem đã hiện dòng "Model tải thành công!" hay chưa.

Lỗi Flash Attention trên Windows:

Trong file model/tasks.py, hãy comment dòng attn_implementation="flash_attention_2" nếu bạn chưa biên dịch được thư viện này trên Windows.

Worker không nhận Task:

Kiểm tra xem tên task trong api/main.py (send_task('generate_mermaid_task', ...)) có khớp 100% với model/tasks.py (name="generate_mermaid_task") không.
