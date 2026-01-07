# AI Mermaid Diagram Generator (Microservices)

Hệ thống tạo và tối ưu hóa sơ đồ Mermaid.js dựa trên kiến trúc Microservices. Dự án hỗ trợ đa mô hình (Multi-model) và tích hợp xử lý thời gian thực qua WebSocket.

## 🌟 Tính năng chính

* **Multi-Model Support:** Tích hợp linh hoạt các dòng Model SOTA: `Qwen3-VL`, `Gemma-3`, `Phi-3.5-Vision`, và `Llama-3.2`.
* **Chế độ hoạt động song song:**
* `generate`: Chuyển đổi mô tả nghiệp vụ thành sơ đồ trình tự (`sequenceDiagram`).
* `fix`: Tự động sửa lỗi cú pháp Mermaid hoặc convert code (Python, JS...) sang sơ đồ.


* **Real-time Streaming:** Theo dõi quá trình AI suy luận từng Token thông qua kết nối **WebSocket**.
* **Kiến trúc hướng sự kiện:** Tách biệt API Gateway và AI Worker thông qua **Redis & Celery**.

---

## 📂 Cấu trúc thư mục

```text
my_project/
├── api/                # Service 1: API Gateway (FastAPI)
│   └── main.py         # REST Endpoint & WebSocket Logic
├── model/              # Service 2: AI Worker (Celery)
│   ├── tasks.py        # Model Inference & Logic
│   └── parser.py       # Công cụ trích xuất mã Mermaid
├── docker-compose.yml  # Quản lý hạ tầng (Redis)
└── README.md           

```

---

## 🛠 Cấu hình & Chạy dự án

### 1. Hạ tầng (Message Broker)

Sử dụng Docker để khởi động Redis nhanh chóng:

```bash
docker run -d -p 6379:6379 --name redis-broker redis

```

### 2. Cài đặt AI Worker (`model/`)

Service này chịu trách nhiệm tải Model nặng và xử lý tính toán.

**Biến môi trường cần thiết:**
| Biến | Mô tả | Giá trị ví dụ |
| :--- | :--- | :--- |
| `MODEL_TYPE` | Loại model muốn chạy | `qwen`, `gemma`, `phi`, `llama` |
| `HF_TOKEN` | Token truy cập HuggingFace | `hf_xxxxxxxxxxxxxxxxx` |
| `REDIS_URL` | Địa chỉ kết nối Redis | `redis://localhost:6379/0` |

**Các bước chạy:**

```bash
cd model
python -m venv venv
source venv/bin/activate  # Hoặc .\venv\Scripts\activate trên Windows
pip install -r requirements.txt

# Khởi động Worker (Sử dụng --pool=solo cho Windows)
export MODEL_TYPE="qwen"
python -m celery -A tasks worker --loglevel=info --pool=solo

```

### 3. Cài đặt API Gateway (`api/`)

Service nhẹ, nhận request và quản lý kết nối WebSocket.

```bash
cd api
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Khởi động server
uvicorn main:app --reload --port 8000

```

---

## 📡 Hướng dẫn tích hợp API

### 1. Gửi yêu cầu tạo sơ đồ

**Endpoint:** `POST /predict`

**Payload:**

```json
{
  "text": "Người dùng đăng ký, hệ thống gửi email xác nhận, người dùng click link để hoàn tất",
  "mode": "generate"
}

```

**Response:** Trả về `task_id` để theo dõi tiến độ.

### 2. Theo dõi tiến độ & Nhận kết quả (WebSocket)

**URL:** `ws://127.0.0.1:8000/ws/task/{task_id}`

Khi kết nối thành công, bạn sẽ nhận được các gói tin JSON chứa trạng thái:

* `PROGRESS`: Chứa `percent` và `partial_result` (mã Mermaid đang được sinh ra).
* `SUCCESS`: Chứa kết quả `result` cuối cùng đã qua bộ lọc cú pháp.

---

## 🤖 Cấu hình AI Model (Prompting)

Hệ thống được thiết kế với **Temperature = 0.01** để đảm bảo tính nhất quán tuyệt đối trong cú pháp Mermaid.

* **Prompt Generate:** Tập trung vào vai trò Kiến trúc sư phần mềm, chuyển đổi logic nghiệp vụ sang `sequenceDiagram`.
* **Prompt Fix:** Tập trung vào vai trò Validator, chỉ sửa lỗi cú pháp và trả về code sạch, không kèm giải thích.
