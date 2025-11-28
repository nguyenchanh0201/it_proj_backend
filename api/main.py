from fastapi import FastAPI
from pydantic import BaseModel
from celery import Celery
from celery.result import AsyncResult

# --- CẤU HÌNH CELERY CLIENT ---
# Chỉ cần kết nối đến cùng Broker (Redis) với Worker
celery_app = Celery(
    'api_client',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

celery_app.conf.update(
    enable_utc=False,
    timezone='Asia/Ho_Chi_Minh',
)

app = FastAPI(title="Mermaid Generator API (Qwen3-VL)")

class PredictRequest(BaseModel):
    text: str

@app.post("/predict", status_code=202)
def create_prediction_task(request: PredictRequest):
    """
    Gửi task sang Worker thông qua Redis.
    """
    # QUAN TRỌNG: Tên task trong send_task phải khớp 100% với name="" bên Worker
    task = celery_app.send_task(
        'generate_mermaid_task',  # Tên định danh task
        args=[request.text]       # Tham số truyền vào hàm
    )
    
    return {
        "message": "Đã gửi yêu cầu tạo sơ đồ", 
        "task_id": task.id,
        "input_text": request.text
    }

@app.get("/results/{task_id}")
def get_result(task_id: str):
    """
    Lấy kết quả từ Redis dựa trên task_id
    """
    task_result = AsyncResult(task_id, app=celery_app)
    
    if task_result.ready():
        # Task đã chạy xong (thành công hoặc thất bại)
        result_data = task_result.get()
        return {
            "task_id": task_id,
            "status": task_result.status,
            "data": result_data
        }
    
    # Task đang chạy (PENDING hoặc STARTED)
    return {
        "task_id": task_id, 
        "status": task_result.status, 
        "message": "Đang xử lý, vui lòng thử lại sau..."
    }