import os
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from celery import Celery
from celery.result import AsyncResult

# --- CẤU HÌNH TỪ BIẾN MÔI TRƯỜNG ---
# Mặc định là localhost nếu chạy ngoài Docker, là 'redis' nếu chạy trong Docker Compose
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# --- CẤU HÌNH CELERY CLIENT ---
celery_app = Celery(
    'api_client',
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    enable_utc=False,
    timezone='Asia/Ho_Chi_Minh',
    task_track_started=True,
)

app = FastAPI(title="Mermaid Generator API (Qwen3-VL)")

class PredictRequest(BaseModel):
    text: str

@app.post("/predict", status_code=202)
def create_prediction_task(request: PredictRequest):
    """
    Gửi task sang Worker.
    """
    task = celery_app.send_task(
        'generate_mermaid_task',  # Tên task khớp với Worker
        args=[request.text]
    )
    
    return {
        "message": "Task submitted", 
        "task_id": task.id
    }

@app.websocket("/ws/task/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await websocket.accept()
    try:
        while True:
            task_result = AsyncResult(task_id, app=celery_app)
            status = task_result.status
            
            response_data = {
                "task_id": task_id,
                "status": status,
                "percent": 0,
                "message": "Đang chờ...",
                "result": None
            }

            if status == 'SUCCESS':
                response_data["percent"] = 100
                response_data["message"] = "Hoàn tất!"
                response_data["result"] = task_result.get()
                await websocket.send_json(response_data)
                await websocket.close()
                break
            
            elif status == 'FAILURE':
                response_data["message"] = "Có lỗi xảy ra."
                response_data["result"] = str(task_result.result)
                await websocket.send_json(response_data)
                await websocket.close()
                break
            
            elif status == 'PROGRESS':
                info = task_result.info
                if isinstance(info, dict):
                    response_data.update(info)
                await websocket.send_json(response_data)
            
            elif status == 'STARTED':
                response_data["message"] = "Worker đang khởi động..."
                await websocket.send_json(response_data)

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        print(f"Client disconnected: {task_id}")
