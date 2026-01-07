import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from celery import Celery
from celery.result import AsyncResult

# --- CẤU HÌNH ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    'api_client',
    broker=REDIS_URL,
    backend=REDIS_URL
)

app = FastAPI(title="Mermaid AI Gateway")

class PredictRequest(BaseModel):
    text: str
    mode: str = "generate" 

@app.post("/predict", status_code=202)
def create_prediction_task(request: PredictRequest):
    """
    Gửi task vào queue mặc định. Worker hiện tại đang chạy model nào 
    thì sẽ nhận task đó.
    """
    task = celery_app.send_task(
        'generate_mermaid_task', 
        args=[request.text, request.mode]
    )
    
    return {
        "task_id": task.id,
        "mode": request.mode,
        "status": "Task submitted to default queue"
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
                "message": "Đang chờ worker...",
                "partial_result": None,
                "result": None
            }

            if status == 'SUCCESS':
                res = task_result.get()
                response_data.update({
                    "percent": 100,
                    "message": "Hoàn tất!",
                    "result": res.get("mermaid_code")
                })
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
                response_data["message"] = "Worker đã nhận task và đang xử lý..."
                await websocket.send_json(response_data)

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        print(f"Kết nối WebSocket bị ngắt: {task_id}")
