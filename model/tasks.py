import os
import time
from threading import Thread
from celery import Celery
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer
import torch

# --- CẤU HÌNH TỪ BIẾN MÔI TRƯỜNG ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"

# --- KHỞI TẠO CELERY ---
celery_app = Celery(
    'worker_app',
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    enable_utc=False,
    timezone='Asia/Ho_Chi_Minh',
    broker_connection_retry_on_startup=True,
    task_track_started=True
)

# --- LOAD MODEL ---
print(f"⏳ Đang tải model {MODEL_NAME}...")
try:
    # Lưu ý: Khi chạy Docker, huggingface cache sẽ được map volume để không phải tải lại
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("✅ Model tải thành công!")
except Exception as e:
    print(f"❌ Lỗi tải model: {e}")
    model = None
    processor = None

@celery_app.task(bind=True, name="generate_mermaid_task")
def generate_mermaid_task(self, scenario_description: str) -> dict:
    if model is None or processor is None:
        return {"error": "Model chưa sẵn sàng trên Server."}

    self.update_state(state='PROGRESS', meta={'percent': 5, 'message': 'Đang tiền xử lý dữ liệu...'})

    system_prompt = "You are an expert Software Architect using Mermaid.js. Convert user scenario into `sequenceDiagram`. Return ONLY mermaid code in markdown."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": scenario_description}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1500, temperature=0.2, do_sample=True)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    token_count = 0
    
    self.update_state(state='PROGRESS', meta={'percent': 10, 'message': 'Model đang suy nghĩ & viết code...'})

    for new_text in streamer:
        generated_text += new_text
        token_count += 1
        if token_count % 5 == 0:
            fake_percent = min(10 + int(token_count / 2), 95)
            self.update_state(state='PROGRESS', meta={
                'percent': fake_percent,
                'message': f'Đang viết... ({token_count} tokens)',
                'partial_result': generated_text
            })

    thread.join()
    
    return {
        "status": "completed",
        "mermaid_code": generated_text
    }