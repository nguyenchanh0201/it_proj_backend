import os
import torch
from threading import Thread
from celery import Celery
from transformers import AutoProcessor, AutoModelForCausalLM, TextIteratorStreamer

# --- CẤU HÌNH ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
# Gemma 3 4B là model multimodal (hình ảnh + văn bản) rất mạnh và nhẹ
MODEL_NAME = "google/gemma-3-4b-it" 
access_token = ""

# --- KHỞI TẠO CELERY ---
celery_app = Celery('worker_app', broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    timezone='Asia/Ho_Chi_Minh',
    broker_connection_retry_on_startup=True,
    task_track_started=True
)

# --- LOAD MODEL (Singleton-style) ---
print(f"⏳ Đang khởi tạo Gemma 3: {MODEL_NAME}...")
try:
    # AutoModelForMultimodalLM là class chuẩn cho Gemma 3
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16, # Khuyến nghị cho card NVIDIA T4/A10/L4
        device_map="auto",
        trust_remote_code=True,
        token= access_token
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("✅ Gemma 3 đã sẵn sàng!")
except Exception as e:
    print(f"❌ Lỗi tải model: {e}")
    model, processor = None, None

@celery_app.task(bind=True, name="generate_mermaid_task")
def generate_mermaid_task(self, scenario_description: str) -> dict:
    if model is None or processor is None:
        return {"error": "Model chưa được load thành công."}

    self.update_state(state='PROGRESS', meta={'percent': 5, 'message': 'Đang chuẩn bị prompt...'})

    # System prompt tối ưu cho Mermaid
    system_prompt = "You are an expert Software Architect. Convert the user scenario into a Mermaid.js `sequenceDiagram`. Return ONLY the code block."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": scenario_description}
    ]

    # Tiền xử lý dữ liệu
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # Cấu hình Streaming
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1500, temperature=0.1)

    # Chạy generation trong thread riêng để không block loop lấy token
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    token_count = 0
    
    for new_text in streamer:
        generated_text += new_text
        token_count += 1
        
        # Gửi update trạng thái sau mỗi 5 token
        if token_count % 5 == 0:
            self.update_state(state='PROGRESS', meta={
                'percent': min(10 + (token_count // 3), 95),
                'message': f'Đang sinh mã... ({token_count} tokens)',
                'partial_result': generated_text
            })

    thread.join()
    
    return {
        "status": "completed",
        "model": MODEL_NAME,
        "mermaid_code": generated_text.strip()
    }
