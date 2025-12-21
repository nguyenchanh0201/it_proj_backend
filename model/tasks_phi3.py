import os
import torch
from threading import Thread
from celery import Celery
from transformers import AutoModelForCausalLM, AutoProcessor, TextIteratorStreamer

# --- CẤU HÌNH ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MODEL_NAME = "microsoft/Phi-3.5-vision-instruct"

# --- KHỞI TẠO CELERY ---
celery_app = Celery('worker_phi', broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    timezone='Asia/Ho_Chi_Minh',
    broker_connection_retry_on_startup=True,
    task_track_started=True
)

# --- LOAD MODEL (Phi-3.5 Vision) ---
print(f"⏳ Đang khởi tạo Phi-3.5 Vision: {MODEL_NAME}...")
try:
    # Sử dụng đúng cấu hình bạn yêu cầu
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True, 
        torch_dtype="auto", 
        device_map="auto",
        # _attn_implementation='flash_attention_2' # Bỏ comment nếu GPU của bạn hỗ trợ (A10, L4, RTX 30/40 series)
    )
    
    # Phi-3.5 Vision cần Processor để xử lý chat template và image (nếu có)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    print("✅ Phi-3.5 Vision đã sẵn sàng!")
except Exception as e:
    print(f"❌ Lỗi tải model: {e}")
    model, processor = None, None

@celery_app.task(bind=True, name="generate_mermaid_phi_task")
def generate_mermaid_task(self, scenario_description: str) -> dict:
    if model is None or processor is None:
        return {"error": "Model Phi-3.5 chưa được load."}

    self.update_state(state='PROGRESS', meta={'percent': 5, 'message': 'Phi-3.5 đang phân tích kịch bản...'})

    # Định dạng Prompt cho Phi-3.5 (Sử dụng cấu trúc <|system|>, <|user|>, <|assistant|>)
    messages = [
        {"role": "system", "content": "You are an AI architect. Convert the description into a Mermaid.js sequenceDiagram. Return ONLY the code block."},
        {"role": "user", "content": scenario_description}
    ]

    # Phi-3.5 Vision chat template
    prompt = processor.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    inputs = processor(prompt, return_tensors="pt").to(model.device)

    # Cấu hình Streaming
    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Phi-3.5 yêu cầu num_crops=16 cho vision, nhưng ở đây chỉ dùng text nên cấu hình cơ bản
    generation_kwargs = dict(
        **inputs, 
        streamer=streamer, 
        max_new_tokens=1000, 
        temperature=0.1,
        do_sample=False # Đặt False để kết quả Mermaid ổn định nhất
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    token_count = 0
    
    for new_text in streamer:
        generated_text += new_text
        token_count += 1
        
        # Cập nhật state cho WebSocket qua mỗi 5 tokens
        if token_count % 5 == 0:
            self.update_state(state='PROGRESS', meta={
                'percent': min(10 + (token_count // 4), 95),
                'message': f'Phi-3.5 đang giải mã... ({token_count} tokens)',
                'partial_result': generated_text
            })

    thread.join()
    
    return {
        "status": "completed",
        "model": "Phi-3.5-vision",
        "mermaid_code": generated_text.strip()
    }

