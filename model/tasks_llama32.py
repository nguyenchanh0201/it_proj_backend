import os
import torch
from threading import Thread
from celery import Celery
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

# --- CẤU HÌNH ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
# Sử dụng Llama 3.2 3B Instruct (nhẹ và rất thông minh trong việc viết code)
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct" 
access_token = "" # Đảm bảo bạn đã chấp nhận điều khoản của Meta trên HuggingFace

# --- KHỞI TẠO CELERY ---
celery_app = Celery('worker_llama', broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    timezone='Asia/Ho_Chi_Minh',
    broker_connection_retry_on_startup=True,
    task_track_started=True
)

# --- LOAD MODEL (Llama 3.2) ---
print(f"⏳ Đang khởi tạo Llama 3.2: {MODEL_NAME}...")
try:
    # Llama 3.2 sử dụng AutoTokenizer thay vì AutoProcessor của multimodal
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=access_token
    )
    # Llama 3.2 yêu cầu pad_token nếu chưa có (thường dùng eos_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("✅ Llama 3.2 đã sẵn sàng!")
except Exception as e:
    print(f"❌ Lỗi tải model: {e}")
    model, tokenizer = None, None

@celery_app.task(bind=True, name="generate_mermaid_llama_task")
def generate_mermaid_task(self, scenario_description: str) -> dict:
    if model is None or tokenizer is None:
        return {"error": "Model Llama 3.2 chưa được load."}

    self.update_state(state='PROGRESS', meta={'percent': 5, 'message': 'Đang phân tích kịch bản...'})

    # System prompt đặc thù cho Llama 3.2 để ép output ra Mermaid chuẩn
    messages = [
        {"role": "system", "content": "You are a professional software architect. Convert the user's description into a valid Mermaid.js sequenceDiagram. Return ONLY the code block, no explanations."},
        {"role": "user", "content": f"Create a sequence diagram for: {scenario_description}"}
    ]

    # Sử dụng apply_chat_template của Llama 3.2
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(input_ids=inputs, streamer=streamer, max_new_tokens=1024, temperature=0.2)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    token_count = 0
    
    for new_text in streamer:
        generated_text += new_text
        token_count += 1
        
        # Gửi update tiến độ qua WebSocket (thông qua Celery backend)
        if token_count % 8 == 0:
            self.update_state(state='PROGRESS', meta={
                'percent': min(15 + (token_count // 5), 98),
                'message': f'Llama 3.2 đang vẽ... ({token_count} tokens)',
                'partial_result': generated_text
            })

    thread.join()
    
    return {
        "status": "completed",
        "model": MODEL_NAME,
        "mermaid_code": generated_text.strip()
    }
