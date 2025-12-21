import os
import torch
from threading import Thread
from celery import Celery
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForImageTextToText, 
    AutoProcessor, 
    AutoTokenizer, 
    TextIteratorStreamer
)

# --- CẤU HÌNH HỆ THỐNG ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MODEL_TYPE = os.getenv("MODEL_TYPE", "qwen").lower() # Mặc định là qwen
HF_TOKEN = os.getenv("HF_TOKEN", "")

# --- KHỞI TẠO CELERY ---
celery_app = Celery('worker_app', broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    timezone='Asia/Ho_Chi_Minh',
    broker_connection_retry_on_startup=True,
    task_track_started=True
)

# --- ĐỊNH NGHĨA DANH SÁCH MODEL ---
MODELS_CONFIG = {
    "qwen": {
        "repo": "Qwen/Qwen3-VL-4B-Instruct",
        "class": AutoModelForImageTextToText,
        "loader": "processor"
    },
    "gemma": {
        "repo": "google/gemma-3-4b-it",
        "class": AutoModelForCausalLM,
        "loader": "processor"
    },
    "phi": {
        "repo": "microsoft/Phi-3.5-vision-instruct",
        "class": AutoModelForCausalLM,
        "loader": "processor"
    },
    "llama": {
        "repo": "meta-llama/Llama-3.2-3B-Instruct",
        "class": AutoModelForCausalLM,
        "loader": "tokenizer"
    }
}

# --- LOGIC LOAD MODEL ĐỘNG ---
print(f"🚀 Khởi tạo Worker cho loại Model: {MODEL_TYPE.upper()}")

model, processor, tokenizer = None, None, None
config = MODELS_CONFIG.get(MODEL_TYPE)

if config:
    try:
        # Load Model
        model = config["class"].from_pretrained(
            config["repo"],
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN
        )
        
        # Load Processor hoặc Tokenizer
        if config["loader"] == "processor":
            processor = AutoProcessor.from_pretrained(config["repo"], trust_remote_code=True, token=HF_TOKEN)
            tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
        else:
            tokenizer = AutoTokenizer.from_pretrained(config["repo"], token=HF_TOKEN)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            processor = tokenizer # Để dùng chung biến ở dưới

        print(f"✅ {MODEL_TYPE.upper()} đã sẵn sàng trên {model.device}!")
    except Exception as e:
        print(f"❌ Lỗi load model {MODEL_TYPE}: {e}")
else:
    print(f"⚠️ MODEL_TYPE '{MODEL_TYPE}' không hợp lệ!")

# --- CELERY TASK CHUNG ---
@celery_app.task(bind=True, name="generate_mermaid_task")
def generate_mermaid_task(self, scenario_description: str) -> dict:
    if model is None:
        return {"error": f"Model {MODEL_TYPE} chưa được nạp thành công."}

    self.update_state(state='PROGRESS', meta={'percent': 5, 'message': 'Đang chuẩn bị prompt...'})

    # Tùy chỉnh System Prompt theo từng loại model nếu cần
    system_prompt = "You are an expert Software Architect. Convert the user scenario into a Mermaid.js `sequenceDiagram`. Return ONLY the code block."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": scenario_description}
    ]

    # Tiền xử lý (Xử lý khác biệt giữa Processor và Tokenizer)
    if MODEL_TYPE == "llama":
        inputs_data = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = {"input_ids": inputs_data.to(model.device)}
    else:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1024, temperature=0.1)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    token_count = 0

    for new_text in streamer:
        generated_text += new_text
        token_count += 1
        if token_count % 5 == 0:
            self.update_state(state='PROGRESS', meta={
                'percent': min(10 + (token_count // 3), 95),
                'message': f'{MODEL_TYPE} đang viết... ({token_count} tokens)',
                'partial_result': generated_text
            })

    thread.join()
    return {
        "status": "completed",
        "model": config["repo"],
        "mermaid_code": generated_text.strip()
    }
