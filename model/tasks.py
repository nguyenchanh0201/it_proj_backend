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

# --- CẤU HÌNH ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MODEL_TYPE = os.getenv("MODEL_TYPE", "qwen").lower()
HF_TOKEN = os.getenv("HF_TOKEN", "")

celery_app = Celery('worker_app', broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    timezone='Asia/Ho_Chi_Minh',
    broker_connection_retry_on_startup=True,
    task_track_started=True
)

MODELS_CONFIG = {
    "qwen": {"repo": "Qwen/Qwen3-VL-4B-Instruct", "class": AutoModelForImageTextToText, "loader": "processor"},
    "gemma": {"repo": "google/gemma-3-4b-it", "class": AutoModelForCausalLM, "loader": "processor"},
    "phi": {"repo": "microsoft/Phi-3.5-vision-instruct", "class": AutoModelForCausalLM, "loader": "processor"},
    "llama": {"repo": "meta-llama/Llama-3.2-3B-Instruct", "class": AutoModelForCausalLM, "loader": "tokenizer"}
}

# --- LOAD MODEL ĐỘNG ---
config = MODELS_CONFIG.get(MODEL_TYPE)
model, processor, tokenizer = None, None, None

if config:
    try:
        model = config["class"].from_pretrained(
            config["repo"],
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN,
            _attn_implementation='eager'
        )
        if config["loader"] == "processor":
            processor = AutoProcessor.from_pretrained(config["repo"], trust_remote_code=True, token=HF_TOKEN)
            tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
        else:
            tokenizer = AutoTokenizer.from_pretrained(config["repo"], token=HF_TOKEN)
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            processor = tokenizer
        print(f"✅ {MODEL_TYPE.upper()} ready!")
    except Exception as e:
        print(f"❌ Error: {e}")

# --- CELERY TASK ---
@celery_app.task(bind=True, name="generate_mermaid_task")
def generate_mermaid_task(self, input_text: str, mode: str = "generate") -> dict:
    """
    mode: "generate" (Scenario -> Mermaid) 
    mode: "fix" (Broken Mermaid or Code Snippet -> Fixed Mermaid)
    """
    if model is None: return {"error": "Model not loaded."}

    # 1. Chọn System Prompt dựa trên Mode
    if mode == "fix":
        system_prompt = (
            "You are a Mermaid.js syntax expert. "
            "If the input is Mermaid code, fix any syntax errors. "
            "If the input is a programming code snippet, convert it into a valid Mermaid.js sequenceDiagram. "
            "Return ONLY the corrected/generated Mermaid code block."
        )
        message_content = f"Fix or convert this code:\n\n{input_text}"
    else:
        system_prompt = "You are an expert Software Architect. Convert the user scenario into a Mermaid.js `sequenceDiagram`. Return ONLY the code block."
        message_content = input_text

    self.update_state(state='PROGRESS', meta={'percent': 5, 'message': f'Mode: {mode} - Đang xử lý...'})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message_content}
    ]

    # 2. Tiền xử lý
    if MODEL_TYPE == "llama":
        inputs_data = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = {"input_ids": inputs_data.to(model.device)}
    else:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)

    # 3. Generation với Streaming
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
                'message': f'Đang xử lý ({token_count} tokens)...',
                'partial_result': generated_text
            })

    thread.join()
    return {
        "status": "completed",
        "mode": mode,
        "mermaid_code": generated_text.strip()
    }
