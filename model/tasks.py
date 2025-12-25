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

# --- 1. CẤU HÌNH HỆ THỐNG ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MODEL_TYPE = os.getenv("MODEL_TYPE", "qwen").lower()
HF_TOKEN = os.getenv("HF_TOKEN", "")

celery_app = Celery('worker_app', broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    timezone='Asia/Ho_Chi_Minh',
    broker_connection_retry_on_startup=True,
    task_track_started=True
)

# --- 2. CẤU HÌNH PROMPT RẠCH RÒI ---
PROMPT_FIX = (
    "You are a strict Mermaid.js Syntax Validator.\n"
    "TASK:\n"
    "1. If input is Mermaid: Fix syntax errors, unclosed brackets, or invalid keywords.\n"
    "2. If input is source code: Convert the logic into a valid Mermaid.js sequenceDiagram.\n\n"
    "STRICT RULES:\n"
    "- Output ONLY the code block starting with ```mermaid and ending with ```.\n"
    "- NO explanations, NO introductory text, NO 'Here is the fixed code'.\n"
    "- NEVER use PlantUML syntax (e.g., avoid @startuml)."
)

PROMPT_GENERATE = (
    "You are an expert Software Architect.\n"
    "TASK: Convert the user's business scenario into a professional Mermaid.js `sequenceDiagram`.\n\n"
    "STRICT RULES:\n"
    "- Output ONLY the code block (must be wrapped in ```mermaid ... ```).\n"
    "- Do not talk or explain anything.\n"
    "- Use clear, descriptive participant names.\n"
    "- Ensure the syntax is 100% compliant with Mermaid.js."
)

# --- 3. LOAD MODEL ĐỘNG ---
MODELS_CONFIG = {
    "qwen": {"repo": "Qwen/Qwen3-VL-4B-Instruct", "class": AutoModelForImageTextToText, "loader": "processor"},
    "gemma": {"repo": "google/gemma-3-4b-it", "class": AutoModelForCausalLM, "loader": "processor"},
    "phi": {"repo": "Lexius/Phi-3.5-vision-instruct", "class": AutoModelForCausalLM, "loader": "processor"},
    "llama": {"repo": "meta-llama/Llama-3.2-3B-Instruct", "class": AutoModelForCausalLM, "loader": "tokenizer"}
}

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
            if MODEL_TYPE == "phi" and hasattr(tokenizer, "chat_template"):
                processor.chat_template = tokenizer.chat_template
        else:
            tokenizer = AutoTokenizer.from_pretrained(config["repo"], token=HF_TOKEN)
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            processor = tokenizer
        print(f"✅ {MODEL_TYPE.upper()} ready!")
    except Exception as e:
        print(f"❌ Initialization Error: {e}")

# --- 4. CELERY TASK ---
@celery_app.task(bind=True, name="generate_mermaid_task")
def generate_mermaid_task(self, input_text: str, mode: str = "generate") -> dict:
    if model is None: 
        return {"error": "Model not loaded."}

    # Chọn System Prompt dựa trên Mode
    if mode == "fix":
        system_prompt = PROMPT_FIX
        user_message = f"Input to fix or convert:\n\n{input_text}\n\nResult (Mermaid code only):"
    else:
        system_prompt = PROMPT_GENERATE
        user_message = f"Scenario to diagram:\n\n{input_text}\n\nResult (Mermaid code only):"

    self.update_state(state='PROGRESS', meta={'percent': 5, 'message': f'Mode: {mode.upper()} - Processing...'})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    # Tiền xử lý token
    if MODEL_TYPE == "llama":
        inputs_data = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = {"input_ids": inputs_data.to(model.device)}
    else:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)

    # Generation với Streaming (Temperature thấp để đảm bảo tính chính xác)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1024, temperature=0.01)

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
                'message': f'Generating ({token_count} tokens)...',
                'partial_result': generated_text
            })

    thread.join()

    # --- 5. HẬU XỬ LÝ (TRÍCH XUẤT CODE BLOCK) ---
    final_output = generated_text.strip()
    
    # Logic bóc tách: Chỉ lấy nội dung giữa ```mermaid và ```
    if "```mermaid" in final_output:
        try:
            parts = final_output.split("```mermaid")
            content = parts[1].split("```")[0].strip()
            final_output = f"```mermaid\n{content}\n```"
        except (IndexError, ValueError):
            # Fallback nếu split lỗi
            pass
    elif "```" in final_output:
        # Nếu AI quên chữ 'mermaid' nhưng vẫn dùng code block
        try:
            content = final_output.split("```")[1].strip()
            final_output = f"```mermaid\n{content}\n```"
        except:
            pass

    return {
        "status": "completed",
        "mode": mode,
        "mermaid_code": final_output
    }
