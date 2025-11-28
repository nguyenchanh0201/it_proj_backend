from celery import Celery
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch

# --- CẤU HÌNH ---
MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"

# --- KHỞI TẠO CELERY ---
# Tên 'worker_app' chỉ là định danh nội bộ, quan trọng là Broker URL
celery_app = Celery(
    'worker_app',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# Cấu hình Timezone để tránh lỗi lệch giờ (Clock drift)
celery_app.conf.update(
    enable_utc=False,
    timezone='Asia/Ho_Chi_Minh',
    broker_connection_retry_on_startup=True
)

# --- LOAD MODEL (Global variable) ---
print(f"Đang tải model {MODEL_NAME}...")
try:
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True
        # attn_implementation="flash_attention_2"
    )
    
    # Processor cũng cần trust_remote_code
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("Model tải thành công!")
except Exception as e:
    print(f"Lỗi tải model: {e}")
    model = None
    processor = None

# --- ĐỊNH NGHĨA TASK ---
# QUAN TRỌNG: name="generate_mermaid_task" định danh chính xác tên task để API gọi
@celery_app.task(name="generate_mermaid_task")
def generate_mermaid_task(scenario_description: str) -> dict:
    if model is None or processor is None:
        return {"error": "Model chưa sẵn sàng."}

    print(f"Worker đang xử lý: {scenario_description}")

    system_prompt = (
        "You are an expert Software Architect using Mermaid.js. "
        "Your task is to convert the user's scenario into a valid, complex `sequenceDiagram`. "
        "Return ONLY the mermaid code inside a markdown block."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": scenario_description}
    ]

    # Xử lý input
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to("cuda")

    # Generate
    try:
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1500,
            temperature=0.2,
            do_sample=True
        )
        
        # Cắt input, lấy output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return {
            "status": "completed",
            "mermaid_code": output_text
        }
    except Exception as e:
        return {"error": str(e)}