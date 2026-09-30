"""
RunPod Serverless handler - استخراج برامج/مشاريع الجمعيات من نص مستند
Mistral Small 24B (4-bit) + LoRA adapter المدرَّب

Input format (job["input"]):
  {"text": "===== PAGE 1 =====\n...نص المستند الكامل..."}

Output:
  {"result": {"programs": [...]}, "input_truncated": bool}
"""
import sys

# حماية إضافية: لو torchvision لسه موجود في الـimage ومش متوافق مع torch (operator torchvision::nms does not exist)،
# نخلّي transformers يعتبره مش متثبّت. الموديل نصي ومش محتاجه. لازم السطر ده يجي قبل import transformers/peft.
sys.modules["torchvision"] = None

import json
import os
import re
import torch
import runpod
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import snapshot_download

import config

# الموديل الأساسي بييجي من "Cached models" بتاع Runpod (خانة Model في إعدادات الـendpoint)،
# فمفيش Network Volume ومفيش تحاسب على وقت التنزيل. لو مالقاهوش في الكاش بيرجع يحمّله من HF (أبطأ وأغلى).
MODEL_NAME = os.getenv("MODEL_NAME", config.MODEL_ID)
HF_CACHE_ROOT = os.getenv("HF_CACHE_ROOT", "/runpod-volume/huggingface-cache/hub")


def resolve_base_path(model_id: str) -> str:
    org, name = model_id.split("/", 1)
    root = os.path.join(HF_CACHE_ROOT, f"models--{org}--{name}")
    snaps = os.path.join(root, "snapshots")
    refs_main = os.path.join(root, "refs", "main")
    if os.path.isfile(refs_main):
        cand = os.path.join(snaps, open(refs_main).read().strip())
        if os.path.isdir(cand):
            return cand
    if os.path.isdir(snaps):
        versions = sorted(d for d in os.listdir(snaps) if os.path.isdir(os.path.join(snaps, d)))
        if versions:
            return os.path.join(snaps, versions[0])
    print("WARNING: الموديل مش في كاش Runpod - هيتحمّل من HF (محتاج Container Disk 80GB+).")
    return model_id


BASE_PATH = resolve_base_path(MODEL_NAME)
print("Base model source:", BASE_PATH)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model (4-bit NF4)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_PATH,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
)

def resolve_lora_path() -> str:
    if config.LORA_PATH and os.path.exists(os.path.join(config.LORA_PATH, "adapter_config.json")):
        return config.LORA_PATH
    print("Downloading LoRA from Hugging Face:", config.LORA_REPO)
    return snapshot_download(
        repo_id=config.LORA_REPO,
        token=os.getenv("HF_TOKEN") or None,
        allow_patterns=["adapter_model.safetensors", "adapter_config.json"],
        cache_dir=os.getenv("LORA_CACHE_DIR", "/tmp/lora_cache"),  # ~350MB بس، بيتنزّل كل cold start في ثواني
    )


lora_path = resolve_lora_path()
print("Loading LoRA adapter from:", lora_path)
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()
print("Model ready.")


def truncate_to_budget(text: str, max_tokens: int):
    """يقصّ النص عشان يدخل في حدود نافذة السياق (32768 توكن للموديل ده).
    ده حل مؤقت - الحل الصح طويل المدى هو تقسيم المستند لأجزاء (multi-pass)
    زي ما كان مخطط في §16 من الملخص الأصلي، لسه محتاج تنفيذ لاحقًا."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text, False, len(ids)
    truncated_ids = ids[:max_tokens]
    truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    return truncated_text, True, len(ids)


def parse_json_output(s: str):
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", s, re.DOTALL)  # لو الموديل لفّ الـJSON بنص أو ```
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def extract_programs(document_text: str, rep_penalty=None, no_repeat_ngram=None) -> dict:
    truncated_text, was_truncated, original_len = truncate_to_budget(
        document_text, config.MAX_INPUT_TOKENS
    )
    if was_truncated:
        print(
            f"WARNING: input truncated from {original_len} to "
            f"{config.MAX_INPUT_TOKENS} tokens (حد الموديل الأقصى 32768). "
            f"جزء من نهاية المستند اتقص - محتاجين تقسيم متعدد المراحل لاحقًا للمستندات الطويلة."
        )

    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": truncated_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    rp = rep_penalty if rep_penalty is not None else config.REPETITION_PENALTY
    ng = no_repeat_ngram if no_repeat_ngram is not None else config.NO_REPEAT_NGRAM_SIZE
    gen_kwargs = dict(
        max_new_tokens=config.MAX_NEW_TOKENS,
        do_sample=config.DO_SAMPLE,
        repetition_penalty=rp,
        pad_token_id=tokenizer.pad_token_id,
    )
    if ng and ng > 0:
        gen_kwargs["no_repeat_ngram_size"] = ng

    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)

    generated_text = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

    parsed = parse_json_output(generated_text)
    if parsed is None:
        parsed = {"error": "invalid_json_output", "raw_output": generated_text}

    return {
        "result": parsed,
        "input_truncated": was_truncated,
        "input_tokens": min(original_len, config.MAX_INPUT_TOKENS),
    }


def handler(job):
    job_input = job.get("input", {}) or {}
    document_text = job_input.get("text", "")

    if not document_text:
        return {"error": "missing 'text' field in input"}

    try:
        return extract_programs(
            document_text,
            rep_penalty=job_input.get("repetition_penalty"),
            no_repeat_ngram=job_input.get("no_repeat_ngram_size"),
        )
    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
