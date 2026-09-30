"""
اختبار الـendpoint بمستند حقيقي (نص OCR في ملف .txt).

    pip install requests
    set RUNPOD_API_KEY=...        (ويندوز)   |   export RUNPOD_API_KEY=...   (لينكس/ماك)
    set ENDPOINT_ID=...           (الـID من صفحة الـendpoint)
    python test_endpoint.py document.txt

بيستخدم /run + polling (مش /runsync) لأن أول طلب بيبقى بطيء بسبب تحميل الموديل.
النتيجة بتتحفظ في result.json.
"""
import json
import os
import sys
import time

import requests

API_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT_ID = os.environ["ENDPOINT_ID"]
BASE = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
HEAD = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

text = open(sys.argv[1], encoding="utf-8").read()
job = requests.post(f"{BASE}/run", headers=HEAD, json={"input": {"text": text}}, timeout=60).json()
job_id = job["id"]
print("job:", job_id)

t0 = time.time()
while True:
    r = requests.get(f"{BASE}/status/{job_id}", headers=HEAD, timeout=60).json()
    status = r.get("status")
    print(f"[{time.time() - t0:5.0f}s] {status}")
    if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
        break
    time.sleep(10)

with open("result.json", "w", encoding="utf-8") as f:
    json.dump(r, f, ensure_ascii=False, indent=2)

out = r.get("output") or {}
res = out.get("result") if isinstance(out, dict) else None
print("\nstatus:", status)
if isinstance(out, dict):
    print("input_truncated:", out.get("input_truncated"), "| input_tokens:", out.get("input_tokens"))
if isinstance(res, dict) and "programs" in res:
    print("عدد البرامج:", len(res["programs"]))
    for i, p in enumerate(res["programs"], 1):
        print(f"{i}. {p.get('name')}  | n={p.get('beneficiaries_count')}")
else:
    print("مفيش programs في الناتج - شوف result.json")
