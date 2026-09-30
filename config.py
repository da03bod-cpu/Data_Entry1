import os

MODEL_ID = os.getenv("MODEL_ID", "mistralai/Mistral-Small-24B-Instruct-2501")
# الـLoRA بيتحمّل وقت التشغيل من Hugging Face (مش من GitHub ولا من جوه الـimage):
#  - LORA_REPO: اسم ريبو الـLoRA على HF بصيغة USERNAME/REPO (يتحط كـenv variable في الـendpoint)
#  - HF_TOKEN : توكن read-only لو الريبو Private
#  - LORA_PATH: اختياري - مسار محلي جاهز (للاختبار بس)، لو موجود بيتجاهل HF
LORA_REPO = os.getenv("LORA_REPO", "YOUR-HF-USERNAME/mistral-nonprofit-lora")
LORA_PATH = os.getenv("LORA_PATH", "")

# مهم جدًا: اكتشفنا وقت التدريب إن النافذة الحقيقية القصوى لهذا الموديل هي 32768 توكن
# (مش 60000 أو 120000 زي ما كان مفترض قديمًا مع Qwen3). أي تجاوز لده بيدخل الموديل
# في منطقة غير مدرّبة عليها من الأساس (positional encoding).
MAX_CONTEXT_TOKENS = 32768
MAX_NEW_TOKENS = 8192  # G_2 محتاج JSON ~6400 توكن، فالحد القديم 4096 كان هيقطعه
SYSTEM_PROMPT_RESERVE = 400  # هامش أمان لبرومبت النظام + قالب المحادثة
MAX_INPUT_TOKENS = MAX_CONTEXT_TOKENS - MAX_NEW_TOKENS - SYSTEM_PROMPT_RESERVE  # ~24000 توكن لنص المستند

DO_SAMPLE = False
# كانت 1.2 و 8 (حل قديم لمشكلة Qwen). مع موديل مدرَّب على JSON ده خطر:
# no_repeat_ngram_size=8 بيمنع تكرار تسلسل مفاتيح الـJSON بين العناصر، و
# repetition_penalty>1 بيضغط على نسخ الأسماء من نص المستند. الافتراضي دلوقتي "متوقف"
# وتقدر ترجعهم من env أو من الطلب نفسه لو لقيت تكرار فعلي.
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.0"))
NO_REPEAT_NGRAM_SIZE = int(os.getenv("NO_REPEAT_NGRAM_SIZE", "0"))  # 0 = متوقف

SYSTEM_PROMPT = (
    "أنت نموذج متخصص في استخراج البرامج والمشاريع والمبادرات والأنشطة "
    "من التقارير السنوية للجمعيات غير الربحية. اقرأ النص كما هو، حتى لو "
    "احتوى على أخطاء OCR بسيطة. استخرج كل عنصر مذكور فعليًا فقط، بدون تكرار "
    "أي عنصر وبدون دمج عناصر مختلفة. أعد JSON فقط وفق المخطط المطلوب، "
    "ولا تضف أي شرح خارج JSON. ضع null فقط عندما تكون المعلومة غير موجودة فعلاً."
)
