import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# ---------------------------------------------------------
# 1. 모델 및 토크나이저 로드 
# ---------------------------------------------------------
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
print(f"[INFO] 모델({MODEL_ID}) 로드 중...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16, # 메모리 효율을 위해 bfloat16 사용
    trust_remote_code=True,
    device_map="auto"
)

# ---------------------------------------------------------
# 2. 캘리브레이션 데이터 준비
# ---------------------------------------------------------
print("[INFO] 캘리브레이션 데이터(MANTA-1M) 전처리 중...")
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
NUM_CALIBRATION_SAMPLES = 256 # 영점 조절을 위한 샘플 수
MAX_SEQUENCE_LENGTH = 512

ds = load_dataset(DATASET_ID, split=f"train[:{NUM_CALIBRATION_SAMPLES}]")

def preprocess(example):
    # EXAONE 전용 대화 템플릿 적용
    return {
        "text": tokenizer.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False
        )
    }
ds = ds.map(preprocess)

# ---------------------------------------------------------
# 3. Marlin 커널용 레시피 설정
# ---------------------------------------------------------
recipe = [
    GPTQModifier(
        scheme="W4A16",         # 가중치 4비트 양자화
        targets=["Linear"],     # 선형 레이어 대상
        ignore=["embed_tokens", "lm_head"], # 성능 보호를 위해 입출력단 제외
        dampening_frac=0.01     # 양자화 안정성 확보
    )
]

# ---------------------------------------------------------
# 4. 경량화 실행 (Oneshot Quantization)
# ---------------------------------------------------------
print("[INFO] 모델 경량화 작업 시작")
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# ---------------------------------------------------------
# 5. 결과 저장
# ---------------------------------------------------------
OUT_DIR = "/content/drive/MyDrive/EXAONE_Quantized/model" 
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[INFO] 저장 중... 경로: {OUT_DIR}")
# save_compressed=True 옵션으로 4비트 압축 형식 저장
model.save_pretrained(OUT_DIR, save_compressed=True) 
tokenizer.save_pretrained(OUT_DIR)

print(f"모든 작업 완료")
