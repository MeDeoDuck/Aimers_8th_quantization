import os
import random  # ← 추가
import numpy as np  # ← 추가
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# ---------------------------------------------------------
# 0. 시드 고정 (재현성 확보)
# ---------------------------------------------------------
SEED = 1

random.seed(SEED)           # Python 기본 random
np.random.seed(SEED)        # NumPy (datasets 내부 셔플 등에 영향)
torch.manual_seed(SEED)     # PyTorch CPU 연산
torch.cuda.manual_seed_all(SEED)  # PyTorch GPU 연산 (멀티 GPU 포함)

# cuDNN 결정론적 모드 (속도는 약간 느려질 수 있음)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"[INFO] 시드 고정 완료 (seed={SEED})")

# ---------------------------------------------------------
# 1. 모델 및 토크나이저 로드
# ---------------------------------------------------------
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
print(f"[INFO] 모델({MODEL_ID}) 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)

# ---------------------------------------------------------
# 2. 캘리브레이션 데이터 준비
# ---------------------------------------------------------
print("[INFO] 캘리브레이션 데이터(MANTA-1M) 전처리 중...")
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

ds = load_dataset(DATASET_ID, split=f"train[:{NUM_CALIBRATION_SAMPLES}]")

def preprocess(example):
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
        scheme="W4A16",
        targets=["Linear"],
        ignore=["embed_tokens", "lm_head", "q_proj", "k_proj"],
        dampening_frac=0.01
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
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)
print("모든 작업 완료")
