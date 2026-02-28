import os
import torch
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# =============================================================================
# 1. 설정 및 경로
# =============================================================================
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"  # 30-layer 기반 모델로 시작 권장
OUT_DIR  = "/content/drive/MyDrive/EXAONE_Quantized/final_nearid_gptq"

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

# 어떤 레이어를 무력화할지 (예: 마지막 6개)
NEAR_ID_LAYERS = list(range(24, 30))   # 24,25,26,27,28,29

print("[INFO] 모델 및 토크나이저 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)

# =============================================================================
# 2. near-identity 무력화 (GPTQ 전에 적용)
# =============================================================================
@torch.no_grad()
def zero_linear_(linear: torch.nn.Linear):
    # weight / bias를 0으로 만들어 출력이 0이 되게 함
    if hasattr(linear, "weight") and linear.weight is not None:
        linear.weight.zero_()
    if hasattr(linear, "bias") and linear.bias is not None:
        linear.bias.zero_()

@torch.no_grad()
def apply_near_identity(model, layer_ids):
    """
    안전한 near-identity:
    - attention output path: o_proj -> 0
    - mlp output path: down_proj -> 0
    이렇게 하면 residual 때문에 레이어 전체가 x ≈ x로 동작.
    """
    layers = model.model.layers  # HF 계열: model.model.layers
    for i in layer_ids:
        blk = layers[i]
        # Attention 출력 경로 무력화
        zero_linear_(blk.self_attn.o_proj)
        # MLP 출력 경로 무력화
        zero_linear_(blk.mlp.down_proj)
    print(f"[INFO] near-identity 적용 완료: layers={layer_ids} (o_proj, down_proj -> 0)")

apply_near_identity(model, NEAR_ID_LAYERS)

# =============================================================================
# 3. 캘리브레이션 데이터 전처리 (기존 형식 유지)
# =============================================================================
print("[INFO] 캘리브레이션 데이터 전처리 중...")
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")

def preprocess(example):
    messages = example["conversations"]
    user_content = messages[0]["content"]
    assistant_content = messages[1]["content"][:300] if len(messages[1]["content"]) > 300 else messages[1]["content"]
    full_prompt = f"### User: {user_content}\n### Assistant: {assistant_content}{tokenizer.eos_token}\n"
    return {"text": full_prompt}

ds = ds.map(preprocess, remove_columns=ds.column_names)

# =============================================================================
# 4. GPTQ 설정 (전 레이어 대상)
# =============================================================================
print(f"[INFO] GPTQ 진행 (Samples: {NUM_CALIBRATION_SAMPLES}, Max Len: {MAX_SEQUENCE_LENGTH})...")

recipe = [
    GPTQModifier(
        scheme="W4A16",
        targets=["Linear"],
        # 여기서는 가능한 ignore를 최소화해서 Marlin/커널 경로를 깨지 않게 함
        ignore=["embed_tokens", "lm_head"],
        dampening_frac=0.01,
    )
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# =============================================================================
# 5. 저장
# =============================================================================
print("[INFO] 양자화 모델 저장 중...")
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

# (선택) safetensors 키 정규화 - 네가 쓰던 보정 그대로 유지 가능
def normalize_safetensors_keys(output_dir):
    from safetensors.torch import load_file, save_file
    for safetensors_file in Path(output_dir).glob("*.safetensors"):
        print(f"  정규화 중: {safetensors_file.name}")
        weights = load_file(str(safetensors_file), device="cpu")
        fixed = {}
        for k, v in weights.items():
            nk = k.replace("model.model.", "model.", 1) if k.startswith("model.model.") else k
            fixed[nk] = v
        save_file(fixed, str(safetensors_file))
        print(f"    ✓ 완료 (총 {len(fixed)} 가중치)")

normalize_safetensors_keys(OUT_DIR)
print(f"[INFO] 완료: {OUT_DIR}")
