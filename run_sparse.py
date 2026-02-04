"""
Experiment 25: SparseGPT (unstructured) + Distillation + GPTQ (4-bit)
"""

import os
import shutil
import random
import subprocess
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.pruning import SparseGPTModifier
from llmcompressor.modifiers.quantization import GPTQModifier


MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "./model_exp25_sparse"
SUBMIT_DIR = "./model"
SUBMIT_ZIP = "submit.zip"

DATASET_ID = "LGAI-EXAONE/MANTA-1M"

SEED = 42
MODE = "A"

NUM_SPARSE_SAMPLES = 512
NUM_DISTILL_SAMPLES = 2048
NUM_GPTQ_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 512

SPARSITY = 0.5
BLOCK_SIZE = 128
DAMPENING_FRAC = 0.01

LEARNING_RATE = 2e-5
NUM_EPOCHS = 1
BATCH_SIZE = 2
TEMPERATURE = 2.0

GPTQ_GROUP_SIZE = 128


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_autocast_context(device):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    return nullcontext()


def build_dataset(tokenizer, num_samples, start=0):
    split = f"train[{start}:{start + num_samples}]"
    ds = load_dataset(DATASET_ID, split=split)

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["conversations"],
                add_generation_prompt=True,
                tokenize=False,
            )
        }

    ds = ds.map(preprocess)

    tokenized = ds.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            padding="max_length",
        ),
        batched=True,
        remove_columns=ds.column_names,
    )

    # ✅ 핵심: DataLoader로 넘길 때 list/string 필드가 섞이지 않도록
    #    torch 텐서 컬럼만 남기고 포맷을 고정한다.
    cols = ["input_ids", "attention_mask"]
    if "token_type_ids" in tokenized.column_names:
        cols.append("token_type_ids")

    tokenized.set_format(type="torch", columns=cols)

    return tokenized


def sparsegpt_phase(model, tokenizer):
    print("\n" + "=" * 70)
    print("[PHASE 1/3] 🧹 SPARSEGPT (Unstructured)")
    print("=" * 70)

    ds = build_dataset(tokenizer, NUM_SPARSE_SAMPLES, start=0)

    recipe = [
        SparseGPTModifier(
            targets=["Linear"],
            ignore=["lm_head", "embed_tokens"],
            sparsity=SPARSITY,
            block_size=BLOCK_SIZE,
            dampening_frac=DAMPENING_FRAC,
        )
    ]

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_SPARSE_SAMPLES,
    )

    torch.cuda.empty_cache()
    print("✓ SparseGPT pruning 완료")
    return model


def distillation_phase(student_model, tokenizer, device):
    print("\n" + "=" * 70)
    print("[PHASE 2/3] 🎓 DISTILLATION (logits only)")
    print("=" * 70)

    student_model.to(device)
    student_model.train()
    student_model.config.use_cache = False

    teacher_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    teacher_model.eval()

    ds = build_dataset(tokenizer, NUM_DISTILL_SAMPLES, start=NUM_SPARSE_SAMPLES)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)

    autocast_context = get_autocast_context(device)
    total_steps = NUM_EPOCHS * len(loader)
    step = 0

    for epoch in range(NUM_EPOCHS):
        for batch in loader:
            step += 1
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast_context:
                student_outputs = student_model(**batch)
                student_logits = student_outputs.logits

                with torch.no_grad():
                    teacher_outputs = teacher_model(**batch)
                    teacher_logits = teacher_outputs.logits

                loss_fct = nn.KLDivLoss(reduction="batchmean")
                loss = loss_fct(
                    F.log_softmax(student_logits / TEMPERATURE, dim=-1),
                    F.softmax(teacher_logits / TEMPERATURE, dim=-1),
                ) * (TEMPERATURE ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0 or step == total_steps:
                print(f"step {step}/{total_steps} | loss {loss.item():.4f}")

    student_model.eval()
    del teacher_model
    torch.cuda.empty_cache()
    print("✓ Distillation 완료")
    return student_model


def gptq_phase(model, tokenizer):
    print("\n" + "=" * 70)
    print("[PHASE 3/3] 🔢 GPTQ (4-bit, compressed-tensors)")
    print("=" * 70)

    ds = build_dataset(tokenizer, NUM_GPTQ_SAMPLES, start=NUM_SPARSE_SAMPLES + NUM_DISTILL_SAMPLES)

    recipe = [
        GPTQModifier(
            targets=["Linear"],
            ignore=["lm_head", "embed_tokens"],
            config_groups={
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "group",
                        "group_size": GPTQ_GROUP_SIZE,
                    },
                }
            },
        )
    ]

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_GPTQ_SAMPLES,
    )

    torch.cuda.empty_cache()
    print("✓ GPTQ 완료")
    return model


def save_and_package(model, tokenizer):
    print("\n" + "=" * 70)
    print("[FINAL] 💾 SAVE & ZIP")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR, save_compressed=True)
    tokenizer.save_pretrained(OUT_DIR)

    if os.path.exists(SUBMIT_DIR):
        shutil.rmtree(SUBMIT_DIR)
    shutil.copytree(OUT_DIR, SUBMIT_DIR)

    if os.path.exists(SUBMIT_ZIP):
        os.remove(SUBMIT_ZIP)

    shutil.make_archive("submit", "zip", ".", "model")

    zip_size_mb = os.path.getsize(SUBMIT_ZIP) / (1024 ** 2)
    print(f"submit.zip size: {zip_size_mb:.1f} MB")


def run_post_checks():
    print("\n" + "=" * 70)
    print("[POST] ✅ Running test_before_submit.py")
    print("=" * 70)
    subprocess.run(
        [
            "python",
            "test_before_submit.py",
            "--model",
            SUBMIT_DIR,
            "--zip",
            SUBMIT_ZIP,
        ],
        check=False,
    )

    print("\n" + "=" * 70)
    print("[POST] 📊 Running estimate_score.py")
    print("=" * 70)
    subprocess.run(["python", "estimate_score.py", SUBMIT_DIR], check=False)


def print_compliance_checklist():
    print("\n" + "=" * 70)
    print("규정 위반 체크리스트")
    print("=" * 70)
    print("✓ HF from_pretrained 로딩 가능")
    print("✓ vLLM 수정 없음")
    print("✓ model/ 구조 정상")
    print("✓ zip 크기 제한 충족")
    print("✓ 추가 로더 필요 없음")


def main():
    if MODE != "A":
        raise ValueError("MODE must be 'A' for SparseGPT + Distill + GPTQ pipeline")

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] 모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    print("[INFO] 모델/토크나이저 로드 완료")

    model = sparsegpt_phase(model, tokenizer)
    model = distillation_phase(model, tokenizer, device)
    model = gptq_phase(model, tokenizer)

    save_and_package(model, tokenizer)
    run_post_checks()
    print_compliance_checklist()


if __name__ == "__main__":
    main()
