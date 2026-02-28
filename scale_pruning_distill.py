"""
파이프라인:
1. Scale Pruning (50% 채널 → 0)
2. Distillation (Pruning 손실 복구)
3. GPTQ 4-bit (최종 압축)
"""

import os
import torch
import torch.nn as nn
import shutil
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier


MODEL_ID = "../Model/EXAONE-4.0-1.2B"
OUT_DIR = "./model_exp25"
DATASET_ID = "LGAI-EXAONE/MANTA-1M"

NUM_DISTILL_SAMPLES = 2048
NUM_CALIBRATION_SAMPLES = 2048
MAX_SEQUENCE_LENGTH = 512

PRUNING_RATIO = 0.5
LEARNING_RATE = 2e-5
NUM_EPOCHS = 1
TEMPERATURE = 2.0

print("=" * 70)
print("Scale Pruning + Distillation + GPTQ")
print("최적 파이프라인!")
print("=" * 70)


# ========== Phase 1: Scale Pruning ==========
def scale_pruning(model, ratio=0.5):
    """채널별 스케일을 0으로 만들어 pruning (학습 시 gradient 유지)"""
    print(f"\n🔪 Scale Pruning ({ratio*100:.0f}% 채널 제거)...")
    
    pruned_channels = 0
    total_channels = 0
    
    for name, module in model.named_modules():
        if 'mlp' in name and isinstance(module, nn.Linear) and 'proj' in name:
            # requires_grad를 False로 설정 (gradient 끊기)
            with torch.no_grad():
                weight = module.weight.data
                
                # 출력 채널별 L2 norm
                channel_norms = torch.norm(weight, p=2, dim=1)
                
                # 하위 ratio% 채널 찾기
                num_to_prune = int(weight.size(0) * ratio)
                threshold = torch.kthvalue(channel_norms, num_to_prune)[0]
                
                # Mask 생성
                keep_mask = channel_norms > threshold
                
                # 약한 채널들을 0으로 (gradient 끊긴 상태에서 수정)
                weight[~keep_mask] = 0
                
                if module.bias is not None:
                    module.bias.data[~keep_mask] = 0
                
                pruned_channels += (~keep_mask).sum().item()
                total_channels += weight.size(0)
    
    print(f"✓ Scale Pruning 완료!")
    print(f"   - Pruned: {pruned_channels:,}/{total_channels:,} ({pruned_channels/total_channels*100:.1f}%)")
    
    return model


# ========== Phase 2: Distillation ==========
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.temperature = temperature
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            teacher_logits = teacher_outputs.logits
        
        loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
        loss = loss_fct(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1)
        ) * (self.temperature ** 2)
        
        return (loss, student_outputs) if return_outputs else loss


def distillation_phase(student_model, tokenizer):
    """Phase 2: Distillation (Pruning 손실 복구)"""
    print("\n" + "=" * 70)
    print("[PHASE 2/3] 🎓 DISTILLATION (Pruning 손실 복구)")
    print("=" * 70)
    
    # Teacher (원본)
    print("Teacher 로딩...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    teacher_model.eval()
    
    # Data
    print("데이터 준비...")
    ds = load_dataset(DATASET_ID, split=f"train[:{NUM_DISTILL_SAMPLES}]")
    ds = ds.map(lambda x: {
        "text": tokenizer.apply_chat_template(x["conversations"], add_generation_prompt=True, tokenize=False)
    })
    
    tokenized_ds = ds.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=MAX_SEQUENCE_LENGTH, padding="max_length"),
        batched=True,
        remove_columns=ds.column_names
    )
    
    # Training
    print("Distillation 시작...")
    training_args = TrainingArguments(
        output_dir="./distill_temp",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=2,
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        logging_steps=100,
        save_steps=1000,
        bf16=True,
        report_to="none",
    )
    
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        temperature=TEMPERATURE,
        model=student_model,
        args=training_args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    print("✓ Distillation 완료 (Pruning 손실 복구됨)")
    
    # Cleanup
    del teacher_model
    torch.cuda.empty_cache()
    
    return student_model


# ========== Phase 3: Quantization ==========
def quantization_phase(model, tokenizer):
    """Phase 3: Quantization"""
    print("\n" + "=" * 70)
    print("[PHASE 3/3] QUANTIZATION")
    print("=" * 70)
    
    # Data (다른 샘플)
    ds = load_dataset(DATASET_ID, split=f"train[{NUM_DISTILL_SAMPLES}:{NUM_DISTILL_SAMPLES + NUM_CALIBRATION_SAMPLES}]")
    ds = ds.map(lambda x: {
        "text": tokenizer.apply_chat_template(x["conversations"], add_generation_prompt=True, tokenize=False)
    })
    
    # Uniform 4-bit GPTQ
    recipe = [
        GPTQModifier(
            targets=["Linear"],
            ignore=["lm_head"],
            config_groups={
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "group",
                        "group_size": 128
                    }
                }
            }
        )
    ]
    
    print("GPTQ 4-bit 진행 중...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )
    
    print("✓ Quantization 완료")
    return model


# ========== Save ==========
def save_model(model, tokenizer):
    """저장"""
    print("\n" + "=" * 70)
    print("[FINAL] 저장")
    print("=" * 70)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR, save_compressed=True)
    tokenizer.save_pretrained(OUT_DIR)
    
    # Cleanup
    if os.path.exists("./distill_temp"):
        shutil.rmtree("./distill_temp")
    
    if os.path.exists("./model"):
        shutil.rmtree("./model")
    shutil.copytree(OUT_DIR, "./model")
    shutil.make_archive("submit", "zip", ".", "model")
    
    zip_size = os.path.getsize("submit.zip") / (1024**2)



# ========== Main ==========
def main():
    print("\n[PHASE 1/3] 🔪 SCALE PRUNING")
    print("=" * 70)
    
    # Load
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    
    # Phase 1: Scale Pruning
    model = scale_pruning(model, ratio=PRUNING_RATIO)
    
    # Phase 2: Distillation
    model = distillation_phase(model, tokenizer)
    
    # Phase 3: Quantization
    model = quantization_phase(model, tokenizer)
    
    # Save
    save_model(model, tokenizer)


if __name__ == "__main__":
    main()
