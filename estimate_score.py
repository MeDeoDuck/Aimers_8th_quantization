"""
Score Estimator
실제 모델을 평가하여 예상 점수 계산

사용법:
    python estimate_score.py <model_dir>

예시:
    python estimate_score.py ./model_exp25
"""

import os
import sys
import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ==================== Configuration ====================
BASELINE_MODEL_PATH = "../Model/EXAONE-4.0-1.2B"
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
NUM_EVAL_SAMPLES = 100  # 평가 샘플 수 (빠른 추정)
MAX_NEW_TOKENS = 256    # 평가용


# ==================== Baseline Metrics (미리 계산) ====================
# 실제 baseline 모델로 측정한 값들
BASELINE_PERF = 1.0  # 기준 (normalized)
BASELINE_TIME_PER_TOKEN = 0.015  # 초당 토큰 (예시)
BASELINE_TOKENS_PER_SAMPLE = 150  # 평균 생성 토큰


def load_model_and_tokenizer(model_path):
    """모델과 토크나이저 로드"""
    print(f"\n모델 로딩: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"✓ 모델 로드 완료")
    
    # 모델 크기 계산
    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    print(f"   - 모델 크기: {param_size:.1f} MB")
    
    return model, tokenizer


def load_eval_dataset(tokenizer, num_samples=100):
    """평가 데이터셋 로드"""
    print(f"\n평가 데이터 로딩 ({num_samples} samples)...")
    
    # MANTA 데이터셋의 뒷부분 사용 (학습에 사용 안 한 부분)
    ds = load_dataset(
        DATASET_ID,
        split=f"train[-{num_samples}:]"  # 마지막 100개
    )
    
    eval_data = []
    for example in ds:
        # Chat template 적용
        prompt = tokenizer.apply_chat_template(
            example["conversations"][:-1],  # 마지막 답변 제외
            add_generation_prompt=True,
            tokenize=False
        )
        
        # 정답 (reference)
        reference = example["conversations"][-1]["content"]
        
        eval_data.append({
            "prompt": prompt,
            "reference": reference
        })
    
    print(f"데이터 로드 완료")
    return eval_data


def evaluate_performance(model, tokenizer, eval_data):
    """성능(PerfNorm) 평가"""
    print(f"\n성능 평가 중...")
    
    model.eval()
    
    total_score = 0.0
    total_samples = len(eval_data)
    
    for i, item in enumerate(eval_data):
        if i % 10 == 0:
            print(f"   진행: {i}/{total_samples}")
        
        # 생성
        inputs = tokenizer(item["prompt"], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # 생성된 텍스트
        generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        reference = item["reference"]
        
        # 간단한 점수 (실제는 더 복잡한 메트릭 사용)
        # ROUGE-L, BLEU 등을 사용하지만 여기서는 단순화
        score = calculate_simple_score(generated, reference)
        total_score += score
    
    avg_score = total_score / total_samples
    
    print(f"✓ 성능 평가 완료")
    print(f"   - 평균 점수: {avg_score:.4f}")
    
    return avg_score


def calculate_simple_score(generated, reference):
    """간단한 유사도 점수 (0-1)"""
    # 실제로는 ROUGE, BLEU 등 사용
    # 여기서는 단어 overlap으로 근사
    
    gen_words = set(generated.lower().split())
    ref_words = set(reference.lower().split())
    
    if len(ref_words) == 0:
        return 0.0
    
    overlap = len(gen_words & ref_words)
    recall = overlap / len(ref_words)
    precision = overlap / len(gen_words) if len(gen_words) > 0 else 0
    
    # F1 score
    if recall + precision == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def evaluate_speed(model, tokenizer, eval_data):
    """속도(SpeedNorm) 평가"""
    print(f"\n속도 평가 중...")
    
    model.eval()
    
    total_time = 0.0
    total_tokens = 0
    num_samples = min(20, len(eval_data))  # 속도 측정은 20개만
    
    for i, item in enumerate(eval_data[:num_samples]):
        if i % 5 == 0:
            print(f"   진행: {i}/{num_samples}")
        
        inputs = tokenizer(item["prompt"], return_tensors="pt").to(model.device)
        
        # 시간 측정
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        # 생성된 토큰 수
        generated_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        
        total_time += (end_time - start_time)
        total_tokens += generated_tokens
    
    avg_time_per_token = total_time / total_tokens if total_tokens > 0 else 0
    avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    
    print(f"✓ 속도 평가 완료")
    print(f"   - 평균 생성 시간: {total_time/num_samples:.3f} sec/sample")
    print(f"   - 평균 토큰/초: {avg_tokens_per_second:.1f}")
    print(f"   - 시간/토큰: {avg_time_per_token:.4f} sec")
    
    return avg_time_per_token, avg_tokens_per_second


def calculate_normalized_scores(perf_score, time_per_token):
    """Normalized 점수 계산"""
    
    # PerfNorm = model_perf / baseline_perf
    perf_norm = perf_score / BASELINE_PERF
    
    # SpeedNorm = 1 - (model_time / baseline_time)
    # 더 빠르면 높은 점수
    speed_ratio = time_per_token / BASELINE_TIME_PER_TOKEN
    speed_norm = max(0, 1 - speed_ratio)
    
    # 최종 점수
    final_score = 0.5 * perf_norm + 0.5 * speed_norm
    
    return perf_norm, speed_norm, final_score


def estimate_with_baseline_comparison(model_path):
    """Baseline과 비교하여 예상 점수 계산"""
    print("=" * 70)
    print("Score Estimator")
    print("=" * 70)
    
    # 모델 로드
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # 평가 데이터
    eval_data = load_eval_dataset(tokenizer, NUM_EVAL_SAMPLES)
    
    # 성능 평가
    perf_score = evaluate_performance(model, tokenizer, eval_data)
    
    # 속도 평가
    time_per_token, tokens_per_sec = evaluate_speed(model, tokenizer, eval_data)
    
    # Normalized 점수 계산
    perf_norm, speed_norm, final_score = calculate_normalized_scores(
        perf_score, time_per_token
    )
    
    # 결과 출력
    print("\n" + "=" * 70)
    print("📊 최종 결과")
    print("=" * 70)
    print(f"\n성능 메트릭:")
    print(f"   - Raw Score: {perf_score:.4f}")
    print(f"   - PerfNorm: {perf_norm:.4f}")
    
    print(f"\n속도 메트릭:")
    print(f"   - 시간/토큰: {time_per_token:.4f} sec")
    print(f"   - 토큰/초: {tokens_per_sec:.1f}")
    print(f"   - SpeedNorm: {speed_norm:.4f}")
    
    print(f"\n최종 점수:")
    print(f"   Score = 0.5 × {perf_norm:.4f} + 0.5 × {speed_norm:.4f}")
    print(f"   Score = {final_score:.4f}")
    
    # 파일로 저장
    result = {
        "model_path": model_path,
        "perf_score": float(perf_score),
        "perf_norm": float(perf_norm),
        "time_per_token": float(time_per_token),
        "tokens_per_sec": float(tokens_per_sec),
        "speed_norm": float(speed_norm),
        "final_score": float(final_score)
    }
    
    result_file = os.path.join(model_path, "estimated_score.json")
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✓ 결과 저장: {result_file}")
    print("=" * 70)
    
    return result


def quick_estimate(model_path):
    """빠른 추정 (간단한 메트릭만)"""
    print("=" * 70)
    print("Quick Score Estimator (빠른 추정)")
    print("=" * 70)
    
    print(f"\n모델 분석: {model_path}")
    
    # 모델 파일 크기
    model_files = []
    for root, dirs, files in os.walk(model_path):
        for file in files:
            if file.endswith(('.bin', '.safetensors')):
                filepath = os.path.join(root, file)
                size_mb = os.path.getsize(filepath) / (1024**2)
                model_files.append((file, size_mb))
    
    total_size = sum(size for _, size in model_files)
    
    print(f"   - 모델 크기: {total_size:.1f} MB")
    
    # 간단한 추정
    # 원본: 2400 MB
    # 압축률 기반 SpeedNorm 추정
    compression_ratio = 2400 / total_size if total_size > 0 else 1
    
    # 경험적 공식
    estimated_speed_norm = min(0.9, 0.2 + 0.15 * compression_ratio)
    estimated_perf_norm = 0.95 - 0.05 * (compression_ratio - 1)  # 압축할수록 약간 성능 하락
    estimated_score = 0.5 * estimated_perf_norm + 0.5 * estimated_speed_norm
    
    print(f"\n추정 결과 (간단):")
    print(f"   - 압축률: {compression_ratio:.2f}x")
    print(f"   - 예상 PerfNorm: ~{estimated_perf_norm:.3f}")
    print(f"   - 예상 SpeedNorm: ~{estimated_speed_norm:.3f}")
    print(f"   - 예상 Score: ~{estimated_score:.3f}")
    
    print(f"\n💡 더 정확한 추정을 위해서는:")
    print(f"   python estimate_score.py {model_path} --full")
    print("=" * 70)


# ==================== Main ====================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법:")
        print("  python estimate_score.py <model_dir> [--quick]")
        print("\n예시:")
        print("  python estimate_score.py ./model_exp25")
        print("  python estimate_score.py ./model_exp25 --quick  # 빠른 추정")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    if not os.path.exists(model_path):
        print(f"모델 경로가 존재하지 않습니다: {model_path}")
        sys.exit(1)
    
    # Quick mode
    if "--quick" in sys.argv:
        quick_estimate(model_path)
    else:
        # Full evaluation
        estimate_with_baseline_comparison(model_path)
