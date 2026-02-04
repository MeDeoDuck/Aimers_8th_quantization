"""
제출 전 종합 테스트 스크립트
실행: python test_before_submit.py --model ./model --zip exp6_submit.zip
"""

import os
import time
import torch
import zipfile
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_model_size(model_path):
    """모델 크기 계산"""
    total_size = 0
    for file in Path(model_path).rglob('*'):
        if file.is_file():
            total_size += file.stat().st_size
    return total_size / (1024**3)

def test_model_loading(model_path):
    """모델 로딩 테스트"""
    print("\n" + "="*50)
    print("1. 모델 로딩 테스트")
    print("="*50)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        print("✅ Tokenizer 로드 성공")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("✅ Model 로드 성공")
        
        return tokenizer, model, True
    except Exception as e:
        print(f"❌ 로딩 실패: {e}")
        return None, None, False

def test_inference(tokenizer, model):
    """추론 테스트"""
    print("\n" + "="*50)
    print("2. 추론 테스트")
    print("="*50)
    
    test_prompts = [
        "Hello, how are you?",
        "Explain machine learning.",
        "Write a poem about AI.",
    ]
    
    try:
        for i, prompt in enumerate(test_prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                )
            end = time.time()
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"\n테스트 {i+1}:")
            print(f"입력: {prompt}")
            print(f"출력: {generated[:100]}...")
            print(f"시간: {end-start:.2f}초")
        
        print("\n✅ 추론 테스트 통과")
        return True
    except Exception as e:
        print(f"❌ 추론 실패: {e}")
        return False

def test_model_size(model_path):
    """모델 크기 확인"""
    print("\n" + "="*50)
    print("3. 모델 크기 확인")
    print("="*50)
    
    size_gb = get_model_size(model_path)
    print(f"모델 크기: {size_gb:.2f} GB")
    
    # 기준 크기 (EXAONE-4.0-1.2B 원본은 약 2.4GB)
    if size_gb > 2.5:
        print("⚠️  원본보다 큼 (양자화 효과 없음?)")
    elif size_gb < 0.5:
        print("⚠️  너무 작음 (문제 있을 수 있음)")
    else:
        print("✅ 적정 크기")
    
    return size_gb

def validate_zip(zip_path):
    """제출 파일 검증"""
    print("\n" + "="*50)
    print("4. 제출 파일 검증")
    print("="*50)
    
    if not os.path.exists(zip_path):
        print(f"❌ {zip_path} 파일이 없습니다!")
        return False
    
    # 크기 확인
    size_gb = os.path.getsize(zip_path) / (1024**3)
    print(f"압축 파일 크기: {size_gb:.2f} GB")
    
    if size_gb > 10:
        print("❌ 10GB 초과! 제출 불가!")
        return False
    else:
        print("✅ 크기 제한 통과")
    
    # 내부 구조 확인
    with zipfile.ZipFile(zip_path, 'r') as zf:
        file_list = zf.namelist()
        
        # 필수 파일
        required = ['model/config.json']
        missing = [f for f in required if f not in file_list]
        
        if missing:
            print(f"❌ 필수 파일 누락: {missing}")
            return False
        
        # safetensors 확인
        safetensors = [f for f in file_list if 'safetensors' in f]
        if not safetensors:
            print("❌ safetensors 파일 없음!")
            return False
        
        print(f"✅ safetensors 파일: {len(safetensors)}개")
        
        # 구조 확인
        if all(f.startswith('model/') for f in file_list):
            print("✅ 디렉토리 구조 정상")
        else:
            print("❌ model/ 외부에 파일 있음!")
            return False
    
    print("✅ 제출 파일 검증 통과!")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='모델 경로')
    parser.add_argument('--zip', type=str, help='제출할 zip 파일 경로')
    args = parser.parse_args()
    
    print("="*50)
    print("제출 전 종합 테스트 시작")
    print("="*50)
    
    results = {
        'loading': False,
        'inference': False,
        'size': False,
        'zip': False,
    }
    
    # 1. 로딩 테스트
    tokenizer, model, results['loading'] = test_model_loading(args.model)
    
    # 2. 추론 테스트
    if results['loading']:
        results['inference'] = test_inference(tokenizer, model)
    
    # 3. 크기 테스트
    size_gb = test_model_size(args.model)
    results['size'] = (0.5 < size_gb < 2.5)
    
    # 4. ZIP 검증
    if args.zip:
        results['zip'] = validate_zip(args.zip)
    
    # 최종 결과
    print("\n" + "="*50)
    print("최종 결과")
    print("="*50)
    
    for test, passed in results.items():
        status = "✅ 통과" if passed else "❌ 실패"
        print(f"{test}: {status}")
    
    if all(results.values()):
        print("\n🎉 모든 테스트 통과! 제출 가능합니다!")
    else:
        print("\n⚠️  일부 테스트 실패. 확인 후 수정하세요.")

if __name__ == "__main__":
    main()
