# Aimers_8th
LG 2차 해커톤 진출을 목적으로 양자화 진행
해커톤 설명 : [Aimers 8기 : 모델 경량화 온라인 해커톤](https://dacon.io/competitions/official/236673/mysubmission)

## Introduction
[모델](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B)
LGAI-EXAONE4.0-1.2B를 양자화 시도
GPTQ W4A16 기반으로 시행

## Features
default.py, default.ipnyb: 대회에서 제공한 기본 코드

scale_pruning_distill.py: 2:4 sparsity pruning -> knowledge distillation -> GPTQ(W4A16)

ignore_dummy_GPTQ.py: 24개 레이어 다시 30개로 만들기 위해 더미 레이어 0으로 추가하여 대회 환경에 맞추기

nearly_zero.py: 더미 레이어 0 대신 0에 가까운 수로 변경

=> 추후 속도 증가를 위해 레이어를 줄였으나 다시 늘리는 방법은 틀렸다 생각하여 lm_head 중복 제거로 해결

awq_gptq.py: awq와 gptq를 조합하여 각 기법의 한계 해결

seed_equal.py: seed 변경해서 점수 올리기 도전

더 자세한 내용을 아래 블로그 참고
[블로그](https://blog.naver.com/deoduck92/224199586717)

## Run
각 python 파일 경로 수정 후 실행

## References
[1] [MARLIN](https://arxiv.org/abs/2408.11743)  
[2] [GPTQ](https://arxiv.org/abs/2210.17323)
