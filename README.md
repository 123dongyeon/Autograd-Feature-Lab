# Autograd-Feature-Lab

## 프로젝트 목표

* PyTorch 텐서의 속성(data, shape, dtype, device, storage)을 손으로 확인한다.
* autograd의 작동 원리와 그래디언트 누적, 그래프 유지, 분리(detach)를 실험한다.
* 벡터-야코비안 곱(VJP)과 배치 VJP의 함정을 이해한다.
* 간단한 데이터셋에서 **특징 추출 + 미니 학습 루프**를 구성한다.

## 핵심 아이디어

> 작은 코드 조각들이 하나의 파이프라인으로 이어질 때, 추상적인 autograd가 촉각을 얻는다.

본 프로젝트는 다음 두 축으로 구성됩니다.

1. **Autograd Playground**: gradient 계산 관련 스크립트들을 실험 순서로 재배치
2. **Tensor Feature Trainer**: `tensor_features/` 모듈을 이용한 간단한 학습 실험

## 디렉터리 구조

```
.
├── 03_tensor_attributes_and_methods_*.py   # 텐서 속성과 연산 실험
├── 05_gradient_computation_*.py            # autograd와 backward 실험
├── tensor_features/
│   ├── config.py       # 실험 설정
│   ├── download.py     # 데이터 다운로드
│   ├── load_data.py    # 데이터 로딩
│   ├── train.py        # 학습 루프
│   └── utils.py        # 보조 함수
└── README.md
```

## 실행 환경

* Python 3.9+
* PyTorch (CUDA 선택)

```bash
pip install torch torchvision
```

## 실험 1: Autograd Playground

아래 순서대로 스크립트를 실행하며 출력과 gradient 변화를 관찰합니다.

1. 기본 backward와 스칼라 미분
2. gradient 누적과 초기화 시점
3. `retain_graph=True`의 의미
4. `detach`, `no_grad`의 차이
5. 벡터 backward와 VJP의 직관

```bash
python 05_gradient_computation_01_basic_scalar_backward.py
```

각 파일은 **하나의 질문**을 던지도록 설계되어 있습니다. 출력이 이해되면 다음 파일로 이동하세요.

## 실험 2: Tensor Feature Trainer

`tensor_features` 폴더에서는 간단한 특징 추출 + 학습을 수행합니다.

```bash
python tensor_features/train.py
```

### 무엇을 관찰할까?

* 입력 텐서가 feature로 변환되는 흐름
* optimizer 없이 수동 업데이트 vs optimizer 사용
* gradient accumulation을 사용할 때의 학습 안정성

## 확장 아이디어

* feature extractor를 선형 → 비선형으로 변경
* batch VJP를 이용한 커스텀 loss 실험
* autograd를 끈 상태에서의 추론 코드 분리

## 대상 독자

* PyTorch를 "쓸 줄"은 알지만, "왜 이렇게 동작하는지"가 궁금한 사람
* autograd를 수식이 아닌 **현상**으로 이해하고 싶은 학생

## 한 줄 요약

> 이 프로젝트는 PyTorch autograd의 내부를 투명한 유리 상자처럼 들여다보는 실험실이다.
