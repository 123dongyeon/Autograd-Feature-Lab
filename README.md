# Autograd Feature Lab

이 프로젝트는 `deep_learning_2025` 저장소의 예제들을 하나의 **실험형 프로젝트**로 엮은 것입니다. PyTorch의 텐서 속성, autograd, 그래디언트 흐름을 직접 만지고 관찰하며, 간단한 **특징 학습(feature learning)** 모델까지 이어집니다. 교재처럼 읽는 코드가 아니라, 실험실처럼 눌러보고 흔들어보는 구조입니다.

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

아래 순서대로 스크립트를 실행하며 출력과 gradient 변화를 관찰합니다. 각 스크립트는 **짧은 코드 + 명확한 실험 포인트**로 구성되어 있습니다.

### 1. 기본 backward와 스칼라 미분

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x

y.backward()
print(x.grad)  # dy/dx = 2x + 2 = 8
```

* `requires_grad=True`가 계산 그래프를 생성함
* 스칼라 출력에 대해 `backward()`는 자동으로 1을 곱함

### 2. Gradient 누적과 초기화

```python
x = torch.tensor(2.0, requires_grad=True)
y1 = x ** 2
y2 = 3 * x

y1.backward()
y2.backward()
print(x.grad)  # 2x + 3 = 7

x.grad.zero_()
y2.backward()
print(x.grad)  # 3
```

* `.grad`는 **누적(accumulate)** 된다
* 반복 학습 시 반드시 zeroing 필요

### 3. retain_graph의 의미

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3

for _ in range(2):
    y.backward(retain_graph=True)
    print(x.grad)
```

* 기본적으로 backward 이후 그래프는 삭제됨
* 여러 번 미분하려면 `retain_graph=True`

### 4. detach vs no_grad

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * 3

z = y.detach()
w = z ** 2
print(w.requires_grad)  # False

with torch.no_grad():
    u = x ** 2
```

* `detach()`는 **그래프에서 분리된 텐서** 생성
* `no_grad`는 **컨텍스트 전체에서 autograd 차단**

### 5. 벡터 출력과 VJP

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x ** 2
v = torch.tensor([1.0, 1.0])

y.backward(v)
print(x.grad)  # [2, 4]
```

* PyTorch는 Jacobian 전체가 아닌 **Vector-Jacobian Product**를 계산

출력이 이해되면 다음 파일로 이동하세요.

## 실험 2: Tensor Feature Trainer

`tensor_features` 폴더에서는 간단한 **특징 추출 + 지도 학습**을 수행합니다.

### 데이터 로딩 (`load_data.py`)

```python
import torch

def load_data(n=100):
    x = torch.linspace(-1, 1, n).unsqueeze(1)
    y = x ** 2 + 0.1 * torch.randn_like(x)
    return x, y
```

* 입력: 1차원 연속 값
* 타깃: 비선형 함수 + 노이즈

### Feature 추출 (`utils.py`)

```python
import torch

def feature_map(x):
    return torch.cat([x, x**2, x**3], dim=1)
```

* 선형 입력을 다항 feature로 확장
* 이후 모델은 **선형 회귀**지만 전체는 비선형 모델

### 학습 루프 (`train.py`)

```python
import torch
from load_data import load_data
from utils import feature_map

x, y = load_data()
phi = feature_map(x)

w = torch.randn(phi.size(1), 1, requires_grad=True)

lr = 0.1
for epoch in range(100):
    pred = phi @ w
    loss = ((pred - y) ** 2).mean()

    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        w.grad.zero_()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss {loss.item():.4f}")
```

### 관찰 포인트

* feature 차원이 늘어날수록 수렴 속도 변화
* `no_grad`가 없으면 업데이트가 그래프에 포함됨
* `.grad.zero_()`를 제거했을 때 발산 여부

### Optimizer 버전 (선택)

```python
optimizer = torch.optim.SGD([w], lr=0.1)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

수동 업데이트와 비교해 동작을 대조해보세요.

## 확장 아이디어

* feature extractor를 선형 → 비선형으로 변경
* batch VJP를 이용한 커스텀 loss 실험
* autograd를 끈 상태에서의 추론 코드 분리

## 대상 독자

* PyTorch를 "쓸 줄"은 알지만, "왜 이렇게 동작하는지"가 궁금한 사람
* autograd를 수식이 아닌 **현상**으로 이해하고 싶은 학생

## 한 줄 요약

> 이 프로젝트는 PyTorch autograd의 내부를 투명한 유리 상자처럼 들여다보는 실험실이다.
