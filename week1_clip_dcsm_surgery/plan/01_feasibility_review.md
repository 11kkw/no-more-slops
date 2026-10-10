# Week 1 — DCSM 기반 제로샷 객체 탐지: 재현성 리뷰

작성일: 2026-10-10 · 대상 코드: `../baseline` (Kang et al., "Is CLIP ideal? No. Can we fix it? Yes!", ICCV 2025 공식 구현)
실험 코드·결과: `../experiment`

## 0. 실험 목표

### 0.1 아이디어의 논리

DCSM(Dense Cosine Similarity Map)은 CLIP의 전역 임베딩 하나를 버리고, **이미지 패치 토큰 197개 × 텍스트 토큰 30개의 코사인 유사도 행렬**을 표현으로 쓴다. 원 논문은 이 행렬 위에 작은 CNN scorer를 얹어 속성 결합·공간 관계·부정 같은 조합 추론을 개선했다. 이 표현을 객체 탐지에 쓰자는 아이디어는 다음 세 단계 논리에 기댄다.

1. **DCSM 행렬은 이미 공간 격자 위에 정의되어 있다.** 행 축이 14×14 패치 격자이므로, 텍스트 토큰 하나를 골라 열을 읽으면 그 토큰에 대한 히트맵이 바로 나온다. 즉 DCSM은 "어떤 패치가 어떤 단어와 닮았는가"를 담은 dense한 지도이고, 객체 탐지는 이 지도에서 영역을 잘라 레이블을 붙이는 일이다.
2. **텍스트 쪽이 자유 어휘다.** CLIP 텍스트 인코더를 그대로 쓰므로 클래스 이름이든 "red car"든 임의 문장을 열로 넣을 수 있다. 따라서 탐지 어휘를 학습 시점에 고정할 필요가 없고, 제로샷·open-vocabulary 탐지로 자연스럽게 이어진다.
3. **CLIP 인코더는 동결하고 읽기 방식만 바꾼다.** 원 논문의 기여도 CLIP을 재학습하지 않고 표현을 다르게 읽는 것이었다. 같은 정신으로, 검출을 위한 추가 학습 없이 "CLIP 마지막 블록을 어떻게 읽어야 DCSM 행렬이 분류 신호로 쓸 만해지는가"를 먼저 따진다.

### 0.2 평가 방법

분류 신호와 위치 신호를 분리해 평가한다. 모두 학습 없이(training-free) 수행한다.

- **실측 1 — localization 프로브** (`experiment/scripts/probe_localization.py`): 정답 클래스를 알려준 상태에서 해당 클래스 열의 히트맵이 GT 박스를 가리키는지 본다. pointing accuracy, 히트맵 임계값 박스의 IoU≥0.5 적중률, 이미지 수준 multi-label mAP. 우연 수준과 CLIP 전역 CLS를 기준선으로 둔다.
- **실측 2 — 제로샷 탐지** (`experiment/scripts/dcsm_detection_eval.py`): COCO 80 클래스 전체를 쿼리로 넣고 표준 COCO AP를 잰다. 두 프로토콜을 둔다.
  - **P1 dense**: 패치별 클래스 softmax → 임계값 → 연결성분 → 박스. DCSM만으로 분류와 위치를 모두 해결하는 경우.
  - **P2 proposals**: OWLv2 objectness 헤드의 class-agnostic 박스 100개를 빌려 쓰고, 박스 안 패치의 DCSM 클래스 맵을 평균 풀링해 분류한다. OWLv2 자체 분류 헤드와 **같은 박스**를 쓰므로 둘의 차이가 순수하게 분류 신호 품질을 격리한다.
- **변인**: 마지막 ViT 블록 readout 3종 (naive = 원 코드 경로, MaskCLIP = value만, ClearCLIP = q-q attention에 residual·FFN 제거) × 텍스트 readout 2종 (DCSM 토큰별 max, CLIP EOS) × 해상도 2종 (224, 448).
- **실측 3 — 조합 쿼리 referring-detection** (3b절, 2026-10-10 확장): 같은 클래스 후보 박스 중 속성(size, color)·공간 관계·부정 쿼리가 가리키는 인스턴스를 고르는 top-1 과제. 원 논문의 사전학습 DCSM scorer를 포함해 12개 scorer를 비교하고, 크기 편향을 통제한 균형 정확도로 보고한다.

### 0.3 예상되는 문제

- **반전된 유사도**: DCSM 논문 자체가 "객체를 설명하는 토큰은 해당 패치와 낮은 코사인 유사도를 가진다"는 현상을 언급하고, 이를 고치지 않은 채 scorer CNN이 패턴을 학습하도록 둔다. 전역 점수 하나를 낼 때는 무해하지만 박스를 뽑을 때는 객체 영역이 오히려 어두워져 localization이 깨질 수 있다. 원인은 CLIP 마지막 블록의 residual 경로와 FFN이 패치 토큰을 전역 표현 쪽으로 끌어당기기 때문이며, MaskCLIP·ClearCLIP 계열이 같은 문제를 다룬 바 있다.
- **박스 품질**: 14×14 격자(224px)에서는 한 패치가 16픽셀이라 소형 객체는 격자 하나에도 못 미친다. 448px로 올려도 interpolated positional embedding이 분포를 벗어난다. P1만으로는 AP가 낮을 것이 예상되므로 P2로 분류 신호를 따로 본다.
- **closed-set 검출기와의 비교는 불공정하다**: YOLO·Faster R-CNN은 COCO 박스로 직접 학습했고 어휘가 고정돼 있다. 동결 CLIP에 학습 없는 헤드를 얹은 모델이 이를 이기는 결과는 나오지 않으며, 이긴다고 해도 의미가 다르다. 비교 대상은 같은 제로샷 조건의 open-vocabulary 검출기여야 한다 (1절).
- **베이스라인 코드의 재현성**: Windows 경로 하드코딩, 197×30에 고정된 scorer, 배치 통계에 의존하는 표준화, train/eval 간 special-token 슬라이싱 불일치 등 검출 실험 전에 걷어내야 할 부분이 있다 (4절).

## 1. 벤치마크와 비교 대상 선정 기준

### 1.1 벤치마크: COCO val2017, 80 클래스 제로샷

- **선정 이유**: 아이디어의 주장은 "검출용 학습 없이 CLIP 표현만으로 탐지한다"이므로, 평가 데이터로 학습한 적이 없는 상태에서 표준 AP를 잴 수 있는 데이터셋이 필요하다. COCO val2017은 (a) 비교 대상인 OWL-ViT, YOLO-World, Grounding DINO가 모두 COCO 제로샷 수치를 보고하고, (b) pycocotools로 AP·AP50·AP75·APs/m/l을 그대로 계산할 수 있으며, (c) 5,000장이라 frozen CLIP 추론을 포함한 전체 평가가 GPU 1장에서 1시간 안에 끝난다.
- **주 지표**: COCO AP(0.5:0.95)와 AP50. P2에서는 OWLv2 native와 같은 박스 집합을 쓰므로 AP 차이가 곧 분류 신호 차이다.
- **보조 지표** (실측 1): pointing accuracy, IoU≥0.5 recall, image-level mAP. 탐지 AP 하나로는 분류 실패와 위치 실패를 구분할 수 없어서 둔다.
- **제외한 벤치마크**: LVIS(1,203 클래스)는 텍스트 쿼리 수가 많아 DCSM 행렬 계산이 80배 커지고 이번 주 범위를 넘는다. OV-COCO 48/17 split은 base 클래스 학습을 전제로 하므로 학습 없는 이번 설정에는 맞지 않다. 둘 다 다음 주 이후 후보로 남긴다.

### 1.2 비교 대상: open-vocabulary 검출기

| 선정 기준 | 설명 |
|---|---|
| 같은 과제 정의 | 텍스트 쿼리를 받아 박스를 내는 open-vocabulary 검출기여야 한다. closed-set 검출기는 참고 상한으로만 적는다. |
| 같은 백본 계열 | OWL-ViT는 CLIP ViT를 백본으로 쓰므로 "같은 인코더를 어떻게 쓰느냐"의 비교가 된다. OWLv2는 그 위에 self-training을 더한 현재 공개 모델 중 가장 강한 CLIP 계열이다. |
| 로컬 재현 가능 | HF transformers로 같은 환경에서 같은 5,000장에 직접 돌릴 수 있어야 한다. 논문 수치만 옮겨 적으면 전처리·해상도 차이로 비교가 흐려진다. |
| 박스 공유 가능 | P2 프로토콜을 위해 class-agnostic objectness와 박스를 내주는 모델이 필요하다. OWLv2가 이를 제공한다. |

- **직접 실행**: OWL-ViT v1 (`google/owlvit-base-patch32`), OWLv2 (`google/owlv2-base-patch16-ensemble`).
- **문헌 수치만 인용**: YOLO-World, Grounding DINO. 둘 다 대규모 grounding 데이터로 검출 헤드를 학습했으므로 "동결 CLIP + 학습 없는 헤드"와 직접 비교하면 불리할 뿐이며, 제로샷 상한선으로만 적는다.
- **참고 상한**: YOLO11x, Faster R-CNN. 사용자가 처음 제안한 비교군이지만 1.2의 "같은 과제 정의" 기준에 맞지 않아 표에만 남긴다.

## 2. 실측 1 — DCSM 패치 표현의 localization 품질 (`experiment/results/probe_500`)

COCO val2017 앞 500장, frozen CLIP ViT-B/16, 정답 클래스를 알려준 상태(oracle class)에서 측정.
pointing = 히트맵 argmax가 GT 박스 안에 들어간 비율, IoU≥0.5 = 히트맵 임계값 박스 적중률,
crude AP50 = 학습 없이 패치별 softmax → 연결성분 → 박스로 pycocotools 측정.

| 마지막 블록 readout | 텍스트 readout | pointing 224/448px | IoU≥0.5 224/448 | image-level mAP | crude AP50 224/448 |
|---|---|---|---|---|---|
| naive = DCSM 코드 경로 | EOS | 12.0 / 11.8 | 5.1 / 6.5 | 0.43 | 1.8 / 2.4 |
| naive = DCSM 코드 경로 | 토큰별 max (DCSM 행렬) | 10.4 / 12.2 | 4.9 / 5.1 | 0.10 | 0.2 / 0.1 |
| MaskCLIP (value만) | EOS | 51.4 / 57.1 | 21.1 / 25.1 | 0.65 | 12.7 / 14.9 |
| ClearCLIP (q-q attn, residual·FFN 제거) | EOS | 62.4 / 67.5 | 26.3 / 29.9 | 0.66 | 12.1 / 13.5 |
| ClearCLIP | 토큰별 max (DCSM 행렬) | 52.5 / 56.8 | 22.8 / 27.7 | 0.59 | 11.0 / 13.8 |
| 우연 수준 | | 17.1 | | 0.04 | |
| CLIP 전역 CLS (참고) | | | | 0.55 | |

- naive readout은 우연(17.1%)보다 낮다. 히트맵(`results/probe_500/qual_bus_pooled.png`)을 보면 객체 영역이 오히려 낮은 유사도로 반전된다. 0.3에서 예상한 문제가 정량적으로 확인된 것이다. 전역 점수 하나를 내는 데는 괜찮지만 박스를 뽑는 데는 치명적이다.
- 원시 DCSM 행렬의 image-level mAP 0.10은 우연 0.04에 가깝다. 원 논문 구조에서는 2층 CNN scorer가 반전 패턴을 복원해 주는 셈이다.
- readout만 바꾸면 DCSM 행렬이 분류와 위치를 모두 가진다. ClearCLIP readout의 DCSM 행렬은 image-level mAP 0.59로 CLIP 전역 CLS 0.55보다 높다.

## 3. 실측 2 — frozen CLIP + DCSM 행렬 분류 신호 × 3 readout의 제로샷 COCO 탐지 (`experiment/results/full_val2017`)

프로토콜과 변인은 0.2 참조. 비교 대상은 1.2의 OWL-ViT v1, OWLv2 텍스트 쿼리 제로샷.

### 3.1 스모크 테스트 (앞 50장, `results/smoke_50/summary.md`)

| config | AP | AP50 | APs | APl |
|---|---|---|---|---|
| owlvit_v1_native | 40.2 | 61.4 | 16.7 | 62.8 |
| owlv2_native (분류 헤드 포함) | 57.7 | 76.8 | 37.9 | 76.8 |
| P2 OWLv2 박스 + naive / dcsm_tok / 224 | 6.1 | 6.8 | 0.2 | 18.0 |
| P2 OWLv2 박스 + maskclip / dcsm_tok / 224 | 32.5 | 41.7 | 13.2 | 70.0 |
| P2 OWLv2 박스 + clearclip / dcsm_tok / 224 | 35.4 | 44.0 | 14.9 | 69.5 |
| P2 OWLv2 박스 + clearclip / pooled / 224 | 39.8 | 50.6 | 17.3 | 70.4 |
| P1 dense, 최고치 (maskclip / pooled / 224) | 11.6 | 23.3 | 0.2 | 24.4 |

- 파이프라인 검증용 실행이다. 앞쪽 50장이 쉬운 이미지라 절대값은 전체 세트(3.2)보다 10~13 AP 높게 나오며, 설정 간 상대 순서만 참고한다.
- 같은 박스 위에서 naive readout은 AP 6, MaskCLIP·ClearCLIP은 AP 32~40. readout surgery 하나로 분류 신호가 살아난다는 경향은 전체 세트에서도 유지된다.
- P1 dense는 AP 12 이하. 위치 신호까지 DCSM에서 뽑으려면 별도 헤드가 필요하다.

### 3.2 전체 5,000장 결과

COCO val2017 전체(어노테이션이 있는 4,952장), 80 클래스 쿼리. 전체 표는 `results/full_val2017/summary.md`.
P2는 모든 행이 OWLv2 박스 100개/장(총 1,485,600개)을 공유한다.

| config | AP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| owlvit_v1_native | 28.8 | 45.4 | 30.5 | 11.9 | 30.6 | 48.1 |
| **owlv2_native** (박스 + 자체 분류 헤드) | **46.8** | **66.7** | 51.2 | 32.9 | 49.8 | 62.7 |
| P2 · naive · dcsm_tok · 224 | 1.2 | 1.6 | 1.2 | 0.2 | 0.5 | 7.7 |
| P2 · naive · pooled · 224 | 2.6 | 3.4 | 2.8 | 0.3 | 0.9 | 8.3 |
| P2 · maskclip · dcsm_tok · 224 | 20.3 | 28.0 | 22.6 | 7.6 | 23.8 | 40.2 |
| P2 · maskclip · pooled · 224 | 26.0 | 35.6 | 28.8 | 10.1 | 28.5 | 45.8 |
| P2 · clearclip · dcsm_tok · 224 | 22.8 | 31.0 | 25.2 | 8.4 | 24.7 | 41.6 |
| P2 · clearclip · pooled · 224 | **27.3** | **37.3** | 30.2 | 11.2 | 29.2 | 46.5 |
| P2 · maskclip · dcsm_tok · 448 | 20.9 | 29.3 | 23.0 | 10.9 | 25.3 | 38.5 |
| P2 · maskclip · pooled · 448 | 26.7 | 37.2 | 29.4 | 13.6 | 29.7 | 43.7 |
| P2 · clearclip · dcsm_tok · 448 | 21.2 | 29.0 | 23.3 | 9.6 | 22.5 | 38.8 |
| P2 · clearclip · pooled · 448 | 26.4 | 36.3 | 29.0 | 12.8 | 27.0 | 44.2 |
| P1 dense, 최고치 (maskclip · pooled · 448) | 4.6 | 11.1 | 3.2 | 0.2 | 2.7 | 10.6 |
| P1 dense, naive 전체 | ≤0.7 | ≤1.8 | | | | |

해석:

- **readout surgery가 결정적이다.** 같은 OWLv2 박스 위에서 naive readout(원 DCSM 코드 경로)은 AP 1.2~2.7로 사실상 분류를 못 한다. 마지막 블록을 MaskCLIP·ClearCLIP 식으로 읽으면 AP 20~27, AP50 28~37로 올라간다. 실측 1에서 본 반전 현상이 탐지 AP로 그대로 이어진다.
- **DCSM 토큰별 max 읽기는 EOS 임베딩보다 약 5 AP 낮다.** (clearclip 224: 22.8 vs 27.3). 단일 클래스 이름 쿼리에서는 "a photo of a {}." 프롬프트의 여러 토큰 중 최대값을 취하는 방식이 EOS 하나보다 잡음이 많다. DCSM 행렬의 장점은 조합 쿼리에서 나와야 하며, 클래스 이름만 넣는 COCO 제로샷에서는 득이 없다. 이는 다음 주 실험(속성·공간·부정 쿼리)의 동기가 된다.
- **학습 없는 DCSM 분류와 OWLv2 자체 헤드의 격차는 약 20 AP다** (27.3 vs 46.8). 대형 객체에서는 46.5 vs 62.7로 좁고, 소형 객체에서는 11.2 vs 32.9로 크다. 448px는 APs를 2~3점 올리지만 APl을 깎아 전체 AP는 거의 같다.
- **ClearCLIP과 MaskCLIP의 차이는 1~2 AP로 작다.** readout 중 하나를 고른다면 구현이 단순한 MaskCLIP(value만)으로 충분하다.
- **P1 dense는 AP 5 이하.** DCSM 히트맵만으로 박스까지 뽑는 것은 현 상태로는 쓸 수 없고, proposal 또는 학습된 박스 헤드가 필요하다.
- **스모크 테스트(50장)는 전체보다 10~13 AP 높게 나왔다** (P2 clearclip pooled 39.8 → 27.3, OWLv2 57.7 → 46.8). 앞쪽 50장이 쉬운 이미지였기 때문이며, 상대 순서는 그대로다. 논문에는 전체 수치만 쓴다.
- OWL-ViT v1 B/32의 28.8 AP는 OWL-ViT 논문 Table A4의 B/16 31.7 AP와 같은 범위다(백본이 다르므로 직접 비교는 아님). 평가 파이프라인이 문헌과 어긋나지 않음을 확인하는 용도.

## 3b. 스펙 확장 — 조합 쿼리(속성 결합·공간 관계·부정) 검출 평가 (2026-10-10 추가)

### 3b.1 목표와 논리

실측 2는 클래스 이름만 넣는 쿼리라 DCSM 행렬의 이점이 드러날 여지가 없었다(EOS 읽기가 토큰별 읽기보다 5 AP 높음). DCSM 원 논문의 주장은 "CLIP의 전역 임베딩은 속성 결합·공간 관계·부정을 표현할 수 없고, 패치×토큰 행렬 위의 scorer는 할 수 있다"는 것이다. 이 주장이 **검출 단위에서도 성립하는지** 묻는다. 즉 같은 클래스의 후보 박스가 여러 개 있을 때, "big dog", "dog to the left of a person", "person without a handbag" 같은 쿼리로 올바른 인스턴스를 고를 수 있는가.

### 3b.2 벤치마크 설계 (`experiment/scripts/build_comp_queries.py`)

- **원천**: COCO val2017 어노테이션만 사용한다(추가 라벨 없음). 모든 항목은 (이미지, 쿼리, 같은 클래스 A의 후보 박스 2~6개, 정답 1개)로 구성된다. 후보가 모두 같은 클래스이므로 클래스 이름만으로는 풀 수 없고, 속성·관계·부정을 읽어야만 정답을 고를 수 있다.
- **지표**: 후보 중 top-1 정확도. 우연 수준 = mean(1/N). 각 subset 결과에 우연을 함께 적는다.
- **쿼리 형식**: DCSM 학습 템플릿("A picture of a {A} to the left of a {B}", "... without a {B}")에 맞춘 소문자 문장을 모든 scorer에 동일하게 입력한다. special word(left/right/above/below/big/small/without)는 공백 분리 토큰으로 등장해야 DCSM scorer의 열 치환이 작동한다.

| subset | 쿼리 | 정답 정의 | 항목 수 | 우연 |
|---|---|---|---|---|
| size | a picture of a big/small {A} | 면적이 다른 모든 A의 3배 이상(또는 1/3 이하) | 655 | 43.1 |
| color | a picture of a {color} {A} | 침식한 인스턴스 마스크의 HSV 주색(≥55%)이 모든 distractor에서 <12% | 142 | 44.1 |
| spatial | a picture of a {A} to the left of / right of / above / below a {B} | B는 이미지에 유일; 정확히 하나의 A만 그쪽, 나머지는 반대쪽 | 527 | 45.6 |
| negation | a picture of a {A} without a {B} | B가 붙은(박스 교집합 ≥ 작은 쪽 30% 또는 중심 포함) A가 하나만 없음 | 315 | 47.4 |
| affirm | a picture of a {A} with a {B} | 대조군: B가 붙은 A가 하나뿐 | 194 | 33.4 |

- **자동 라벨의 한계**: 20개 무작위 spot-check(`data/comp_queries/spotcheck_*.png`) 기준 spatial 20/20, negation 19/20, size 20/20, color 약 16/20이 정확하다. color는 마스크 배경 번짐 때문에 하늘 배경 클래스(airplane, kite, bird)와 대형 가구를 제외하고 마스크를 3픽셀 침식했는데도 잡음이 남는다. 논문에서는 color 결과를 참고치로만 쓰거나, 수동 검수한 부분집합을 따로 보고한다.
- **negation 의미 제약**: "bottle without a dining table"처럼 기하학적으로는 맞지만 어색한 쌍을 막기 위해 A는 host(person, chair, table, 탈것, 동물), B는 portable(가방, 식기, 운동기구, 사람 등)로 제한했다.
- **클래스 편중 방지**: subset×클래스당 60개, 이미지당 2개 상한.

### 3b.3 비교 대상(scorer) (`experiment/scripts/compositional_eval.py`)

모두 이 벤치마크로는 학습하지 않는다. 후보별 점수를 내는 방식만 다르다.

| scorer | 입력 | 점수 |
|---|---|---|
| owlv2 / owlvit_v1 | 전체 이미지 + 쿼리 문장 | 후보와 IoU≥0.5인 검출의 최대 logit |
| clip_crop | 후보 박스 crop(10% 여백) | CLIP 전역 임베딩 · 쿼리 EOS 임베딩 |
| clip_ctx | 후보 ∪ B 박스 crop | 위와 같음. spatial·negation에서 B가 포함된 문맥을 보게 함 |
| dcsm_obja_ctx / dcsm_coco_ctx | 후보 ∪ B 박스 crop | **원 논문의 사전학습 scorer CNN** (공개 체크포인트 2종). naive 197×30 행렬, special word 열 삽입, 배치 표준화까지 저장소 eval 코드 그대로 |
| readout/{naive,maskclip,clearclip}/{pooled,dcsm_tok} | 전체 이미지 224px 히트맵 | 후보 박스 안 패치 평균 (실측 2의 P2와 같은 방식) |

- OWL 계열은 "같은 과제 정의·같은 백본 계열·로컬 재현"의 1.2 기준에 따라 비교 대상이다. 다만 둘 다 grounding 데이터로 학습되었으므로 관계·부정에 어느 정도 노출되었을 수 있다.
- clip_crop은 ViLD식 region 분류의 대리이며, crop이 B를 잘라내므로 spatial·negation을 원리상 풀 수 없다. clip_ctx는 같은 문맥을 주었을 때 CLIP 전역 임베딩이 관계를 읽는지 본다(DCSM 논문의 "CLIP은 못 한다" 주장의 검출판 재현).
- dcsm_*_ctx가 핵심 비교다. 원 논문 scorer는 이미지 단위 점수만 내므로 후보별 문맥 crop으로 "지역화"했다. 체크포인트의 CLIP 가중치는 HF 원본과 완전히 동일함을 확인했다(frozen).

### 3b.4 예상되는 문제

- **DCSM scorer의 분포 이동**: 합성 Objaverse 렌더·COCO-spatial 쌍으로 학습된 scorer를 실사 crop에 적용한다. 이미지 단위 매칭에서 검증된 성능이 crop 단위 순위 매기기로 옮겨지지 않을 수 있다.
- **표준화의 배치 의존성**: 저장소의 `standardize()`는 배치 전체 통계를 쓴다. 후보 crop들을 한 배치로 묶어 쿼리마다 표준화하므로, 같은 crop의 점수가 함께 넣은 후보에 따라 달라진다. 순위 비교에는 영향이 제한적이나 절대 점수 비교는 불가하다.
- **special word 위치 오프셋**: 저장소는 공백 분리 단어 인덱스를 CLIP 토큰 열 인덱스로 그대로 쓴다. 다단어 클래스("dining table")에서는 열이 한 칸씩 밀린다. 학습 때도 같았으므로 그대로 둔다.
- **full-image readout은 관계를 읽을 수 없다**: 토큰별 max나 EOS 히트맵을 박스 안에서 평균내면 "dog" 패치가 어느 dog든 비슷하게 밝다. spatial·negation에서 우연 수준이 예상되며, 이는 "surgery만으로는 조합 추론이 안 되고 scorer가 필요하다"는 결론의 근거가 된다.
- **affirm 대조군의 역할**: negation에서 우연 이하가 나오고 affirm에서 높게 나오면, scorer가 부정어를 무시하고 B의 존재에 끌린다는 뜻이다(DCSM 논문이 CLIP에 대해 보고한 현상).

### 3b.5 결과 (`experiment/results/comp_full`)

1,833개 항목 전체. 원본 표 `summary.md`(subset·관계·색상별 분해 포함), 휴리스틱 `heuristics.md`, 균형 정확도 `summary_balanced.md`.

**(a) 크기 편향 발견.** 모델 없이 박스 크기만 보는 휴리스틱이 subset에 따라 우연과 크게 다르다. 후보 대부분이 N=2라 필터링으로는 편향을 못 없애므로, **크기 균형 정확도**(정답이 최대 박스인 항목과 아닌 항목의 정확도 평균; size는 big/small 평균)를 주 지표로 쓴다. 순수 크기 휴리스틱은 정확히 50이 된다.

| 휴리스틱 (raw top-1) | size | color | spatial | negation | affirm |
|---|---|---|---|---|---|
| 우연 (mean 1/N) | 43.1 | 44.1 | 45.6 | 47.4 | 33.4 |
| 가장 큰 박스 | 87.3 | 63.4 | 44.2 | 23.5 | 60.3 |
| 가장 작은 박스 | 12.5 | 26.8 | 47.1 | 72.7 | 16.5 |

size는 big 항목이 575/655라 큰 박스 휴리스틱이 87을 맞추고, negation은 "B가 붙지 않은 A"가 배경의 작은 인스턴스인 경우가 많아 작은 박스 휴리스틱이 73을 맞춘다. spatial만 크기 편향이 없다.

**(b) 크기 균형 top-1 정확도** (괄호: 정답이 최대 박스인 항목 / 아닌 항목; size는 big / small)

| scorer | size | color | spatial | negation | affirm |
|---|---|---|---|---|---|
| 항목 수 (최대 / 아님) | 575 / 80 | 90 / 52 | 233 / 294 | 74 / 241 | 117 / 77 |
| 우연 (균형) | 37.6 | 43.5 | 45.7 | 48.1 | 33.0 |
| 크기 휴리스틱 | 50 | 50 | 50 | 50 | 50 |
| owlv2 | 41.7 (72/11) | **73.5** (78/69) | 43.2 (49/37) | 39.5 (45/34) | 45.3 (53/38) |
| owlvit_v1 | 44.4 (78/11) | 63.9 (78/50) | 43.8 (56/31) | 41.8 (55/28) | 44.2 (60/29) |
| clip_crop | 36.8 (49/25) | 69.2 (71/67) | 44.2 (48/41) | 53.0 (54/52) | 38.3 (43/34) |
| clip_ctx | 36.8 (49/25) | 69.2 (71/67) | 39.0 (37/41) | 34.6 (36/33) | 46.2 (44/48) |
| dcsm_obja_ctx (원 논문 scorer) | 38.2 (45/31) | 65.7 (62/69) | **58.1** (60/56) | 38.7 (39/38) | **47.8** (46/49) |
| dcsm_coco_ctx (원 논문 scorer) | 38.8 (53/25) | 61.7 (60/63) | **57.3** (60/54) | 37.5 (36/39) | 46.0 (48/44) |
| readout/naive/pooled | 38.5 (43/34) | 21.3 (31/12) | 50.3 (58/42) | 48.0 (55/41) | 26.4 (31/22) |
| readout/naive/dcsm_tok | 38.5 (51/26) | 37.5 (42/33) | 50.0 (55/46) | 55.7 (59/52) | 36.3 (40/32) |
| readout/maskclip/pooled | 40.7 (49/32) | 62.1 (59/65) | 39.4 (40/38) | 51.1 (49/54) | 36.5 (39/34) |
| readout/maskclip/dcsm_tok | **44.5** (54/35) | 64.8 (62/67) | 43.4 (44/43) | 49.9 (51/49) | 36.3 (43/30) |
| readout/clearclip/pooled | 39.8 (43/36) | 57.6 (56/60) | 39.4 (37/41) | 48.8 (39/59) | 42.2 (40/44) |
| readout/clearclip/dcsm_tok | 43.4 (44/42) | 63.2 (61/65) | 42.5 (43/42) | 48.1 (42/54) | 35.5 (33/38) |

spatial 관계별 (raw top-1): dcsm_obja_ctx left 61.2 / right 61.3 / above 40.3 / below 51.9; dcsm_coco_ctx 62.1 / 54.8 / 41.9 / 59.6; owlv2 37.9 / 44.6 / 46.8 / 50.0.

**(c) 해석**

- **공간 관계는 원 논문의 DCSM scorer만 읽는다.** 두 체크포인트 모두 균형 57~58로 우연(45.7)보다 12점 높고, 정답이 큰 박스든 작은 박스든 일관된다(60/56). OWLv2·OWL-ViT(43), CLIP 문맥 crop(39), 모든 readout 히트맵(39~50)은 우연 이하다. DCSM 논문의 "CLIP 전역 임베딩은 공간 관계를 표현하지 못한다"는 주장이 **검출 단위에서 재현**되고, scorer가 그것을 고친다는 주장도 left/right(61~62)에서 재현된다. above/below(40~60)는 약하다. 이것이 이번 주의 핵심 positive finding이다.
- **부정은 아무도 읽지 못한다.** 균형 negation에서 우연(48.1)을 유의하게 넘는 scorer가 없다. OWL(40~42), DCSM scorer(38), clip_ctx(35)는 오히려 우연 이하이며, 같은 모델들이 affirm("with a B")에서는 우연(33)보다 12~15점 높다. 즉 "without"을 무시하고 B의 존재에 끌린다. DCSM 논문이 CLIP에 대해 보고한 현상이 **사전학습 DCSM scorer에서도 그대로** 나타난다. 원 논문의 negation 학습은 이미지 쌍 수준("image of A without B")이라 지역 단위 부정으로 전이되지 않는 것으로 보인다.
- **속성 결합(색)은 전역 임베딩으로도 충분하다.** OWLv2 73.5, clip_crop 69.2, DCSM scorer 62~66, surgery readout 58~65. 후보 crop만 보면 되는 과제라 CLIP 전역 임베딩이 가장 강하다. naive readout(21~38)은 우연 아래이며, 실측 1·2의 반전 현상이 여기서도 나타난다. 단 color는 자동 라벨 정확도 약 80%가 상한이다.
- **크기("big/small")는 아무도 읽지 않는다.** OWL은 big 72~78 / small 11로 순수 큰-박스 선호이고, 나머지도 우연 근방이다. clearclip/dcsm_tok만 44/42로 big·small에 대칭적이다. 균형 지표 없이는 OWL의 size 65~70이 "속성 이해"로 오독될 뻔했다.
- **readout surgery는 조합 추론을 만들지 않는다.** MaskCLIP·ClearCLIP 히트맵은 색에서는 naive 대비 +30~40점이지만 spatial·negation·affirm에서는 우연 근방이다. 3b.4의 예상대로 surgery는 "어디에 무엇이 있는가"를 복원할 뿐이며, 관계·부정에는 학습된 scorer가 필요하다.
- **OWL 계열은 조합 쿼리에 약하다.** 색을 제외하면 모든 subset에서 우연 이하다. grounding 데이터로 학습했음에도 문장의 관계·부정 구조를 쓰지 않는다는 뜻이며, 비교 대상으로서 "open-vocabulary 검출기도 조합 추론은 못 한다"는 배경을 제공한다.

## 3c. 스펙 확장 2 — readout surgery 위에서 DCSM scorer 재학습 (2026-10-10 추가)

### 3c.1 동기와 결정 사항

3b에서 사전학습 DCSM scorer는 공간 관계를 읽지만 naive readout(반전된 패치 표현) 위에서 학습되었고, surgery readout 히트맵은 위치는 잘 잡지만 관계를 읽지 못했다. 둘을 합쳐 **surgery readout(MaskCLIP/ClearCLIP) 행렬 위에 scorer를 다시 학습**하면 관계 추론과 localization이 동시에 개선되는지 본다. 사용자 결정(2026-10-10):

| 결정 | 선택 | 이유 |
|---|---|---|
| 학습 데이터 | 원 논문 Drive 데이터, 레시피 그대로 | readout 변수만 격리. COCO val2017 평가 벤치마크와 데이터가 겹치지 않음 |
| 학습 단위 | 이미지 단위(원 논문과 동일) | 영역 단위로 바꾸려면 COCO train2017 기반 데이터가 필요해 원 데이터를 쓸 수 없음(합성 데이터에 박스 라벨 없음). 한때 영역 단위로 변경했다가 이미지 단위로 복귀 |
| 검증 | naive readout 재학습을 공개 체크포인트와 조합 벤치마크에서 비교 | 같은 레시피의 naive 재학습이 공개 체크포인트와 비슷하면 파이프라인이 맞다고 봄 |

### 3c.2 재학습 레시피 (`experiment/scripts/train_dcsm_scorer.py`, 저장소 `train.py` 포팅)

- 데이터: Objaverse 공간 관계 합성(`A_rel_B.png` + `CUTMIX_` hard negative), Objaverse 색상·small/big 합성(`c1_A_c2_B.png` + `MIRROR_`), COCO train 부정 쌍(json). 각 데이터셋을 settype 0/1(캡션 순서 교체)로 짝지어 쓰고, 매 epoch 가장 작은 데이터셋 크기로 무작위 절단.
- 배치: 데이터셋별·버전별로 이미지 8 + 반대 이미지 8 × 캡션 8 + 반대 캡션 8 → 16×16 점수 행렬, 행·열 셔플 후 항등 라벨에 양방향 cross-entropy. 3개 데이터셋 × 2 버전의 기울기를 누적하고 3으로 나눠 한 번 step.
- 행렬: 197×30 코사인 맵, special word 열을 학습 가능한 197차원 임베딩으로 **교체**(train.py 규칙), 배치 전체 표준화. 증강: RandomPerspective(0.05, p=0.2), RandomRotation(±4°), salt&pepper 2% (train.py의 RandomResizedCrop은 만들어지지만 적용되지 않음을 코드에서 확인).
- 최적화: Adam lr 1e-3, 27 epoch, seed 15, dropout 0, CLIP 고정.
- 차이점: CLIP forward 한 번으로 naive·MaskCLIP·ClearCLIP 세 readout을 모두 뽑아 scorer 3개를 **같은 배치·같은 증강**으로 동시에 학습한다. 체크포인트에 `readout`과 `special_slicing="replace"`를 기록해 평가 시 학습과 같은 열 규칙을 쓴다(공개 체크포인트는 저장소 eval 규칙인 "삽입 후 마지막 열 삭제"를 유지).
- 평가: 3b 조합 벤치마크에 `dcsm_{naive,maskclip,clearclip}_retrain_ctx` scorer로 추가. 검증 기준: naive 재학습 ≈ 공개 체크포인트(spatial 균형 57~58).

### 3c.3 현재 상태 (2026-10-10)

- 학습 스크립트 작성 완료. 원 데이터 파일 규약을 흉내 낸 더미 데이터로 코드 경로(데이터 로딩·special word 교체·손실·체크포인트 저장·평가 로딩)를 검증했다.
- Drive 학습 데이터는 익명 다운로드 할당량 제한에 걸려 사용자가 브라우저로 두 zip(17.5GB + 5.0GB)을 받아 넣었다(2026-10-10). 압축 해제 결과와 코드 기대 폴더의 대응:
  - 공간 관계: `objaverse_train_cutmix_newrelations_withbg/` (CUTMIX_ hard negative 포함, 24,648장). 저장소 코드가 기대하는 이름(`objaverse_train_composite_newrelations_withbg`)과 달라 `--spatial_dir`로 지정. 파일명의 관계 토큰은 above·leftof·on·behind 각 3,081장인데, 저장소 필터는 above/below/leftof/rightof만 통과시키므로 실제 학습에는 above·leftof 6,162장만 쓰이고 settype 1이 below·rightof 캡션을 만든다.
  - 색상·크기: `objaverse_attributes_overlay_withbg/` 5,402장 + `objaverse_smallbig_withbg/` 6,162장 (MIRROR_ 포함).
  - 부정: `negated_pairs_cocotrain_v1.json` 10,000쌍. **이미지는 COCO train2017 원본 14,056장을 가리키며 Drive 데이터에 없다.** COCO 서버에서 해당 파일만 병렬로 받아 `train2017/train2017/`에 두었다.
  - 사용하지 않는 폴더: `coco_cutmix_train`, `coco_ATTRSWAP`, `objaverse_composite_newrelations_withbg`(CUTMIX 없음), `objaverse_singleimg_crop`, `negated_pairs_objaverse_v1.json`.
- 실행: `uv run scripts/train_dcsm_scorer.py --data_root data/dcsm_train --spatial_dir objaverse_train_cutmix_newrelations_withbg --out results/retrain` → `compositional_eval.py --dcsm_ckpts data/dcsm_ckpt/*.pt results/retrain/*.pt --out results/comp_retrain` → `balanced_view.py results/comp_retrain`.

### 3c.4 결과 (2026-10-10, `experiment/results/retrain`, `experiment/results/comp_retrain`)

**학습**: 27 epoch × 722 step, epoch당 약 170초(총 약 77분). 16-way 정합 학습 정확도(우연 0.06): naive 0.685, MaskCLIP 0.706, ClearCLIP 0.684. 세 scorer는 같은 배치·같은 증강으로 학습되었고 readout만 다르다.

**조합 벤치마크, 크기 균형 top-1** (괄호: 타깃이 가장 큰 후보 / 아닌 후보; size는 big / small):

| scorer | size | color | spatial | negation | affirm |
|---|---|---|---|---|---|
| chance (balanced) | 37.6 | 43.5 | 45.7 | 48.1 | 33.0 |
| OWLv2 | 41.7 (72/11) | **73.5** (78/69) | 43.2 (49/37) | 39.5 (45/34) | 45.3 |
| CLIP 문맥 crop | 36.8 | 69.2 (71/67) | 39.0 (37/41) | 34.6 (36/33) | 46.2 |
| DCSM 공개 ckpt (obja / coco) | 38.2 / 38.8 | 65.7 / 61.7 | 58.1 / 57.3 | 38.7 / 37.5 | 47.8 / 46.0 |
| DCSM 재학습 · naive readout | 35.8 (38/34) | 64.2 (61/67) | 56.2 (58/54) | 41.6 (46/37) | 51.6 |
| DCSM 재학습 · **MaskCLIP readout** | 38.6 (35/42) | 59.1 (57/62) | **71.7** (75/69) | 37.5 (42/33) | 48.0 |
| DCSM 재학습 · **ClearCLIP readout** | 40.7 (50/31) | 68.4 (66/71) | **66.6** (70/64) | 36.2 (34/39) | 50.7 |

**공간 관계 세부** (raw top-1, n = 62 / 52 / 227 / 186):

| scorer | above | below | left of | right of |
|---|---|---|---|---|
| OWLv2 | 46.8 | 50.0 | 37.9 | 44.6 |
| DCSM 공개 obja | 40.3 | 51.9 | 61.2 | 61.3 |
| 재학습 naive | 41.9 | 48.1 | 59.9 | 58.1 |
| 재학습 MaskCLIP | **54.8** | **67.3** | **73.6** | **75.3** |
| 재학습 ClearCLIP | 46.8 | 65.4 | 68.7 | 69.9 |

**paired bootstrap 95% CI** (10,000 resample, raw top-1, spatial n=527):

| 비교 | 차이 | 95% CI |
|---|---|---|
| MaskCLIP 재학습 − naive 재학습 (spatial) | +15.4 | [+10.6, +20.3] |
| ClearCLIP 재학습 − naive 재학습 (spatial) | +10.2 | [+5.1, +15.4] |
| naive 재학습 − 공개 obja ckpt (spatial) | −1.9 | [−5.9, +2.1] |
| MaskCLIP 재학습 − OWLv2 (spatial) | +28.8 | [+23.3, +34.5] |
| MaskCLIP 재학습 − naive 재학습 (color) | −4.9 | [−14.1, +4.2] |
| MaskCLIP 재학습 − naive 재학습 (negation) | −4.1 | [−11.1, +2.9] |

**해석**

1. **검증 통과**: 같은 레시피로 naive readout을 재학습한 scorer는 공개 체크포인트와 통계적으로 구분되지 않는다(spatial 56.2 vs 57.3~58.1, CI가 0을 포함). 포팅한 학습 파이프라인이 원 논문과 동등하다고 볼 수 있다.
2. **핵심 결과**: readout을 MaskCLIP으로 바꾸고 scorer를 다시 학습하면 공간 관계 정확도가 56 → 72로 오른다(+15.4, CI 하한 +10.6). ClearCLIP도 +10.2로 유의하다. 특히 학습 데이터에 직접 없던 above/below(settype 1로만 생성)에서 공개 ckpt 40~52 → MaskCLIP 55~67로 가장 크게 개선되어, 반전되지 않은 패치 표현이 수직 관계의 일반화를 돕는다. OWLv2와의 격차는 +28.8이다.
3. **1주차 주장이 완성된다**: (실측 1) naive readout은 anti-localized → (3b) 그 위에 학습된 공개 scorer는 관계를 읽지만 localization이 반전된 표현에 의존 → (3c) readout surgery 위에서 재학습하면 scorer가 위치가 올바른 표현을 쓰게 되어 관계 추론이 크게 좋아진다. 변경점은 마지막 블록 readout 하나뿐이고 CLIP은 동결 상태다.
4. **한계**: color와 negation은 readout 교체로 유의하게 변하지 않는다(CI가 0 포함). negation은 모든 방법이 우연 이하이고 affirm에서는 높아, "without"을 읽지 못하는 문제는 표현이 아니라 학습 신호(합성 부정 쌍) 쪽 문제로 보인다. size는 누구도 못 읽는다. MaskCLIP > ClearCLIP 순서는 학습 정확도와 일치하지만 seed 1개 결과이므로, 논문에는 seed 3개 평균을 넣어야 한다(epoch당 170초라 비용은 작다).

## 3d. 1주차 결론

- 아이디어(DCSM 패치 표현 → 제로샷 탐지)는 **readout surgery를 전제로** 재현 가능하다. 원 코드 경로 그대로는 localization이 우연 이하다.
- COCO 80클래스 제로샷 탐지에서는 동결 CLIP + DCSM 분류 신호가 OWLv2 박스 위에서 AP 20~27을 내며, OWLv2 자체 헤드(46.8)에는 못 미친다. 이 설정에서는 DCSM 행렬이 EOS 임베딩보다 나을 이유가 없다.
- DCSM의 가치는 조합 쿼리에서 나온다. 공개 scorer만으로도 공간 관계에서 OWLv2·CLIP을 앞서고(58 vs 43), MaskCLIP readout 위에서 재학습하면 72까지 오른다. 이것이 논문의 헤드라인 결과다.
- 다음 주 후보: seed 3개 반복, 영역 단위 학습(COCO train2017 기반 쿼리 생성) 또는 negation 학습 신호 개선, OWLv2 proposal + 재학습 scorer를 결합한 end-to-end 조합 쿼리 탐지(박스까지 출력) 평가.

## 4. 베이스라인 코드·환경 재현성 이슈

- **환경**: `baseline/requirements.txt`는 UTF-16 인코딩에 torch 2.5.1+cu121 고정. 이 서버 GPU는 Blackwell(sm_120)이라 cu121 빌드는 동작하지 않는다. uv로 torch 2.11.0+cu128, transformers 5.19 환경을 구성해 동작 확인(`experiment/pyproject.toml`). lavis, open_clip, objaverse는 검출 실험에 불필요.
- **Windows 경로 하드코딩**: `src/util/models_and_dataloaders.py`의 `NegBench\evaluation data\images`, `D:\...` 체크포인트 경로 때문에 평가 파이프라인은 리눅스에서 그대로 실행되지 않는다. 검출 실험에 필요한 핵심은 `src/evaluating/eval_simple.py`의 DCSM 추출 40줄뿐이다.
- **자산**: 체크포인트·학습 데이터는 저장소에 없고 Google Drive에 있다. `gdown --folder`로 두 체크포인트(obja, coco, 각 638MB, CLIP state dict 포함) 다운로드 성공.
- **scorer 구조**: `CrossModalConvNetwork`의 FC가 197×30에 하드코딩되어 있고 전역 스칼라 하나만 출력한다. 해상도 변경·패치별 출력에는 헤드 재설계가 필요하다.
- **train/eval 불일치**: `train.py`는 special word 열을 교체(`special_location + 1:`)하지만 `eval_simple.py`는 한 칸 밀어 넣는다(`special_location:-1`). "small", "big" 같은 속성 쿼리에 영향.
- **배치 의존 점수**: `standardize()`가 배치 전체 통계를 쓰므로 같은 이미지의 점수가 함께 넣은 텍스트 집합에 따라 달라진다. 80개 클래스 쿼리 시 일관성 문제.
- **기타**: `initialize_clip`은 ViT-L/14를 로드하지만 실제 학습은 B/16(dead code). 평가 main의 `is not "all"` 문자열 비교.

## 5. 문헌 수치 (비교 상한선)

| 모델 | 설정 | 수치 | 출처 |
|---|---|---|---|
| OWL-ViT B/16 | OI+VG 학습, COCO (zero-shot 아님) | 31.7 AP / 49.2 AP50 | OWL-ViT Table A4 |
| OWL-ViT L/14 | LVIS zero-shot | 34.6 AP | OWL-ViT Table 1 |
| YOLO-World-L | COCO zero-shot | 44.4 AP | YOLO-World Table 6 |
| Grounding DINO-T | COCO zero-shot | 52.5 AP | Grounding DINO abstract |
| OV-COCO novel AP50 | ViLD 27.6 · OV-DETR 29.4 · Detic 27.8 · RegionCLIP 31.4 · CORA 35.1 (RN50) / 41.7 (RN50x4) | | CORA Table 1 |
| YOLO11x (참고, closed-set) | COCO 학습 | 54.7 AP | Ultralytics docs |
| Faster R-CNN R50-FPN v1 / v2 (참고, closed-set) | COCO 학습 | 37.0 / 46.7 AP | torchvision docs |

## 6. 논문 포지셔닝 제안

- 제목 후보: *"Reading CLIP's last block correctly: DCSM readout surgery turns a frozen CLIP into a zero-shot region classifier"*.
- 주장 1: DCSM 원 코드의 readout은 반(反)localized이며, 이는 논문이 인정한 반전 현상의 정량적 증거다 (실측 1).
- 주장 2: 마지막 블록 readout을 MaskCLIP/ClearCLIP 식으로 바꾸면, 추가 학습 없이 DCSM 행렬이 OWLv2 proposal 위에서 유효한 제로샷 분류 신호가 된다. 같은 박스 위에서 AP 1.2 → 20.3~22.8 (DCSM 토큰 읽기), 2.6 → 26.0~27.3 (EOS 읽기) (실측 2, COCO val2017 전체).
- 주장 3 (조합 쿼리, 3b): 같은 클래스 후보 중 공간 관계로 인스턴스를 고르는 과제에서 원 논문의 사전학습 DCSM scorer(문맥 crop)만 우연을 넘는다(균형 57~58 vs 우연 45.7; OWLv2 43, CLIP 39). 부정은 OWL·CLIP·DCSM scorer 모두 우연 이하이고 affirm에서는 높아, 부정어가 무시되고 B의 존재에 끌린다. 색 속성은 CLIP 전역 임베딩·OWLv2가 가장 강하고(69~74), naive readout은 우연 이하다.
- 벤치마크 기여: COCO val2017 어노테이션만으로 만든 1,833개 조합 referring-detection 항목과, 크기 편향을 드러내는 휴리스틱 기준선·크기 균형 지표. 자동 라벨(특히 color)과 크기 편향은 논문에 명시한다.
- 정직한 한계: (a) 분류 신호만 frozen CLIP에서 오므로 검출 데이터로 fine-tune된 OWLv2 자체 헤드(46.8 AP)에는 약 20 AP 못 미친다. (b) 박스는 OWLv2에서 빌려 왔고, DCSM 히트맵만으로 박스를 뽑는 P1은 AP 5 이하다. (c) 클래스 이름 단일 쿼리에서는 DCSM 토큰별 읽기가 EOS 임베딩보다 약 5 AP 낮다. (d) 조합 쿼리에서 DCSM scorer의 이득은 left/right에 집중되고 above/below·negation·size로는 확장되지 않는다. (e) DCSM scorer는 crop 단위 적용이라 "지역화"는 문맥 crop 설계에 의존한다. 다음 주 후보: surgery readout 행렬 위에 지역 단위 scorer를 직접 학습해 negation·size까지 확장.

## 7. 참고 문헌

- Kang et al., Is CLIP ideal? No. Can we fix it? Yes! ICCV 2025. https://arxiv.org/abs/2503.08723
- Zhou et al., Extract Free Dense Labels from CLIP (MaskCLIP). ECCV 2022.
- Lan et al., ClearCLIP. ECCV 2024. https://arxiv.org/abs/2407.12442
- Minderer et al., OWL-ViT. ECCV 2022. https://arxiv.org/abs/2205.06230 · OWLv2, NeurIPS 2023.
- Cheng et al., YOLO-World. CVPR 2024. https://arxiv.org/abs/2401.17270
- Liu et al., Grounding DINO. https://arxiv.org/abs/2303.05499
- Wu et al., CORA. CVPR 2023. https://arxiv.org/abs/2303.13076
