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

- 같은 박스 위에서 naive readout은 AP 6, MaskCLIP·ClearCLIP은 AP 32~40. readout surgery 하나로 분류 신호가 살아난다.
- 대형 객체(APl)에서는 학습 없는 DCSM 분류가 OWLv2 자체 헤드(70 vs 77)에 근접한다. 소형 객체(APs)는 격자 해상도 한계로 크게 뒤진다.
- P1 dense는 AP 12 이하. 위치 신호까지 DCSM에서 뽑으려면 별도 헤드가 필요하다.

### 3.2 전체 5,000장 결과

<!-- RESULTS_FULL -->
(실행 중. 완료 후 `results/full_val2017/summary.md`에서 옮겨 적는다.)

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
- 주장 2: 마지막 블록 readout을 MaskCLIP/ClearCLIP 식으로 바꾸면, 추가 학습 없이 DCSM 행렬이 OWLv2 proposal 위에서 유효한 제로샷 분류 신호가 된다 (실측 2, 같은 박스 통제 비교).
- 정직한 한계: 분류 신호만 frozen CLIP에서 오므로 검출 데이터로 fine-tune된 OWLv2 자체 헤드에는 못 미친다. 박스는 OWLv2에서 빌려 왔다. 조합 쿼리(속성·공간·부정) 검출은 다음 주 범위.

## 7. 참고 문헌

- Kang et al., Is CLIP ideal? No. Can we fix it? Yes! ICCV 2025. https://arxiv.org/abs/2503.08723
- Zhou et al., Extract Free Dense Labels from CLIP (MaskCLIP). ECCV 2022.
- Lan et al., ClearCLIP. ECCV 2024. https://arxiv.org/abs/2407.12442
- Minderer et al., OWL-ViT. ECCV 2022. https://arxiv.org/abs/2205.06230 · OWLv2, NeurIPS 2023.
- Cheng et al., YOLO-World. CVPR 2024. https://arxiv.org/abs/2401.17270
- Liu et al., Grounding DINO. https://arxiv.org/abs/2303.05499
- Wu et al., CORA. CVPR 2023. https://arxiv.org/abs/2303.13076
