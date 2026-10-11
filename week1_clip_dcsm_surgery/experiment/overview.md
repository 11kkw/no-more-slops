# Week 1 — DCSM readout surgery: 제로샷 탐지와 조합 쿼리 선택

작성 2026-10-10 · 갱신 2026-10-11 · 대상 코드 [`../baseline`](../baseline) (Kang et al., "Is CLIP ideal? No. Can we fix it? Yes!", ICCV 2025 공식 구현) · 실험 코드 [`scripts/`](scripts) · 결과 [`results/`](results)

이 문서는 실험 목표, 벤치마크·비교 대상 선정 기준, 최종 결과만 담는다. 중간 단계 실측, 수치 해석, 재학습 레시피, 검증 감사, 베이스라인 재현성 이슈는 [findings.md](findings.md)에 있다.

## 1. 실험 목표

### 1.1 아이디어의 논리

DCSM(Dense Cosine Similarity Map)은 CLIP의 전역 임베딩 하나를 버리고, **이미지 패치 토큰 197개 × 텍스트 토큰 30개의 코사인 유사도 행렬**을 표현으로 쓴다. 원 논문은 이 행렬 위에 작은 CNN scorer를 얹어 속성 결합·공간 관계·부정 같은 조합 추론을 개선했다. 이 표현을 객체 탐지에 쓰자는 아이디어는 세 단계 논리에 기댄다.

1. **DCSM 행렬은 이미 공간 격자 위에 정의되어 있다.** 행 축이 14×14 패치 격자이므로, 텍스트 토큰 하나의 열을 읽으면 그 토큰의 히트맵이 바로 나온다. 객체 탐지는 이 지도에서 영역을 잘라 레이블을 붙이는 일이다.
2. **텍스트 쪽이 자유 어휘다.** CLIP 텍스트 인코더를 그대로 쓰므로 클래스 이름이든 "red car"든 임의 문장을 열로 넣을 수 있다. 제로샷·open-vocabulary 탐지로 자연스럽게 이어진다.
3. **CLIP 인코더는 동결하고 읽기 방식만 바꾼다.** 원 논문의 기여도 CLIP을 재학습하지 않고 표현을 다르게 읽는 것이었다. 같은 정신으로 "CLIP 마지막 블록을 어떻게 읽어야 DCSM 행렬이 쓸 만해지는가"를 먼저 따진다.

### 1.2 질문과 실측 설계

| 실측 | 질문 | 스크립트 | 결과 폴더 |
|---|---|---|---|
| 1. localization 프로브 | 정답 클래스를 알려주면 DCSM 히트맵이 GT 박스를 가리키는가 | `probe_localization.py` | `results/probe_500` |
| 2. 제로샷 COCO 탐지 | 80 클래스 전체를 쿼리로 넣었을 때 표준 AP가 얼마인가 | `dcsm_detection_eval.py` | `results/full_val2017` |
| 3. 조합 쿼리 선택 | 같은 클래스 후보 여러 개 중 "big dog", "dog to the left of a person", "person without a handbag"의 정답을 고를 수 있는가 | `build_comp_queries.py`, `compositional_eval.py` | `results/comp_full`, `results/comp_retrain` |
| 3′. scorer 재학습 | 마지막 블록 readout을 바꾼 행렬 위에 DCSM scorer를 다시 학습하면 조합 추론이 좋아지는가 | `train_dcsm_scorer.py`, `run_bpe_multiseed.py` | `results/retrain`, `results/bpe_multiseed` |

실측 1·2는 학습 없이(training-free) 수행한다. 실측 3의 scorer는 원 논문 데이터로만 학습하고 평가 벤치마크로는 학습하지 않는다.

**변인.** 마지막 ViT 블록 readout 3종 × 텍스트 readout 2종 × 해상도 2종(224, 448).

| readout | 마지막 블록에서 바꾸는 것 | 출처 |
|---|---|---|
| naive | 없음. `last_hidden_state → post_layernorm → visual_projection`, DCSM 저장소의 코드 경로 | Kang et al. 2025 |
| MaskCLIP | value 임베딩만 사용 (q-k attention·residual·FFN 제거) | Zhou et al. ECCV 2022 |
| ClearCLIP | q-q self-self attention, residual·FFN 제거 | Lan et al. ECCV 2024 |

텍스트 readout: `dcsm_tok` = 내용 토큰 열의 최대값(DCSM 행렬 읽기), `pooled` = CLIP EOS 임베딩(참고 기준).

## 2. 벤치마크와 비교 대상

### 2.1 벤치마크 A — COCO val2017, 80 클래스 제로샷 탐지

- **선정 이유.** 주장이 "검출용 학습 없이 CLIP 표현만으로 탐지한다"이므로 평가 데이터로 학습한 적이 없는 상태에서 표준 AP를 잴 수 있어야 한다. COCO val2017은 (a) 비교 대상 OWL-ViT·YOLO-World·Grounding DINO가 모두 COCO 제로샷 수치를 보고하고, (b) pycocotools로 AP·AP50·AP75·APs/m/l을 그대로 계산하며, (c) 5,000장이라 frozen CLIP 추론을 포함한 전체 평가가 GPU 1장에서 1시간 안에 끝난다.
- **주 지표.** COCO AP(0.5:0.95), AP50. **보조 지표**(실측 1): pointing accuracy, IoU≥0.5 recall, image-level mAP. 탐지 AP 하나로는 분류 실패와 위치 실패를 구분할 수 없어서 둔다.
- **프로토콜.** P1 dense = 패치별 클래스 softmax → 임계값 → 연결성분 → 박스. P2 proposals = OWLv2 objectness 헤드의 class-agnostic 박스 100개를 빌려 박스 안 DCSM 클래스 맵을 평균 풀링해 분류. OWLv2 자체 헤드와 **같은 박스**를 쓰므로 둘의 차이가 분류 신호 품질을 격리한다.
- **제외.** LVIS(1,203 클래스)는 DCSM 행렬 계산이 80배 커져 이번 주 범위를 넘는다. OV-COCO 48/17 split은 base 클래스 학습을 전제로 하므로 학습 없는 설정에 맞지 않는다.

### 2.2 벤치마크 B — COCO val2017 어노테이션에서 만든 조합 쿼리 선택

원 논문의 주장 "CLIP 전역 임베딩은 속성·관계·부정을 표현하지 못하고 DCSM scorer는 한다"가 **검출 단위에서도** 성립하는지 묻는다. 추가 라벨 없이 COCO val2017 어노테이션만으로 만든다(`build_comp_queries.py`). 모든 항목은 (이미지, 쿼리, 같은 클래스 A의 후보 박스 2~6개, 정답 1개)이므로 클래스 이름만으로는 풀 수 없다.

| subset | 쿼리 | 정답 정의 | 항목 | 우연 |
|---|---|---|---|---|
| size | a picture of a big/small {A} | 면적이 다른 모든 A의 3배 이상(또는 1/3 이하) | 655 | 43.1 |
| color | a picture of a {color} {A} | 침식한 인스턴스 마스크의 HSV 주색(≥55%)이 모든 distractor에서 <12% | 142 | 44.1 |
| spatial | a picture of a {A} to the left of / right of / above / below a {B} | B는 이미지에 유일, 정확히 하나의 A만 그쪽 | 527 | 45.6 |
| negation | a picture of a {A} without a {B} | B가 붙은 A가 하나만 없음 | 315 | 47.4 |
| affirm | a picture of a {A} with a {B} | 대조군: B가 붙은 A가 하나뿐 | 194 | 33.4 |

- **주 지표.** 후보 중 top-1 정확도를 **크기 균형**으로 보고한다(정답이 최대 박스인 항목과 아닌 항목의 정확도 평균; size는 big/small 평균). 박스 크기만 보는 휴리스틱이 size에서 87, negation에서 73을 맞추기 때문에 raw 정확도는 쓰지 않는다(근거는 findings §3).
- **라벨 품질.** 무작위 20개 spot-check에서 spatial 20/20, size 20/20, negation 19/20, color 약 16/20. color는 참고치로만 쓴다.

### 2.3 비교 대상 선정 기준

| 기준 | 설명 |
|---|---|
| 같은 과제 정의 | 텍스트 쿼리를 받아 박스(또는 후보 점수)를 내는 open-vocabulary 방식. closed-set 검출기는 참고 상한으로만 적는다. |
| 같은 백본 계열 | OWL-ViT는 CLIP ViT 백본이므로 "같은 인코더를 어떻게 쓰느냐"의 비교가 된다. OWLv2는 그 위에 self-training을 더한 현재 가장 강한 CLIP 계열 공개 모델이다. |
| 로컬 재현 가능 | HF transformers로 같은 5,000장에 직접 돌릴 수 있어야 한다. |
| 박스 공유 가능 | P2를 위해 class-agnostic objectness와 박스를 내주는 모델이 필요하다. OWLv2가 제공한다. |

**직접 실행하는 비교 대상 (scorer).** 모두 벤치마크 B로는 학습하지 않는다.

| scorer | 입력 | 점수 | Functional Rows 사용 |
|---|---|---|---|
| owlv2 / owlvit_v1 | 전체 이미지 + 쿼리 | 후보와 IoU≥0.5인 검출의 최대 logit | 아니오 |
| clip_crop | 후보 crop(10% 여백) | CLIP 전역 임베딩 · 쿼리 EOS | 아니오 |
| clip_ctx | 후보 ∪ B 박스 crop | 위와 같음 (spatial·negation에서 B를 보게 함) | 아니오 |
| dcsm_obja / dcsm_coco | 후보 ∪ B 박스 crop | **원 논문 공개 scorer CNN** 2종, naive 197×30 행렬, 저장소 eval 코드 그대로 | 예 (원 저자 학습) |
| dcsm_{naive,maskclip,clearclip}_retrain | 후보 ∪ B 박스 crop | 원 논문 레시피·데이터로 readout만 바꿔 재학습한 scorer, BPE 정렬 수정, seed 3개 | 예 (재학습) |
| readout/{naive,maskclip,clearclip}/{pooled,dcsm_tok} | 전체 이미지 히트맵 | 후보 박스 안 패치 평균 (P2와 같은 방식, 학습 없음) | 아니오 |

Functional Rows는 원 논문이 above/below/left/right/small/big/not/no/without/but/absent 11개 단어에 대해 학습한 197차원 열 벡터로, 해당 단어의 코사인 열을 통째로 대체한다. 실측 1·2와 `readout/*` scorer에는 들어가지 않는다.

**문헌 수치만 인용하는 상한선.**

| 모델 | 설정 | 수치 | 출처 |
|---|---|---|---|
| OWL-ViT B/16 | OI+VG 학습, COCO | 31.7 AP / 49.2 AP50 | OWL-ViT Table A4 |
| YOLO-World-L | COCO zero-shot | 44.4 AP | YOLO-World Table 6 |
| Grounding DINO-T | COCO zero-shot | 52.5 AP | Grounding DINO abstract |
| CORA RN50x4 | OV-COCO novel AP50 | 41.7 | CORA Table 1 |
| YOLO11x / Faster R-CNN R50-FPN v2 (closed-set 참고) | COCO 학습 | 54.7 / 46.7 AP | Ultralytics docs / torchvision docs |

YOLO-World·Grounding DINO는 대규모 grounding 데이터로 검출 헤드를 학습했으므로 "동결 CLIP + 학습 없는 헤드"와 직접 비교하지 않고 상한선으로만 적는다.

## 3. 최종 결과

### 3.1 벤치마크 A — COCO val2017 제로샷 탐지 (`results/full_val2017/summary.md`)

어노테이션이 있는 4,952장, 80 클래스 쿼리. P2 행은 모두 OWLv2 박스 100개/장(총 1,485,600개)을 공유한다. 전체 28개 config는 원본 표 참조.

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
| P2 · maskclip · pooled · 448 | 26.7 | 37.2 | 29.4 | 13.6 | 29.7 | 43.7 |
| P1 dense 최고치 (maskclip · pooled · 448) | 4.6 | 11.1 | 3.2 | 0.2 | 2.7 | 10.6 |

- 같은 박스 위에서 naive readout(원 DCSM 코드 경로)은 AP 1~3으로 분류를 못 하고, MaskCLIP·ClearCLIP readout은 AP 20~27이다. readout surgery가 결정적이다.
- 단일 클래스 이름 쿼리에서는 DCSM 토큰별 max 읽기가 EOS보다 약 5 AP 낮다. DCSM 행렬의 이점은 조합 쿼리에서 나와야 한다.
- 학습 없는 DCSM 분류와 OWLv2 자체 헤드의 격차는 약 20 AP다(대형 객체 46.5 vs 62.7, 소형 11.2 vs 32.9). P1 dense는 AP 5 이하로 proposal 없이는 쓸 수 없다.

### 3.2 벤치마크 B — 조합 쿼리 선택, 크기 균형 top-1 (%)

공개 scorer·베이스라인은 `results/comp_retrain/summary_balanced.md`(단일 실행, 결정적). 재학습 scorer는 BPE 정렬을 수정한 뒤 seed 15·16·17로 학습한 3회 평균 ± 표준편차(`results/bpe_multiseed/seed_*/evaluation/summary_balanced.md`). 수정 전 단일 seed 수치와 수정 내용은 findings §5.

| scorer | size | color | spatial | negation | affirm |
|---|---|---|---|---|---|
| 우연 (균형) | 37.6 | 43.5 | 45.7 | 48.1 | 33.0 |
| 크기 휴리스틱 | 50 | 50 | 50 | 50 | 50 |
| owlv2 | 41.7 | **73.5** | 43.2 | 39.5 | 45.3 |
| owlvit_v1 | 44.4 | 63.9 | 43.8 | 41.8 | 44.2 |
| clip_crop | 36.8 | 69.2 | 44.2 | 53.0 | 38.3 |
| clip_ctx | 36.8 | 69.2 | 39.0 | 34.6 | 46.2 |
| dcsm_obja (공개 scorer) | 38.2 | 65.7 | 58.1 | 38.7 | 47.8 |
| dcsm_coco (공개 scorer) | 38.8 | 61.7 | 57.3 | 37.5 | 46.0 |
| 재학습 · naive readout | 38.9 ± 1.7 | 56.5 ± 2.6 | 59.0 ± 3.9 | 40.5 ± 3.3 | 47.2 ± 2.6 |
| 재학습 · **MaskCLIP readout** | 41.3 ± 1.7 | 61.2 ± 5.1 | **69.8 ± 2.2** | 45.7 ± 2.4 | 44.4 ± 4.8 |
| 재학습 · **ClearCLIP readout** | 42.1 ± 4.5 | 59.6 ± 6.8 | **69.3 ± 3.4** | 39.4 ± 2.7 | 41.6 ± 4.0 |
| readout/maskclip/dcsm_tok (학습 없음) | 44.5 | 64.8 | 43.4 | 49.9 | 36.3 |
| readout/clearclip/pooled (학습 없음) | 39.8 | 57.6 | 39.4 | 48.8 | 42.2 |

**spatial 관계별 raw top-1** (n = 62 / 52 / 227 / 186; 재학습은 3-seed 평균 ± 표준편차):

| scorer | above | below | left of | right of |
|---|---|---|---|---|
| owlv2 | 46.8 | 50.0 | 37.9 | 44.6 |
| dcsm_obja (공개) | 40.3 | 51.9 | 61.2 | 61.3 |
| 재학습 · naive | 47.8 ± 5.2 | 50.6 ± 2.9 | 64.6 ± 4.9 | 57.7 ± 4.2 |
| 재학습 · MaskCLIP | 60.8 ± 0.9 | 56.4 ± 6.2 | 73.6 ± 0.4 | 70.8 ± 6.3 |
| 재학습 · ClearCLIP | 58.1 ± 5.8 | 67.3 ± 5.1 | 73.0 ± 5.3 | 68.6 ± 5.0 |

### 3.3 결론

- **아이디어는 readout surgery를 전제로 성립한다.** 원 코드 경로(naive)의 패치 표현은 객체 영역이 반전되어 localization이 우연 이하이고(실측 1), 같은 박스 위 분류 AP도 1~3이다. 마지막 블록을 MaskCLIP·ClearCLIP 식으로 읽으면 학습 없이 AP 27까지 오른다. CLIP은 동결 상태이며 변경점은 readout 하나다.
- **클래스 이름 탐지에서는 DCSM 행렬이 EOS 임베딩보다 나을 이유가 없고**, OWLv2 자체 헤드(46.8)에는 약 20 AP 못 미친다.
- **DCSM의 가치는 조합 쿼리, 그중 공간 관계에서 나온다.** 공개 scorer만으로도 spatial에서 OWLv2·CLIP(39~44)을 앞서고(57~58), readout surgery 위에서 재학습하면 69~70까지 오른다. naive 재학습(59.0 ± 3.9)과 비교해 MaskCLIP·ClearCLIP 모두 약 +10점이며, 학습 데이터에 직접 없던 above/below에서 개선이 가장 크다. seed 3개 기준 MaskCLIP과 ClearCLIP의 차이는 없다.
- **color·negation·size는 readout 교체로 유의하게 변하지 않는다.** negation은 모든 방법이 우연 근방이거나 이하이고 affirm에서는 높아, "without"을 읽지 못하는 문제는 표현이 아니라 학습 신호 쪽에 있다. size는 아무도 읽지 않는다.
- **정직한 한계.** 조합 평가는 정답 후보 박스가 주어진 선택 과제이며 referring detection 전체 성능이 아니다. 박스는 OWLv2에서 빌려 왔다. 공개 scorer 수치는 저장소 eval 코드의 어긋난 special-word 삽입을 그대로 재현한 것이다(findings §5).
- **다음 주 후보.** OWLv2 proposal + 재학습 scorer의 end-to-end 조합 쿼리 탐지, negation 학습 신호 개선(영역 단위 쌍), Functional Rows를 학습된 lookup 대신 "텍스트 임베딩 → 공간 마스크" projection 또는 히트맵 기하 추론으로 대체하는 변형 비교.

## 4. 참고 문헌

- Kang et al., Is CLIP ideal? No. Can we fix it? Yes! ICCV 2025. https://arxiv.org/abs/2503.08723
- Zhou et al., Extract Free Dense Labels from CLIP (MaskCLIP). ECCV 2022.
- Lan et al., ClearCLIP. ECCV 2024. https://arxiv.org/abs/2407.12442
- Minderer et al., OWL-ViT. ECCV 2022. https://arxiv.org/abs/2205.06230 · OWLv2, NeurIPS 2023.
- Cheng et al., YOLO-World. CVPR 2024. https://arxiv.org/abs/2401.17270
- Liu et al., Grounding DINO. https://arxiv.org/abs/2303.05499
- Wu et al., CORA. CVPR 2023. https://arxiv.org/abs/2303.13076
