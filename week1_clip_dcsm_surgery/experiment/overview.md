# Week 1 — Readout Surgery Turns DCSM into a Spatial Relationship Detection Head for CLIP

작성 2026-10-10 · 갱신 2026-10-11 · 대상 코드 [`../baseline`](../baseline) (Kang et al., "Is CLIP ideal? No. Can we fix it? Yes!", ICCV 2025 공식 구현) · 실험 코드 [`src/dcsm_surgery`](src/dcsm_surgery) + [`scripts/`](scripts) · 결과 [`results/`](results)

이 문서는 논문 제목·목표, 용어, 벤치마크·비교 대상 선정 기준, 최종 결과만 담는다. 중간 단계 실측, 수치 해석, 재학습 레시피, 검증 감사, 베이스라인 재현성 이슈는 [findings.md](findings.md)에 있다.

## 0. 논문 제목, 주장, 용어

**제목.** *Readout Surgery Turns DCSM into a Spatial Relationship Detection Head for CLIP* (2026-10-11 확정).

**주장.** 동결 CLIP ViT-B/16의 마지막 블록 readout을 MaskCLIP/ClearCLIP 방식으로 바꾸고(readout surgery) 그 위에서 원 논문 레시피로 DCSM scorer를 재학습하면, OWLv2 클래스명 proposal을 재정렬하는 **공간 관계 referring detection 헤드**가 된다. 공간 관계 쿼리("A to the left of / above a B")에서 이 헤드는 같은 문장을 받은 OWLv2·OWL-ViT·Grounding DINO-T보다 높은 Acc@0.5와 AP를 낸다.

**Disclaimer (주장의 범위 밖).**
- 단일 클래스명 쿼리(COCO 80 클래스 제로샷 탐지)에서는 DCSM 행렬이 EOS 임베딩보다 나을 이유가 없고, 학습 없는 분류 신호로서 OWLv2 자체 헤드에 약 20 AP 뒤진다.
- 속성 결합(size, color)과 부정(negation)에서는 readout surgery도 재학습도 이점을 만들지 않는다. 속성은 CLIP 전역 임베딩과 Grounding DINO가 더 낫고, 부정은 모든 방법이 우연 근방이다.
- Grounding DINO-B(Swin-B)는 COCO 탐지 라벨과 RefCOCO referring 라벨로 사전학습된 모델이라 제로샷 비교 대상이 아니며, 공간 관계에서도 우리 헤드보다 높다(Acc@0.5 63.0 vs 49.7). 제로샷 비교 대상은 GDINO-T(O365, GoldG, Cap4M)이다.
- 평가 쿼리는 COCO val2017 어노테이션에서 자동 생성한 템플릿 문장이며, 사람이 검증한 독립 테스트셋은 없다.

**용어 (원 논문 Kang et al.의 정의를 따른다).** 이전 문서의 "조합 쿼리(compositional query)"는 조합 관계 일반을 암시하므로 폐기한다.

| 용어 | 영어 | 정의 |
|---|---|---|
| 실패 유형 | failure mode | 원 논문이 정의한 CLIP의 세 실패 유형: 속성 결합(attribute binding), 공간 관계(spatial relationship: 한 객체의 절대 위치 또는 두 객체 사이의 관계), 부정(negation) |
| 실패 유형 쿼리 벤치마크 | failure-mode query benchmark | COCO val2017 어노테이션으로 만든 쿼리 집합. 서브셋 size·color(속성 결합), spatial(공간 관계), negation(부정) + 대조군 affirm |
| 속성 쿼리 / 공간 관계 쿼리 / 부정 쿼리 | attribute / spatial-relationship / negation query | 위 서브셋을 가리키는 이름. "공간 관계 쿼리"는 두 객체 사이의 이미지 평면 관계(left of / right of / above / below)만 뜻한다 |
| 후보 선택 | candidate selection | 정답(GT) 후보 박스 2~6개가 주어진 상태에서 쿼리에 맞는 하나를 고르는 과제. 지표 = top-1 정확도(크기 균형) |
| referring detection (참조 탐지) | referring detection | 이미지와 쿼리 문장만 주고 박스를 출력하는 과제. 지표 = top-1 박스의 Acc@0.5, COCO식 AP·AP50 |
| 단일 클래스명 탐지 | single-class-name detection | COCO 80 클래스 이름을 쿼리로 넣는 제로샷 탐지. 프로토콜 P1 dense / P2 proposals |
| readout surgery | | 마지막 ViT 블록 readout 교체: naive(원 코드 경로) / MaskCLIP(value만) / ClearCLIP(q-q attention, residual·FFN 제거) |
| DCSM scorer | DCSM scoring network | 197×30 코사인 행렬 위 2-conv + 2-FC CNN (`CrossModalConvNetwork`) |
| Functional Rows | | above/below/left/right/small/big/not/no/without/but/absent 11개 단어의 코사인 열을 통째로 대체하는 학습된 197차원 벡터. 코드명 `SPECIAL_WORDS` |
| 제로샷 | zero-shot | 평가 벤치마크(COCO val2017)로 학습하지 않음. scorer는 원 논문의 Objaverse 합성·COCO-train 부정 데이터로만 학습 |

## 1. 실험 목표

### 1.1 아이디어의 논리

DCSM(Dense Cosine Similarity Map)은 CLIP의 전역 임베딩 하나를 버리고, **이미지 패치 토큰 197개 × 텍스트 토큰 30개의 코사인 유사도 행렬**을 표현으로 쓴다. 원 논문은 이 행렬 위에 작은 CNN scorer를 얹어 속성 결합·공간 관계·부정을 개선했다. 이 표현을 객체 탐지에 쓰자는 아이디어는 세 단계 논리에 기댄다.

1. **DCSM 행렬은 이미 공간 격자 위에 정의되어 있다.** 행 축이 14×14 패치 격자이므로, 텍스트 토큰 하나의 열을 읽으면 그 토큰의 히트맵이 바로 나온다.
2. **텍스트 쪽이 자유 어휘다.** CLIP 텍스트 인코더를 그대로 쓰므로 클래스 이름이든 "dog to the left of a person"이든 임의 문장을 열로 넣을 수 있다.
3. **CLIP 인코더는 동결하고 읽기 방식만 바꾼다.** 원 논문의 기여도 CLIP을 재학습하지 않고 표현을 다르게 읽는 것이었다. 같은 정신으로 "CLIP 마지막 블록을 어떻게 읽어야 DCSM 행렬이 쓸 만해지는가"를 먼저 따지고, 그 위에서 scorer만 다시 학습한다.

### 1.2 질문과 실측 설계

| 실측 | 질문 | 스크립트 | 결과 폴더 |
|---|---|---|---|
| 1. localization 프로브 | 정답 클래스를 알려주면 DCSM 히트맵이 GT 박스를 가리키는가 | `probe_localization.py` | `results/probe_500` |
| 2. 단일 클래스명 탐지 | 80 클래스 전체를 쿼리로 넣었을 때 표준 AP가 얼마인가 | `dcsm_detection_eval.py` | `results/full_val2017` |
| 3. 후보 선택 | 같은 클래스 후보 중 "big dog", "dog to the left of a person", "person without a handbag"의 정답을 고를 수 있는가 | `build_comp_queries.py`, `compositional_eval.py` | `results/comp_full`, `results/comp_retrain` |
| 3′. scorer 재학습 | readout을 바꾼 행렬 위에 DCSM scorer를 다시 학습하면 후보 선택이 좋아지는가 | `train_dcsm_scorer.py`, `run_bpe_multiseed.py` | `results/bpe_multiseed` |
| 4. referring detection | GT 후보 없이 이미지+쿼리만으로 박스를 내면 OWLv2·Grounding DINO보다 나은가 | `referring_detection_eval.py` | `results/refdet_full` |

실측 1·2는 학습 없이(training-free) 수행한다. 실측 3′·4의 scorer는 원 논문 데이터로만 학습하고 평가 벤치마크로는 학습하지 않는다.

**변인.** 마지막 ViT 블록 readout 3종 × 텍스트 readout 2종 × 해상도 2종(224, 448).

| readout | 마지막 블록에서 바꾸는 것 | 출처 |
|---|---|---|
| naive | 없음. `last_hidden_state → post_layernorm → visual_projection`, DCSM 저장소의 코드 경로 | Kang et al. 2025 |
| MaskCLIP | value 임베딩만 사용 (q-k attention·residual·FFN 제거) | Zhou et al. ECCV 2022 |
| ClearCLIP | q-q self-self attention, residual·FFN 제거 | Lan et al. ECCV 2024 |

텍스트 readout: `dcsm_tok` = 내용 토큰 열의 최대값(DCSM 행렬 읽기), `pooled` = CLIP EOS 임베딩(참고 기준).

## 2. 벤치마크와 비교 대상

### 2.1 벤치마크 A — COCO val2017, 단일 클래스명 제로샷 탐지

- **선정 이유.** 주장이 "검출용 학습 없이 CLIP 표현만으로 탐지한다"이므로 평가 데이터로 학습한 적이 없는 상태에서 표준 AP를 잴 수 있어야 한다. COCO val2017은 (a) 비교 대상 OWL-ViT·YOLO-World·Grounding DINO가 모두 COCO 제로샷 수치를 보고하고, (b) pycocotools로 AP·AP50·AP75·APs/m/l을 그대로 계산하며, (c) 5,000장이라 frozen CLIP 추론을 포함한 전체 평가가 GPU 1장에서 1시간 안에 끝난다.
- **주 지표.** COCO AP(0.5:0.95), AP50. **보조 지표**(실측 1): pointing accuracy, IoU≥0.5 recall, image-level mAP.
- **프로토콜.** P1 dense = 패치별 클래스 softmax → 임계값 → 연결성분 → 박스. P2 proposals = OWLv2 objectness 헤드의 class-agnostic 박스 100개를 빌려 박스 안 DCSM 클래스 맵을 평균 풀링해 분류. OWLv2 자체 헤드와 같은 박스 생성기를 쓰지만 proposal 선택·점수 규칙은 다르다.
- **제외.** LVIS(1,203 클래스)는 DCSM 행렬 계산이 80배 커져 범위를 넘는다. OV-COCO 48/17 split은 base 클래스 학습을 전제로 하므로 학습 없는 설정에 맞지 않는다.

### 2.2 벤치마크 B — COCO val2017 어노테이션에서 만든 실패 유형 쿼리

원 논문의 주장 "CLIP 전역 임베딩은 속성·관계·부정을 표현하지 못하고 DCSM scorer는 한다"가 **검출 단위에서도** 성립하는지 묻는다. 원 논문의 평가셋(CLEVR-bind·NCD·VG-attribution / WhatsUp·COCO-QA·VG-QA / NegBench)은 이미지–캡션 매칭 정확도 벤치마크라 타깃 박스가 없어 AP를 정의할 수 없으므로, 세 실패 유형을 COCO val2017 어노테이션에서 박스 단위로 다시 만든다(`build_comp_queries.py`, 추가 라벨 없음). 모든 항목은 (이미지, 쿼리, 같은 클래스 A의 후보 박스 2~6개, 정답 1개)이므로 클래스 이름만으로는 풀 수 없다.

| 실패 유형 | subset | 쿼리 | 정답 정의 | 항목 | 우연 |
|---|---|---|---|---|---|
| 속성 결합 | size | a picture of a big/small {A} | 면적이 다른 모든 A의 3배 이상(또는 1/3 이하) | 655 | 43.1 |
| 속성 결합 | color | a picture of a {color} {A} | 침식한 인스턴스 마스크의 HSV 주색(≥55%)이 모든 distractor에서 <12% | 142 | 44.1 |
| 공간 관계 | spatial | a picture of a {A} to the left of / right of / above / below a {B} | B는 이미지에 유일, 정확히 하나의 A만 그쪽 | 527 | 45.6 |
| 부정 | negation | a picture of a {A} without a {B} | B가 붙은 A가 하나만 없음 | 315 | 47.4 |
| (대조군) | affirm | a picture of a {A} with a {B} | B가 붙은 A가 하나뿐 | 194 | 33.4 |

두 프로토콜로 평가한다.

- **후보 선택.** GT 후보 박스가 주어진다. 주 지표는 **크기 균형** top-1 정확도(정답이 최대 박스인 항목과 아닌 항목의 정확도 평균; size는 big/small 평균). 박스 크기만 보는 휴리스틱이 size에서 87, negation에서 73을 맞추기 때문에 raw 정확도는 쓰지 않는다(findings §3).
- **referring detection.** 이미지와 쿼리 문장만 준다. 지표는 top-1 박스의 **Acc@0.5**(RefCOCO 관례; raw와 크기 균형)와 subset을 카테고리로, 쿼리 항목을 이미지로 보는 **COCO식 AP(0.5:0.95)·AP50**. AP는 항목 간 점수 보정까지 평가한다.
- **라벨 품질.** 무작위 20개 spot-check에서 spatial 20/20, size 20/20, negation 19/20, color 약 16/20. color는 참고치로만 쓴다.

### 2.3 비교 대상 선정 기준

| 기준 | 설명 |
|---|---|
| 같은 과제 정의 | 텍스트 쿼리를 받아 박스(또는 후보 점수)를 내는 open-vocabulary 방식. closed-set 검출기는 참고 상한으로만 적는다. |
| 같은 백본 계열 | OWL-ViT는 CLIP ViT 백본이므로 "같은 인코더를 어떻게 쓰느냐"의 비교가 된다. OWLv2는 그 위에 self-training을 더한 현재 가장 강한 CLIP 계열 공개 모델이다. |
| 문장 쿼리 탐지의 공개 SOTA | Grounding DINO는 문장 전체를 grounding하는 공개 검출기 중 가장 널리 쓰이며 RefCOCO류 평가가 있다. T(Swin-T: O365, GoldG, Cap4M)는 COCO를 보지 않은 제로샷 비교 대상, B(Swin-B: COCO, O365, GoldG, Cap4M, OpenImage, ODinW-35, RefCOCO)는 COCO·RefCOCO 감독을 받은 상한선이다. |
| 로컬 재현 가능 | HF transformers로 같은 데이터에 직접 돌릴 수 있어야 한다. |
| 박스 공유 가능 | P2와 referring detection 파이프라인을 위해 class-agnostic objectness와 박스를 내주는 모델이 필요하다. OWLv2가 제공한다. |

**직접 실행하는 비교 대상.** 모두 벤치마크 B로는 학습하지 않는다.

| 방법 | 입력 | 점수 | Functional Rows |
|---|---|---|---|
| owlv2 / owlvit_v1 (native) | 전체 이미지 + 쿼리 문장 | 쿼리 logit (후보 선택: IoU≥0.5 검출의 최대 logit) | 아니오 |
| gdino_tiny / gdino_base (native) | 전체 이미지 + 쿼리 문장 | 토큰 확률 최대값 | 아니오 |
| clip_crop / clip_ctx | 후보 crop / 후보 ∪ B 박스 crop | CLIP 전역 임베딩 · 쿼리 EOS | 아니오 |
| dcsm_obja / dcsm_coco | 문맥 crop | 원 논문 공개 scorer 2종, naive 행렬, 저장소 eval 규칙 | 예 (원 저자 학습) |
| dcsm_{naive,maskclip,clearclip}_retrain | 문맥 crop | 원 논문 레시피·데이터로 readout만 바꿔 재학습, BPE 정렬 수정, seed 3개 | 예 (재학습) |
| readout/{naive,maskclip,clearclip}/{pooled,dcsm_tok} | 전체 이미지 히트맵 | 후보 박스 안 패치 평균 (학습 없음; 후보 선택만) | 아니오 |
| cands/owlv2_classname | OWLv2 클래스명 proposal | p(A)만 (관계를 읽지 않는 같은 후보 집합의 기준선; referring detection만) | 아니오 |

referring detection에서 `cands/*` 행은 모두 같은 파이프라인을 쓴다: OWLv2 한 번의 forward에 [쿼리, "a photo of a {A}", "a photo of a {B}"]를 넣고, p(A) ≥ 0.1인 박스(NMS 0.5, 최대 10개)를 후보로, p(B) 최상위 박스(spatial) 또는 후보에 가장 가까운 B 박스(negation/affirm)를 참조로 삼아 후보 ∪ 참조 crop을 scorer에 넣고, 최종 점수 = p(A) × softmax(scorer)로 정렬한다.

**문헌 수치만 인용하는 상한선.**

| 모델 | 설정 | 수치 | 출처 |
|---|---|---|---|
| OWL-ViT B/16 | OI+VG 학습, COCO | 31.7 AP / 49.2 AP50 | OWL-ViT Table A4 |
| YOLO-World-L | COCO zero-shot | 44.4 AP | YOLO-World Table 6 |
| Grounding DINO-T | COCO zero-shot | 48.4 AP (O365+GoldG+Cap4M) | Grounding DINO Table 2 |
| CORA RN50x4 | OV-COCO novel AP50 | 41.7 | CORA Table 1 |
| YOLO11x / Faster R-CNN R50-FPN v2 (closed-set 참고) | COCO 학습 | 54.7 / 46.7 AP | Ultralytics docs / torchvision docs |

## 3. 최종 결과

### 3.1 벤치마크 A — 단일 클래스명 제로샷 탐지 (`results/full_val2017/summary.md`)

어노테이션이 있는 4,952장, 80 클래스 쿼리. P2 행은 모두 OWLv2 박스 100개/장을 공유한다.

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
| P1 dense 최고치 (maskclip · pooled · 448) | 4.6 | 11.1 | 3.2 | 0.2 | 2.7 | 10.6 |

- 같은 박스 위에서 naive readout(원 DCSM 코드 경로)은 AP 1~3으로 분류를 못 하고, MaskCLIP·ClearCLIP readout은 AP 20~27이다. readout surgery가 결정적이다. 실측 1에서도 naive readout의 pointing 정확도는 12%로 우연(17%) 이하, ClearCLIP은 62%다.
- **Disclaimer.** 단일 클래스 이름 쿼리에서는 DCSM 토큰별 max 읽기가 EOS보다 약 5 AP 낮고, 학습 없는 DCSM 분류와 OWLv2 자체 헤드의 격차는 약 20 AP다. 이 벤치마크에서 DCSM의 이점은 없다.

### 3.2 벤치마크 B — 후보 선택, 크기 균형 top-1 (%)

공개 scorer·베이스라인은 `results/comp_retrain/summary_balanced.md`(단일 실행, 결정적). 재학습 scorer는 BPE 정렬을 수정한 뒤 seed 15·16·17로 학습한 3회 평균 ± 표준편차(`results/bpe_multiseed/seed_*/evaluation/summary_balanced.md`).

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

### 3.3 벤치마크 B — referring detection, 이미지 + 쿼리만 입력 (`results/refdet_full/summary.md`)

1,833개 항목. `cands/*`는 OWLv2 클래스명 proposal 파이프라인(§2.3), native는 쿼리 문장을 그대로 받은 검출기. 재학습 scorer는 seed 3개 평균 ± 표준편차. 후보 집합이 타깃을 포함하는 비율(IoU≥0.5, `cands/*`의 상한)은 size 91.9 / color 94.4 / spatial 89.2 / negation 81.9 / affirm 92.3.

**공간 관계 쿼리 (spatial, n=527)**

| 방법 | Acc@0.5 | Acc@0.5 균형 | AP | AP50 |
|---|---|---|---|---|
| owlv2_native | 7.8 | 8.1 | 6.7 | 9.6 |
| owlvit_v1_native | 12.0 | 12.6 | 5.5 | 9.2 |
| gdino_tiny (제로샷) | 17.1 | 18.2 | 14.0 | 19.4 |
| gdino_base (COCO + RefCOCO 학습) | **63.0** | **64.3** | **52.0** | **66.3** |
| cands/owlv2_classname (p(A)만) | 37.8 | 39.2 | 27.5 | 37.5 |
| cands/clip_ctx | 27.9 | 28.4 | 19.7 | 27.1 |
| cands/dcsm_obja (공개 scorer) | 41.6 | 42.1 | 30.7 | 41.0 |
| cands/dcsm_coco (공개 scorer) | 43.1 | 43.9 | 32.3 | 43.8 |
| cands/재학습 · naive | 40.7 ± 1.3 | 41.3 ± 1.4 | 28.1 ± 0.6 | 38.5 ± 0.9 |
| cands/재학습 · **MaskCLIP** | **49.7 ± 0.6** | **50.6 ± 0.4** | **34.4 ± 1.0** | **47.5 ± 1.2** |
| cands/재학습 · **ClearCLIP** | 48.3 ± 1.3 | 49.1 ± 1.2 | 34.4 ± 1.2 | 47.1 ± 1.5 |

관계별 Acc@0.5 (above / below / left of / right of): MaskCLIP 재학습 37.6 / 40.4 / 52.9 / 52.5, naive 재학습 28.0 / 34.0 / 43.5 / 43.5, cands/owlv2_classname 32.3 / 32.7 / 37.4 / 41.4, gdino_base 32.3 / 48.1 / 65.2 / 74.7.

paired bootstrap(10,000회, spatial Acc@0.5, seed 15 MaskCLIP 재학습 기준): vs owlv2_native +41.4 [+36.4, +45.9], vs gdino_tiny +32.1 [+27.1, +37.0], vs cands/owlv2_classname +11.4 [+6.1, +16.7], vs cands/clip_ctx +21.3 [+16.5, +25.8], vs gdino_base −13.9 [−19.0, −8.9]. seed 16·17도 같은 부호와 겹치는 구간이다(`summary.md`).

**나머지 서브셋 (Acc@0.5 raw / AP)**

| 방법 | size | color | negation | affirm |
|---|---|---|---|---|
| owlv2_native | 58.9 / 28.1 | 69.7 / 40.4 | 13.0 / 5.9 | 4.1 / 1.9 |
| gdino_tiny | 71.6 / 55.2 | 69.0 / 51.9 | 16.2 / 6.2 | 38.7 / 20.3 |
| gdino_base | 74.8 / 57.4 | 72.5 / 55.7 | 30.2 / 18.2 | 57.7 / 38.4 |
| cands/owlv2_classname | 57.6 / 32.2 | 48.6 / 32.4 | 31.1 / 20.0 | 46.9 / 24.3 |
| cands/재학습 · MaskCLIP | 42.8 ± 3.2 / 31.4 ± 2.6 | 54.2 ± 1.1 / 38.7 ± 0.7 | 28.1 ± 1.2 / 16.6 ± 0.5 | 38.7 ± 2.2 / 23.7 ± 1.7 |

size의 높은 raw 수치는 큰 박스 선호로 설명된다(gdino_base 균형 44.8, owlv2 38.9). negation은 모든 방법이 p(A)만 쓰는 기준선(31.1) 이하다.

### 3.4 결론

- **아이디어는 readout surgery를 전제로 성립한다.** 원 코드 경로(naive)의 패치 표현은 객체 영역이 반전되어 localization이 우연 이하이고(실측 1), 같은 박스 위 분류 AP도 1~3이다. 마지막 블록을 MaskCLIP·ClearCLIP 식으로 읽으면 학습 없이 AP 27까지 오른다. CLIP은 동결 상태이며 변경점은 readout 하나다.
- **readout surgery + scorer 재학습은 공간 관계 탐지 헤드를 만든다.** 후보 선택에서 naive 재학습 59.0 → MaskCLIP·ClearCLIP 재학습 69~70(+10점, 학습 데이터에 직접 없던 above/below에서 개선이 가장 큼). GT 후보 없이 OWLv2 클래스명 proposal 위에서 박스를 내는 referring detection에서는 Acc@0.5 49.7 / AP 34.4로, 같은 문장을 받은 OWLv2(7.8 / 6.7)·OWL-ViT(12.0 / 5.5)·Grounding DINO-T(17.1 / 14.0)와 관계를 읽지 않는 같은 후보 기준선(37.8 / 27.5)을 모두 앞선다. naive 재학습(40.7 / 28.1)은 후보 기준선과 구분되지 않으므로 이득은 readout surgery에서 온다.
- **상한과 한계.** COCO·RefCOCO로 학습된 Grounding DINO-B는 63.0 / 52.0으로 더 높다. 후보 recall 89.2가 파이프라인의 상한이고, 공개 scorer를 surgery 표현에 그대로 쓰면 우연 이하이므로 표현과 scorer를 함께 맞춰야 한다(findings §5.3). seed 3개 기준 MaskCLIP과 ClearCLIP의 차이는 없다.
- **Disclaimer.** 단일 클래스명 탐지(§3.1), 속성 결합(size, color), 부정(negation)에서는 DCSM 헤드의 이점이 없다. 부정은 모든 방법이 우연 근방이거나 이하이고 affirm에서는 높아, "without"을 읽지 못하는 문제는 표현이 아니라 학습 신호 쪽에 있다. 평가 쿼리는 자동 생성 템플릿이며 사람이 검증한 독립 테스트셋은 없다.
- **다음 주 후보.** negation 학습 신호 개선(영역 단위 쌍), Functional Rows를 "텍스트 임베딩 → 공간 마스크" projection 또는 히트맵 기하 추론으로 대체하는 변형, RefCOCO/RefCOCOg 공식 split에서의 평가, proposal 소스 교체(Grounding DINO-T 박스 위 재정렬).

## 4. 참고 문헌

- Kang et al., Is CLIP ideal? No. Can we fix it? Yes! ICCV 2025. https://arxiv.org/abs/2503.08723
- Zhou et al., Extract Free Dense Labels from CLIP (MaskCLIP). ECCV 2022.
- Lan et al., ClearCLIP. ECCV 2024. https://arxiv.org/abs/2407.12442
- Minderer et al., OWL-ViT. ECCV 2022. https://arxiv.org/abs/2205.06230 · OWLv2, NeurIPS 2023.
- Liu et al., Grounding DINO. ECCV 2024. https://arxiv.org/abs/2303.05499 · 체크포인트 학습 데이터: https://github.com/IDEA-Research/GroundingDINO
- Cheng et al., YOLO-World. CVPR 2024. https://arxiv.org/abs/2401.17270
- Wu et al., CORA. CVPR 2023. https://arxiv.org/abs/2303.13076
