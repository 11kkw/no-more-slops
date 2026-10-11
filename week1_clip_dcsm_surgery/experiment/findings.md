# Week 1 — 중간 실측과 핵심 발견

[overview.md](overview.md)의 목표·벤치마크·최종 결과를 뒷받침하는 중간 단계 실측, 수치 해석, 재학습 레시피, 검증 감사, 베이스라인 재현성 이슈를 모았다. 절 번호는 overview에서 "findings §n"으로 참조한다.

## 1. 실측 1 — DCSM 패치 표현의 localization 품질 (`results/probe_500`)

COCO val2017 앞 500장, frozen CLIP ViT-B/16, 정답 클래스를 알려준 상태(oracle class). pointing = 히트맵 argmax가 GT 박스 안에 들어간 비율, IoU≥0.5 = 히트맵 임계값 박스 적중률, crude AP50 = 패치별 softmax → 연결성분 → 박스로 pycocotools 측정.

| 마지막 블록 readout | 텍스트 readout | pointing 224/448px | IoU≥0.5 224/448 | image-level mAP | crude AP50 224/448 |
|---|---|---|---|---|---|
| naive (DCSM 코드 경로) | EOS | 12.0 / 11.8 | 5.1 / 6.5 | 0.43 | 1.8 / 2.4 |
| naive | 토큰별 max (DCSM 행렬) | 10.4 / 12.2 | 4.9 / 5.1 | 0.10 | 0.2 / 0.1 |
| MaskCLIP (value만) | EOS | 51.4 / 57.1 | 21.1 / 25.1 | 0.65 | 12.7 / 14.9 |
| ClearCLIP (q-q attn, residual·FFN 제거) | EOS | 62.4 / 67.5 | 26.3 / 29.9 | 0.66 | 12.1 / 13.5 |
| ClearCLIP | 토큰별 max | 52.5 / 56.8 | 22.8 / 27.7 | 0.59 | 11.0 / 13.8 |
| 우연 수준 | | 17.1 | | 0.04 | |
| CLIP 전역 CLS (참고) | | | | 0.55 | |

- **naive readout은 우연(17.1%)보다 낮다.** 히트맵(`results/probe_500/qual_bus_pooled.png`)을 보면 객체 영역이 오히려 낮은 유사도로 반전된다. DCSM 논문이 언급한 "객체를 설명하는 토큰은 해당 패치와 낮은 코사인 유사도를 가진다"는 현상의 정량적 확인이다. 원인은 CLIP 마지막 블록의 residual 경로와 FFN이 패치 토큰을 전역 표현 쪽으로 끌어당기기 때문이며, MaskCLIP·ClearCLIP이 같은 문제를 다룬다.
- 원시 DCSM 행렬의 image-level mAP 0.10은 우연 0.04에 가깝다. 원 논문 구조에서는 2층 CNN scorer가 반전 패턴을 복원해 주는 셈이다.
- readout만 바꾸면 DCSM 행렬이 분류와 위치를 모두 가진다. ClearCLIP readout의 DCSM 행렬은 image-level mAP 0.59로 CLIP 전역 CLS 0.55보다 높다.

## 2. 실측 2 — 제로샷 COCO 탐지의 중간 결과와 해석

### 2.1 스모크 테스트 (앞 50장, `results/smoke_50/summary.md`)

| config | AP | AP50 | APs | APl |
|---|---|---|---|---|
| owlvit_v1_native | 40.2 | 61.4 | 16.7 | 62.8 |
| owlv2_native | 57.7 | 76.8 | 37.9 | 76.8 |
| P2 + naive / dcsm_tok / 224 | 6.1 | 6.8 | 0.2 | 18.0 |
| P2 + maskclip / dcsm_tok / 224 | 32.5 | 41.7 | 13.2 | 70.0 |
| P2 + clearclip / dcsm_tok / 224 | 35.4 | 44.0 | 14.9 | 69.5 |
| P2 + clearclip / pooled / 224 | 39.8 | 50.6 | 17.3 | 70.4 |
| P1 dense 최고치 (maskclip / pooled / 224) | 11.6 | 23.3 | 0.2 | 24.4 |

스모크 테스트는 전체보다 10~13 AP 높게 나왔다(P2 clearclip pooled 39.8 → 27.3, OWLv2 57.7 → 46.8). 앞쪽 50장이 쉬운 이미지였기 때문이며 상대 순서는 같다. 논문에는 전체 수치만 쓴다.

### 2.2 전체 5,000장 해석 (`results/full_val2017/summary.md`, 전체 표는 overview §3.1)

- 같은 OWLv2 박스 위에서 naive readout은 AP 1.2~2.7로 사실상 분류를 못 한다. 실측 1의 반전 현상이 탐지 AP로 그대로 이어진다.
- DCSM 토큰별 max 읽기는 EOS보다 약 5 AP 낮다(clearclip 224: 22.8 vs 27.3). "a photo of a {}." 프롬프트의 여러 토큰 중 최대값을 취하는 방식이 EOS 하나보다 잡음이 많다. 이것이 조합 쿼리 실험(§3)의 동기다.
- 448px는 APs를 2~3점 올리지만 APl을 깎아 전체 AP는 거의 같다. interpolated positional embedding이 분포를 벗어나는 비용과 소형 객체 이득이 상쇄된다.
- ClearCLIP과 MaskCLIP의 차이는 1~2 AP로 작다. 하나를 고른다면 구현이 단순한 MaskCLIP(value만)으로 충분하다.
- OWL-ViT v1 B/32의 28.8 AP는 OWL-ViT 논문 Table A4의 B/16 31.7 AP와 같은 범위다. 평가 파이프라인이 문헌과 어긋나지 않음을 확인하는 용도.

## 3. 조합 쿼리 벤치마크 — 설계 근거, 크기 편향, 공개 scorer 결과 (`results/comp_full`)

### 3.1 설계 세부와 예상했던 문제

- **쿼리 형식.** DCSM 학습 템플릿("A picture of a {A} to the left of a {B}", "... without a {B}")에 맞춘 소문자 문장을 모든 scorer에 동일하게 입력한다. special word는 공백 분리 토큰으로 등장해야 DCSM scorer의 열 치환이 작동한다.
- **negation 의미 제약.** "bottle without a dining table"처럼 기하학적으로는 맞지만 어색한 쌍을 막기 위해 A는 host(person, chair, table, 탈것, 동물), B는 portable(가방, 식기, 운동기구, 사람 등)로 제한했다. subset×클래스당 60개, 이미지당 2개 상한.
- **color 라벨 잡음.** 하늘 배경 클래스(airplane, kite, bird)와 대형 가구를 제외하고 마스크를 3픽셀 침식했는데도 spot-check 정확도가 약 80%다.
- **DCSM scorer의 분포 이동.** 합성 Objaverse 렌더·COCO-spatial 쌍으로 학습된 scorer를 실사 crop에 적용한다. 원 논문 scorer는 이미지 단위 점수만 내므로 후보별 문맥 crop(후보 ∪ B 박스)으로 "지역화"했다.
- **표준화의 배치 의존성.** 저장소의 `standardize()`는 배치 전체 통계를 쓴다. 후보 crop들을 한 배치로 묶어 쿼리마다 표준화하므로 순위 비교는 가능하나 절대 점수 비교는 불가하다.
- **special word 위치 오프셋.** 저장소는 공백 분리 단어 인덱스를 CLIP 토큰 열 인덱스로 그대로 쓴다. 처음에는 "다단어 클래스에서만 한 칸 밀린다"고 보고 그대로 두었으나, §5의 감사에서 BOS 토큰 때문에 **모든** 문장에서 한 칸 이상 어긋남이 확인되었다.
- **full-image readout은 관계를 읽을 수 없다**는 것이 예상이었다. 토큰별 max나 EOS 히트맵을 박스 안에서 평균내면 "dog" 패치가 어느 dog든 비슷하게 밝다.

### 3.2 크기 편향 발견

모델 없이 박스 크기만 보는 휴리스틱이 subset에 따라 우연과 크게 다르다(`results/comp_full/heuristics.md`).

| 휴리스틱 (raw top-1) | size | color | spatial | negation | affirm |
|---|---|---|---|---|---|
| 우연 (mean 1/N) | 43.1 | 44.1 | 45.6 | 47.4 | 33.4 |
| 가장 큰 박스 | 87.3 | 63.4 | 44.2 | 23.5 | 60.3 |
| 가장 작은 박스 | 12.5 | 26.8 | 47.1 | 72.7 | 16.5 |
| 가장 중앙 박스 | 62.0 | 45.8 | 41.4 | 24.8 | 53.6 |

size는 big 항목이 575/655라 큰 박스 휴리스틱이 87을 맞추고, negation은 "B가 붙지 않은 A"가 배경의 작은 인스턴스인 경우가 많아 작은 박스 휴리스틱이 73을 맞춘다. 후보 대부분이 N=2라 필터링으로는 편향을 못 없애므로 **크기 균형 정확도**를 주 지표로 쓴다. 순수 크기 휴리스틱은 정확히 50이 된다. 균형 지표 없이는 OWL의 size raw 65~70이 "속성 이해"로 오독될 뻔했다.

### 3.3 공개 scorer와 베이스라인의 균형 정확도 (괄호: 정답이 최대 박스인 항목 / 아닌 항목)

| scorer | size | color | spatial | negation | affirm |
|---|---|---|---|---|---|
| 항목 수 (최대 / 아님) | 575 / 80 | 90 / 52 | 233 / 294 | 74 / 241 | 117 / 77 |
| owlv2 | 41.7 (72/11) | 73.5 (78/69) | 43.2 (49/37) | 39.5 (45/34) | 45.3 (53/38) |
| owlvit_v1 | 44.4 (78/11) | 63.9 (78/50) | 43.8 (56/31) | 41.8 (55/28) | 44.2 (60/29) |
| clip_crop | 36.8 (49/25) | 69.2 (71/67) | 44.2 (48/41) | 53.0 (54/52) | 38.3 (43/34) |
| clip_ctx | 36.8 (49/25) | 69.2 (71/67) | 39.0 (37/41) | 34.6 (36/33) | 46.2 (44/48) |
| dcsm_obja_ctx | 38.2 (45/31) | 65.7 (62/69) | 58.1 (60/56) | 38.7 (39/38) | 47.8 (46/49) |
| dcsm_coco_ctx | 38.8 (53/25) | 61.7 (60/63) | 57.3 (60/54) | 37.5 (36/39) | 46.0 (48/44) |
| readout/naive/pooled | 38.5 (43/34) | 21.3 (31/12) | 50.3 (58/42) | 48.0 (55/41) | 26.4 (31/22) |
| readout/naive/dcsm_tok | 38.5 (51/26) | 37.5 (42/33) | 50.0 (55/46) | 55.7 (59/52) | 36.3 (40/32) |
| readout/maskclip/pooled | 40.7 (49/32) | 62.1 (59/65) | 39.4 (40/38) | 51.1 (49/54) | 36.5 (39/34) |
| readout/maskclip/dcsm_tok | 44.5 (54/35) | 64.8 (62/67) | 43.4 (44/43) | 49.9 (51/49) | 36.3 (43/30) |
| readout/clearclip/pooled | 39.8 (43/36) | 57.6 (56/60) | 39.4 (37/41) | 48.8 (39/59) | 42.2 (40/44) |
| readout/clearclip/dcsm_tok | 43.4 (44/42) | 63.2 (61/65) | 42.5 (43/42) | 48.1 (42/54) | 35.5 (33/38) |

spatial 관계별 raw top-1: dcsm_obja left 61.2 / right 61.3 / above 40.3 / below 51.9; dcsm_coco 62.1 / 54.8 / 41.9 / 59.6; owlv2 37.9 / 44.6 / 46.8 / 50.0. 이미지 단위 클러스터 부트스트랩(10,000회)으로 dcsm_obja의 spatial 균형 정확도는 58.1 [53.7, 62.4], 우연 대비 +12.4 [+8.0, +16.7]이다(`../paper/evidence/cluster_bootstrap.json`).

### 3.4 해석

- **공간 관계는 원 논문의 DCSM scorer만 읽는다.** 두 체크포인트 모두 균형 57~58로 우연(45.7)보다 12점 높고, 정답이 큰 박스든 작은 박스든 일관된다. OWLv2·OWL-ViT(43), CLIP 문맥 crop(39), 모든 readout 히트맵(39~50)은 우연 이하다. DCSM 논문의 "CLIP 전역 임베딩은 공간 관계를 표현하지 못한다"는 주장이 검출 단위에서 재현되고, scorer가 그것을 고친다는 주장도 left/right(61~62)에서 재현된다. above/below(40~60)는 약하다.
- **부정은 아무도 확실히 읽지 못한다.** OWL(40~42), DCSM scorer(38), clip_ctx(35)는 우연(48.1) 이하이며 같은 모델들이 affirm에서는 우연(33)보다 12~15점 높다. "without"을 무시하고 B의 존재에 끌린다. 다만 naive/dcsm_tok(55.7)과 clip_crop(53.0)은 우연을 조금 넘으므로 "전혀 못 읽는다"고 쓰지는 않는다. negation과 affirm은 같은 장면의 단어 교체 대조군이 아니라는 점(생성기가 without을 우선 선택, 두 후보 항목 273/315 vs 44/194)도 해석을 제한한다.
- **속성 결합(색)은 전역 임베딩으로도 충분하다.** 후보 crop만 보면 되는 과제라 CLIP 전역 임베딩이 가장 강하다(OWLv2 73.5, clip_crop 69.2). naive readout(21~38)은 우연 아래이며, 실측 1·2의 반전 현상이 여기서도 나타난다.
- **크기는 아무도 읽지 않는다.** OWL은 big 72~78 / small 11로 순수 큰-박스 선호다. clearclip/dcsm_tok만 44/42로 big·small에 대칭적이다.
- **readout surgery는 조합 추론을 만들지 않는다.** MaskCLIP·ClearCLIP 히트맵은 색에서 naive 대비 +30~40점이지만 spatial·negation·affirm에서는 우연 근방이다. surgery는 "어디에 무엇이 있는가"를 복원할 뿐이며, 관계·부정에는 학습된 scorer가 필요하다. 이것이 §4 재학습의 동기다.
- **OWL 계열은 조합 쿼리에 약하다.** 색을 제외하면 모든 subset에서 우연 이하다. grounding 데이터로 학습했음에도 문장의 관계·부정 구조를 쓰지 않는다.

## 4. readout surgery 위에서 DCSM scorer 재학습 (`results/retrain`, `results/comp_retrain`)

### 4.1 동기와 결정 사항

§3에서 공개 DCSM scorer는 공간 관계를 읽지만 naive readout(반전된 패치 표현) 위에서 학습되었고, surgery readout 히트맵은 위치는 잘 잡지만 관계를 읽지 못했다. 둘을 합쳐 surgery readout 행렬 위에 scorer를 다시 학습하면 관계 추론과 localization이 동시에 개선되는지 본다.

| 결정 (2026-10-10) | 선택 | 이유 |
|---|---|---|
| 학습 데이터 | 원 논문 Drive 데이터, 레시피 그대로 | readout 변수만 격리. COCO val2017 평가 벤치마크와 겹치지 않음 |
| 학습 단위 | 이미지 단위(원 논문과 동일) | 영역 단위로 바꾸려면 COCO train2017 기반 데이터가 필요해 원 데이터를 쓸 수 없음 |
| 검증 | naive readout 재학습을 공개 체크포인트와 비교 | 같은 레시피의 naive 재학습이 공개 체크포인트와 비슷하면 파이프라인이 맞다고 봄 |

### 4.2 레시피 (`scripts/train_dcsm_scorer.py`, 저장소 `train.py` 포팅)

- **데이터.** Objaverse 공간 관계 합성(`A_rel_B.png` + `CUTMIX_` hard negative), Objaverse 색상·small/big 합성(`c1_A_c2_B.png` + `MIRROR_`), COCO train 부정 쌍(json). 각 데이터셋을 settype 0/1(캡션 순서 교체)로 짝지어 쓰고, 매 epoch 가장 작은 데이터셋 크기로 무작위 절단.
- **배치.** 데이터셋별·버전별로 이미지 8 + 반대 이미지 8 × 캡션 8 + 반대 캡션 8 → 16×16 점수 행렬, 행·열 셔플 후 항등 라벨에 양방향 cross-entropy. 3개 데이터셋 × 2 버전의 기울기를 누적하고 3으로 나눠 한 번 step.
- **행렬.** 197×30 코사인 맵, special word 열을 학습 가능한 197차원 임베딩으로 교체, 배치 전체 표준화. 증강: RandomPerspective(0.05, p=0.2), RandomRotation(±4°), salt&pepper 2%. train.py의 RandomResizedCrop은 만들어지지만 적용되지 않음을 코드에서 확인.
- **최적화.** Adam lr 1e-3, 27 epoch, dropout 0, CLIP 고정. 27 epoch × 722 step, epoch당 약 170초.
- **차이점.** CLIP forward 한 번으로 naive·MaskCLIP·ClearCLIP 세 readout을 모두 뽑아 scorer 3개를 같은 배치·같은 증강으로 동시에 학습한다. 체크포인트에 `readout`과 `special_slicing`을 기록해 평가 시 학습과 같은 열 규칙을 쓴다.

### 4.3 학습 데이터 출처 확인 — Objaverse는 원 논문의 선택이다

- 원 논문 §5.1: "We train one model on synthetic data created from Objaverse renders and another on a subset of COCO2017." 부록 B.2: 합성 세트는 Objaverse 3D 자산을 무작위 배경 위에 above/below/left/right로 지터를 주어 배치(관계), 서로 다른 색·크기 부여(속성), 겹치지 않는 다른 물체 쌍을 "but not / and no / without"으로 묶음(부정). 모든 positive에 속성 교환·관계 반전 hard negative를 짝지은다. 공개 체크포인트 `dcsm_obja_ckpt.pt` / `dcsm_coco_ckpt.pt`가 이 두 모델에 대응한다.
- 공개 코드의 유일한 데이터 조립 함수 `make_and_return_train_data`는 Objaverse 관계 + Objaverse 색·크기 + COCO-train 부정 json을 쓴다. 논문이 말한 Objaverse 부정 데이터는 Drive에 json만 있고 이미지 폴더가 없으며, COCO 모델용 조립 함수는 코드에 없다. 따라서 우리 재학습은 **공개 코드가 재현 가능한 유일한 레시피 = obja 모델 변형**이다.
- 수량 대조: 속성 5,402장은 논문과 일치. 관계는 논문 11,324장 vs 데이터 12,324장으로 1,000장 차이. 코드는 논문이 따로 세지 않은 small/big 6,162장도 추가로 쓴다. 부정 10,000쌍은 일치.
- 원 논문도 Objaverse로 평가하지 않고 실사 벤치마크로만 평가한다. 합성→실사 전이가 원 논문의 설정이며, COCO val2017 조합 벤치마크 평가도 같은 프레임이다.

### 4.4 데이터 준비 기록

- Drive 학습 데이터는 익명 다운로드 할당량 제한에 걸려 두 zip(17.5GB + 5.0GB)을 브라우저로 받았다(2026-10-10).
- 공간 관계: `objaverse_train_cutmix_newrelations_withbg/` 24,648장. 저장소 코드가 기대하는 폴더명과 달라 `--spatial_dir`로 지정. 파일명의 관계 토큰은 above·leftof·on·behind 각 3,081장인데 저장소 필터는 above/below/leftof/rightof만 통과시키므로 실제 학습에는 above·leftof 6,162장만 쓰이고 settype 1이 below·rightof 캡션을 만든다.
- 색상·크기: `objaverse_attributes_overlay_withbg/` 5,402장 + `objaverse_smallbig_withbg/` 6,162장.
- 부정: `negated_pairs_cocotrain_v1.json` 10,000쌍. 이미지는 COCO train2017 원본 14,056장을 가리키며 Drive에 없어 COCO 서버에서 받아 `data/dcsm_train/train2017/train2017/`에 두었다.
- 사용하지 않는 폴더: `coco_cutmix_train`, `coco_ATTRSWAP`, `objaverse_composite_newrelations_withbg`, `objaverse_singleimg_crop`, `negated_pairs_objaverse_v1.json`.

### 4.5 첫 재학습 결과 (seed 15, 수정 전 special-word 삽입)

16-way 정합 학습 정확도(우연 0.06): naive 0.685, MaskCLIP 0.706, ClearCLIP 0.684.

균형 top-1 (`results/comp_retrain/summary_balanced.md`):

| scorer | size | color | spatial | negation | affirm |
|---|---|---|---|---|---|
| 재학습 · naive | 35.8 | 64.2 | 56.2 | 41.6 | 51.6 |
| 재학습 · MaskCLIP | 38.6 | 59.1 | 71.7 | 37.5 | 48.0 |
| 재학습 · ClearCLIP | 40.7 | 68.4 | 66.6 | 36.2 | 50.7 |

paired bootstrap 95% CI (10,000 resample, raw top-1, spatial n=527):

| 비교 | 차이 | 95% CI |
|---|---|---|
| MaskCLIP 재학습 − naive 재학습 (spatial) | +15.4 | [+10.6, +20.3] |
| ClearCLIP 재학습 − naive 재학습 (spatial) | +10.2 | [+5.1, +15.4] |
| naive 재학습 − 공개 obja ckpt (spatial) | −1.9 | [−5.9, +2.1] |
| MaskCLIP 재학습 − OWLv2 (spatial) | +28.8 | [+23.3, +34.5] |
| MaskCLIP 재학습 − naive 재학습 (color) | −4.9 | [−14.1, +4.2] |
| MaskCLIP 재학습 − naive 재학습 (negation) | −4.1 | [−11.1, +2.9] |

- 검증 통과: naive 재학습은 공개 체크포인트와 통계적으로 구분되지 않는다(CI가 0을 포함). 포팅한 파이프라인이 원 논문과 동등하다.
- readout을 바꾸고 재학습하면 spatial이 56 → 72(MaskCLIP), 67(ClearCLIP)로 오른다. 이 수치는 §5의 삽입 위치 버그를 포함한 단일 seed 결과이며, 최종 수치는 §5.3이다.

## 5. 검증 감사와 BPE 정렬 수정 (`results/validation_audit`, `results/bpe_alignment`, `results/bpe_multiseed`)

### 5.1 감사에서 드러난 것 (`results/validation_audit/REPORT.md`)

- **special-word 삽입 위치가 BPE 토큰과 어긋난다.** 저장소 코드(그리고 그것을 그대로 포팅한 `compositional_eval.py`·`train_dcsm_scorer.py`)는 문장을 공백으로 나눈 단어 인덱스를 코사인 행렬의 열 인덱스로 쓴다. 그러나 CLIP 토크나이저는 0번 열에 BOS를 두고, 다단어 클래스는 여러 열을 차지한다. "A picture of a cat above a dog"에서 "above"는 6번 열인데 5번 열("cat")이 교체되었다. 학습 캡션 71,240건과 평가 쿼리 651건 모두 어긋났고, 반복 단어는 첫 번째만 교체되었다(`test_alignment_regression.py`, 5개 실패 사례).
- **학습·평가 데이터 중복 없음.** 학습 풀 37,944장과 평가 1,025장 사이에 COCO ID 중복 0, SHA256 동일 파일 0.
- **대조 실험(어긋난 체크포인트 기준, spatial n=527).** 관계 단어를 제거하면 정확도가 56→37(naive), 71→43(MaskCLIP), 66→42(ClearCLIP)로 떨어진다. 관계를 반전하면 예측의 57~67%가 바뀐다. 배치 표준화를 학습 통계 고정 또는 후보별 표준화로 바꿔도 ±2점 안에서 같다. 단, 관계 단어 제거는 토큰 위치도 바꾸므로 "관계를 읽는다"는 증명이 아니라 민감도 측정이다.
- **독립 인간 라벨 테스트셋은 없다.** 모든 평가는 COCO val 어노테이션에서 자동 생성한 쿼리다.

### 5.2 수정 내용 (`scripts/bpe_alignment.py`, 브랜치 병합 2026-10-11)

- 학습과 평가가 같은 함수 `replace_special_bpe`를 쓴다. 실제 CLIP 토크나이저로 위치를 계산하고(BOS 포함, 앞 단어의 BPE 확장 반영), 모든 standalone 등장을 교체하며, 한 단어가 여러 열에 걸치면 같은 학습 벡터를 반복한다. 잘린 등장은 건너뛰고 BOS/EOS/패딩은 건드리지 않는다.
- 새 체크포인트는 `special_slicing = bpe_replace_v1`을 기록한다. 공개 체크포인트와 첫 재학습 체크포인트(`replace`, `eval_shift`)는 경고와 함께 기존 동작을 유지한다. 메타데이터를 바꾼다고 올바르게 학습된 체크포인트가 되지는 않는다.
- 검증: 단위·통합 테스트 11개 통과, 평가 쿼리 1,833개의 기능어 등장 1,497건 모두 정렬 일치, 세 readout 모두 학습 스모크·평가 스모크 통과.
- 세 scorer가 같은 초기 가중치에서 출발하도록 초기화를 공유했다(`tests/test_shared_initialization.py`).

### 5.3 수정 후 seed 3개 재학습 결과 (seed 15, 16, 17; `results/bpe_multiseed`)

균형 top-1, 3-seed 평균 ± 표준편차 (seed별 원본은 `seed_*/evaluation/summary_balanced.md`):

| scorer | size | color | spatial | negation | affirm |
|---|---|---|---|---|---|
| 재학습 · naive | 38.9 ± 1.7 | 56.5 ± 2.6 | 59.0 ± 3.9 | 40.5 ± 3.3 | 47.2 ± 2.6 |
| 재학습 · MaskCLIP | 41.3 ± 1.7 | 61.2 ± 5.1 | 69.8 ± 2.2 | 45.7 ± 2.4 | 44.4 ± 4.8 |
| 재학습 · ClearCLIP | 42.1 ± 4.5 | 59.6 ± 6.8 | 69.3 ± 3.4 | 39.4 ± 2.7 | 41.6 ± 4.0 |

spatial raw top-1, seed별: naive 60.7 / 61.5 / 54.3, MaskCLIP 67.6 / 71.7 / 68.9, ClearCLIP 69.4 / 72.3 / 65.7. seed 간 t-구간(n=3): naive [49.0, 68.7], MaskCLIP [64.1, 74.7], ClearCLIP [60.9, 77.4].

**적응 대조.** 공개 scorer(naive 표현으로 학습)를 MaskCLIP·ClearCLIP 행렬에 그대로 적용하면 spatial 33.7 ± 1.3, 28.5 ± 1.0으로 우연 이하다. 즉 readout을 바꾸면 scorer를 반드시 다시 학습해야 하며, 재학습 이득(+35.7, +40.7)은 표현과 scorer를 함께 맞췄을 때만 나온다.

### 5.4 수정 전후 비교와 결론 변경

| spatial 균형 | 수정 전 (seed 15) | 수정 후 (3-seed 평균) |
|---|---|---|
| 재학습 · naive | 56.2 | 59.0 ± 3.9 |
| 재학습 · MaskCLIP | 71.7 | 69.8 ± 2.2 |
| 재학습 · ClearCLIP | 66.6 | 69.3 ± 3.4 |

- **유지되는 결론.** surgery readout 위 재학습이 naive 재학습보다 spatial에서 약 +10점 높다. above/below에서 개선이 가장 크다. color·negation·size는 유의하게 변하지 않는다.
- **바뀌는 결론.** 수정 전 "MaskCLIP > ClearCLIP" 순서는 seed 노이즈 안에 있어 더 이상 주장하지 않는다. 수정 전 bootstrap CI(+15.4 [+10.6, +20.3])는 어긋난 삽입을 포함한 단일 seed 수치이므로 논문에는 §5.3의 3-seed 수치를 쓴다.
- **남은 한계.** seed 3개는 정밀도가 낮다(naive 표준편차 4.0). 공개 scorer의 §3 수치는 저장소 eval 코드를 그대로 재현한 것이라 어긋난 삽입을 포함한다. 올바르게 정렬된 공개 scorer 수치는 원 저자의 재학습 없이는 얻을 수 없다.

## 6. 베이스라인 코드·환경 재현성 이슈

- **환경.** `baseline/requirements.txt`는 UTF-16 인코딩에 torch 2.5.1+cu121 고정. 이 서버 GPU는 Blackwell(sm_120)이라 cu121 빌드는 동작하지 않는다. uv로 torch 2.11.0+cu128, transformers 5.19 환경을 구성해 동작 확인(`pyproject.toml`). lavis, open_clip, objaverse는 검출 실험에 불필요.
- **Windows 경로 하드코딩.** `src/util/models_and_dataloaders.py`의 `NegBench\evaluation data\images`, `D:\...` 체크포인트 경로 때문에 평가 파이프라인은 리눅스에서 그대로 실행되지 않는다. 검출 실험에 필요한 핵심은 `src/evaluating/eval_simple.py`의 DCSM 추출 40줄뿐이다.
- **자산.** 체크포인트·학습 데이터는 저장소에 없고 Google Drive에 있다. 두 체크포인트(obja, coco, 각 638MB, CLIP state dict 포함)는 `gdown --folder`로 받았고, 체크포인트의 CLIP 가중치는 HF 원본과 완전히 동일함을 확인했다.
- **scorer 구조.** `CrossModalConvNetwork`의 FC가 197×30에 하드코딩되어 있고 전역 스칼라 하나만 출력한다. 해상도 변경·패치별 출력에는 헤드 재설계가 필요하다.
- **train/eval 불일치.** `train.py`는 special word 열을 교체(`special_location + 1:`)하지만 `eval_simple.py`는 한 칸 밀어 넣는다(`special_location:-1`). 여기에 §5.1의 BOS 오프셋이 겹친다.
- **배치 의존 점수.** `standardize()`가 배치 전체 통계를 쓰므로 같은 이미지의 점수가 함께 넣은 텍스트 집합에 따라 달라진다.
- **기타.** `initialize_clip`은 ViT-L/14를 로드하지만 실제 학습은 B/16(dead code). 평가 main의 `is not "all"` 문자열 비교.

## 7. 논문 포지셔닝 메모

- 제목 후보: *"Reading CLIP's last block correctly: readout surgery turns a frozen CLIP into a zero-shot region classifier, and a retrained DCSM scorer into a spatial-relation selector"*.
- 주장 1: DCSM 원 코드의 readout은 반(反)localized이며, 이는 논문이 인정한 반전 현상의 정량적 증거다(§1).
- 주장 2: 마지막 블록 readout을 MaskCLIP/ClearCLIP 식으로 바꾸면 추가 학습 없이 DCSM 행렬이 OWLv2 proposal 위에서 유효한 제로샷 분류 신호가 된다(§2, 같은 박스 통제 비교).
- 주장 3: 같은 readout 위에서 원 논문 레시피로 scorer를 재학습하면 공간 관계 선택이 naive 대비 약 +10점 오른다(§5.3, 3 seed). 공개 scorer를 새 표현에 그대로 쓰면 우연 이하이므로 표현과 scorer를 함께 맞춰야 한다.
- 현재 `../paper/` 초안은 §4~5 이전 상태(재학습 없음)를 기준으로 쓰였다. 재학습 결과를 반영하려면 초안의 "수정된 특징에 학습된 DCSM scorer를 결합한 실험은 수행하지 않았다"는 문장과 표 5를 갱신해야 한다.
