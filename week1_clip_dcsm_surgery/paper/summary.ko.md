# 논문 초안 해설

영문 초안: [Readout Surgery Turns DCSM into a Spatial Relationship Detection Head for CLIP](manuscript.md) (2026-10-11 개정)

이 논문의 주장은 **동결 CLIP의 마지막 블록 readout을 MaskCLIP/ClearCLIP 방식으로 바꾸고 그 위에서 DCSM scorer를 다시 학습하면, OWLv2 proposal을 재정렬하는 공간 관계 referring detection 헤드가 된다**는 것입니다. 용어는 `experiment/overview.md` §0의 표를 따릅니다(원 논문 Kang et al.의 세 실패 유형: 속성 결합·공간 관계·부정; "조합 쿼리"라는 표현은 쓰지 않습니다).

## 1. 무엇을 보였는가

1. **진단.** DCSM 코드가 읽는 CLIP 패치 표현은 반전되어 있습니다. 정답 클래스를 알려줘도 가장 닮은 패치가 그 객체 박스 안에 있는 비율이 12%(우연 17%)이고, OWLv2 proposal을 분류하면 AP 1~3입니다. 마지막 블록만 MaskCLIP(value만) 또는 ClearCLIP(q-q attention) 방식으로 읽으면 학습 없이 pointing 62%, AP 27이 됩니다.
2. **scorer 재학습.** 원 논문 레시피와 데이터(Objaverse 합성 + COCO-train 부정 쌍)로 DCSM scorer를 세 readout 위에서 각각 seed 3개로 다시 학습했습니다. 공간 관계 후보 선택(크기 균형 top-1)에서 naive readout 59.0 ± 3.9 → MaskCLIP 69.8 ± 2.2, ClearCLIP 69.3 ± 3.4. 학습 데이터에 직접 없던 above/below에서 개선이 가장 큽니다.
3. **탐지 헤드.** GT 후보 없이 이미지와 문장만 주는 referring detection에서, OWLv2 클래스명 proposal을 재학습 scorer로 재정렬하면 공간 관계 쿼리 Acc@0.5 49.7 ± 0.6 / AP 34.4 ± 1.0입니다. 같은 문장을 받은 OWLv2(7.8 / 6.7), OWL-ViT(12.0 / 5.5), Grounding DINO-T(17.1 / 14.0), 관계를 읽지 않는 같은 후보 기준선(37.8 / 27.5)을 모두 앞섭니다.

## 2. 주장의 범위 밖 (disclaimer)

- **단일 클래스명 탐지에서는 이점이 없습니다.** COCO 80 클래스에서 DCSM 토큰 읽기는 EOS 임베딩보다 약 5 AP 낮고, OWLv2 자체 헤드(46.8)에 20 AP 뒤집니다.
- **속성 결합과 부정에서는 이점이 없습니다.** 색은 CLIP 전역 임베딩과 Grounding DINO가 더 낫고, 크기는 아무도 읽지 않으며, 부정은 모든 방법이 우연 근방입니다.
- **Grounding DINO-B가 더 강합니다**(63.0 / 52.0). 단, 이 체크포인트는 COCO 탐지 라벨과 RefCOCO referring 라벨로 사전학습되어 제로샷 비교 대상이 아닙니다. 제로샷 비교 대상은 GDINO-T입니다.
- **후보 recall이 상한입니다.** OWLv2 proposal이 타깃을 포함하는 비율은 공간 관계 쿼리에서 89.2%입니다. 헤드는 관계 재정렬기이지 localizer가 아닙니다.
- **벤치마크는 COCO val2017 어노테이션에서 자동 생성한 템플릿 문장**이며 사람이 검증한 독립 테스트셋은 없습니다. 크기 균형 지표는 큰-박스 편향 하나만 제거합니다.

## 3. 통제 실험

- naive readout 재학습은 공개 체크포인트와 통계적으로 구분되지 않습니다(포팅 검증).
- naive scorer를 surgery 표현에 그대로 쓰면 우연 이하입니다. 표현과 scorer를 함께 맞춰야 합니다.
- 쿼리에서 관계 단어를 지우면 정확도가 71 → 43(MaskCLIP)으로 떨어지고, 관계를 뒤집으면 예측의 57~67%가 바뀝니다.
- 학습 이미지 37,944장과 평가 이미지 1,025장 사이에 COCO ID·SHA-256 중복은 없습니다.
- 저장소의 Functional Rows 삽입 위치는 BOS와 다중 토큰 단어 때문에 모든 캡션에서 어긋나 있었습니다. BPE 정렬로 고친 뒤 3 seed로 다시 학습했고, MaskCLIP > ClearCLIP이라는 순서 주장은 철회했습니다. 공개 체크포인트는 저장소 규칙 그대로 평가합니다.

## 4. 산출물

- [영문 논문 본문](manuscript.md): 초록, 서론, 관련 연구, 방법, 평가 프로토콜, 결과(표 4개), 통제 실험, 한계·disclaimer, 결론, 재현 부록.
- 실험 코드: `experiment/src/dcsm_surgery`(패키지) + `experiment/scripts`(실행 스크립트) + `experiment/tests`(pytest).
- 결과: `experiment/results/full_val2017`(단일 클래스명 탐지), `comp_retrain`·`bpe_multiseed`(후보 선택), `refdet_full`(referring detection).
- 구버전 short paper(`shortpaper/`)는 재학습 이전 분석 논문 프레임이며 상단에 대체 표시를 남겼습니다.

This work was created as part of the No-More-Slops group activities at Pseudo Lab.
