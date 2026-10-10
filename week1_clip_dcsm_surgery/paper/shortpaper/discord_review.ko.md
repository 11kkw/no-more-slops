# Hermes 내부 검토 및 수정 기록

작업 ID: `DCSM-SHORT-20261010-001`
대상: `shortpaper.md`, 생성 LaTeX와 실제 PDF, 보존 원고/근거/평가 코드.
검토 유형: **동일 Hermes의 순차적인 작성자측 내부 검토**. 별도 에이전트·독립 인간 심사·편집 결정·합격 가능성 평가가 아니다. 점수나 가상의 reviewer 합의를 만들지 않았다.

## summary

4쪽 숏페이퍼로 지지되는 관찰은 category-sensitive localization/detection과 relation-conditioned candidate selection의 순위가 일치하지 않았다는 것이다. P2의 ordinary pooled AP 2.6에서 ClearCLIP-style pooled 27.3으로의 이득과 공간 선택 50.3%에서 39.4%로의 변화는 서로 다른 과제를 보여준다. DCSM ObjA/COCO의 공간 선택 58.1%/57.3%는 실제 측정 우위지만, crop·학습·행렬 scoring이 달라 CNN의 단독 인과 효과로 읽을 수 없다.

**내부 판단:** 제한된 경험적 관찰을 설명하는 숏페이퍼 작업본으로는 적합하다. 신규성/벤치마크 구성 타당성/통제 비교/학습 중복이 해결되지 않았으므로 제출 준비 완료로 판정하지 않는다. 수치·원문·코드 기반의 수정과 PDF 점검을 마쳤다.

## strengths

1. 위치·검출·조건부 후보 선택을 한 점수로 혼동하지 않고 같은 질문에 연결한다.
2. 핵심 공간 결과에 실제 저장된 paired image-cluster bootstrap 불확실성을 사용하고, 없는 검출/pointing 구간은 만들지 않았다.
3. 정답 박스와 reference가 주어진 조건, 크기 shortcut, unmatched with/without을 가까운 본문에 명시했다.
4. trained DCSM CNN과 `dcsm_tok`의 토큰 최댓값을 분리하고 surgery+CNN의 미실험을 반복해서 명시했다.
5. 원본을 보존하고 2개 표/1개 그림에 집중했다. 수치 전사·인용키·실제 페이지·조판 검사와 supporting paths가 있다.

## major_issues

### M1. 검출 P2와 native OWLv2 비교의 공정성 — 본문에서 범위 제한으로 해결

- 위치: 방법 §2.2, 표 1 캡션, 결과 §3.1.
- 근거: `../../experiment/scripts/dcsm_detection_eval.py`, `native_dets`, P2 proposal/top-class 경로; `../../experiment/results/full_val2017/summary.json`.
- 관찰: P2는 objectness top-100 박스, class-softmax × objectness와 박스당 3개 클래스를 사용한다. native는 전체 박스–클래스 쌍에서 sigmoid class-logit top-300을 선택한다.
- 영향: 동일 source detector를 사용해도 최종 proposal/confidence rule이 같지 않다. native/P2 AP 차이를 pure classification 성능이라고 주장할 수 없다.
- 수정: “같은 proposal”은 P2 readout 내부에만 적용; native 46.8은 다른 프로토콜의 reference로 표시. 학습·backbone도 다름을 명시.

### M2. `dcsm_tok`를 학습된 DCSM으로 오인할 위험 — 해결

- 위치: 방법 §2.1/2.3, 결과 §3.2, 논의 §4.
- 근거: `../../experiment/scripts/compositional_eval.py`의 text token map reduction/learned scorer 경로, original DCSM §5.1의 CNN pipeline.
- 관찰: 토큰 최댓값은 내용 토큰의 scalar max이며 learned CNN이 아니다. learned arm은 ordinary CLIP features, released trained CNN을 사용한다.
- 영향: 수술 readout과 learned CNN을 factorial로 비교한 결과가 아니다.
- 수정: 두 경로를 분리하고 **modified readouts with learned CNN were not evaluated**를 명시. 추측한 조합 결과 없음.

### M3. 관계 추론·end-to-end 탐지를 과장할 위험 — 해결/외적 타당성은 미완료

- 위치: 초록, 방법 §2.3, 표 2 캡션, 논의 §4.
- 근거: `../../experiment/results/comp_full/benchmark/comp_queries.json`, `compositional_eval.py`의 crop/OWL matching.
- 관찰: 후보 박스와 reference annotation이 제공되고, DCSM/CLIP context crop은 crop geometry·이전 학습·matching 조건이 다르다.
- 수정: conditional candidate selection으로 정의; unmatched inputs/training을 설명; CNN/토폴로지 단독 인과 효과를 주장하지 않음. 사람 라벨 검증/독립 데이터는 미완료.

### M4. 구성 편향과 with/without matched-control 오해 — 해결/추가 검증은 미완료

- 위치: 결과 §3.3, 논의 §4.
- 근거: `compositional_eval.py`의 생성 분기; `../evidence/measurements.json`의 `strata_counts`, `counts`, `heuristics`, `raw_top1`, `balanced_top1`, `balanced_chance`.
- 관찰: big/small 575/80, without 우선 분기, 두 후보 without 273/315 대 with 44/194. 대상 크기 층 균형은 category/relation/candidate-count 불균형까지 제거하지 않는다.
- 수정: raw size heuristic 87.3%, OWLv2 64.7→41.7, without의 38.7/37.5 대 48.1 chance와 ordinary token-max 55.7을 함께 보여줌. 보편적 부정 추론 실패나 polarity의 causal effect라고 쓰지 않음.

### M5. 불확실성과 재현 provenance — 확인 가능한 범위로 제한

- 위치: 방법 §2.4, 표/그림 캡션, 논의 §4, README.
- 근거: `../evidence/cluster_bootstrap.json`; `evidence_export/evidence/manifest.json`; 저장 item predictions.
- 관찰: 저장된 10,000 paired image-cluster resamples와 percentile CI는 표본 변동을 다루지만 annotation 오류·checkpoint 중복을 다루지 않는다. 검출/pointing은 aggregate summaries뿐이고 현재 repository revision이 원래 실행 revision임을 증명하지 않는다.
- 수정: exploratory, no multiple-comparison correction으로 명시; unavailable intervals를 표시; snapshot revision과 original execution revision을 구분. 모델 추론은 재실행하지 않음.

## minor_issues 및 실제 조판 수정

| ID | 발견 위치/문제 | 수행한 수정 | 최종 확인 |
| --- | --- | --- | --- |
| L1 | 첫 PDF 초록이 39.4% 뒤에서 끊김: raw LaTeX environment에 unescaped `%`가 들어감 | 초록을 Pandoc으로 별도 변환하여 percent를 escape한 후 environment로 감쌈 | PDF에 58.1/57.3, 구간, 마지막 “not tested learned DCSM…” 문장까지 표시; 자동 completeness 검사 |
| L2 | ##/### 변환이 subsection/subsubsection으로 시작해 절 번호가 0.1 형태 | Pandoc heading level을 한 단계 이동 | 최종 §1–6, §2.1–2.4 / §3.1–3.3 정상 |
| L3 | 첫 빌드의 긴 모델 식별자에서 수평 넘침 경고 | 모델 ID 문장 재구성, URL path의 linebreak 허용 | 최종 overfull hbox 0; 식별자는 정확히 유지 |
| L4 | 별도 수식이 페이지 경계에서 떨어져 보임 | readout 수식을 하나의 aligned display로 묶음 | 2쪽 시작에서 수식 전체를 함께 읽을 수 있음 |
| L5 | 마지막 본문 페이지 결론이 양쪽 단에 나뉘고 단 높이가 크게 다름 | 마지막 페이지의 두 단을 `balance`로 균등 배치, references에서는 해제 | 최종 4쪽의 결론이 오른쪽 단 안에 완결. 내용/글꼴/여백은 유지 |
| L6 | 마지막 균등 배치의 내부 overfull vbox 1.29pt | 오류로 숨기지 않고 bounded noncritical warning으로 기록; 실제 페이지 좌표/렌더 확인 | 눈에 보이는 겹침·잘림·페이지 경계 초과 없음; 1.29pt 경고는 남음 |
| L7 | underfull hbox 줄 간격 경고 | 실제 렌더의 본문·식별자·문헌 가독성 확인 | 최종 15건 남음. 무경고라고 보고하지 않음 |

검사 스크립트 자체도 실행하며 수정했다: localization probe JSON은 중첩 `naive/pooled`가 아니라 literal key `results.size224["naive/pooled"]`를 사용한다. 초기 검사기의 KeyError를 해당 실제 schema에 맞게 수정했다. microtype의 의도된 hanging punctuation에 지나치게 엄격했던 glyph bounding box 기준은 3 PDF bp tolerance로 명시했다. 수평 넘침을 숨기기 위한 글꼴 축소는 없었다.

## revision_actions / 검증 근거

- 기존 예측 exporter를 `evidence_export/`로 실행했다. 문항 identity·raw·balanced·chance를 기존 데이터와 비교했으며 새 모델 추론이 아니다.
- `validate.py`의 17개 검사 그룹 통과: 표 1의 9개 수치, 표 2의 8개 estimate와 가용 CI endpoints, 숫자 본문, paired differences, 그림 데이터, BibTeX/Bbl key 일치, 본문 페이지, 초록 완전성, 두 단/페이지 영역, 원본 보존 등을 검사했다.
- 표 1 직접 근거: `../../experiment/results/probe_500/localization_probe.json`의 `results.size224["naive/pooled"|"maskclip/pooled"|"clearclip/pooled"].pointing_acc`, 검출 `measurements.detection["P1_dense/{mode}/pooled/224"|"P2_owlv2boxes/{mode}/pooled/224"].AP`.
- 표 2 직접 근거: `../evidence/cluster_bootstrap.json`의 `subsets.spatial.balanced[scorer].estimate/ci95`. chance는 expected balanced uniform selection이며 50%가 아니다.
- DCSM ObjA–chance 12.4pp [8.0,16.7]와 ObjA–ClearCLIP-pooled 18.7pp [13.1,24.2]는 `subsets.spatial.paired_differences`와 대조했다. CI overlap만으로 차이를 판단하지 않았다.
- 실제 원문 8개에 대한 지지 위치/검증 범위는 `citation_audit.md`; inherited bibliography는 수정 없이 복사. **신규성 전반을 검증했다고 말하지 않는다.**
- 최종 PDF 전체 5쪽을 렌더링해 확인; 마지막 균등 배치 이후 4쪽을 다시 확인. 1–3/5쪽은 이전 검토본과 텍스트/페이지 구성이 동일함.

## 품질 체크리스트

| 항목 | 상태 | 위치/범위 |
| --- | --- | --- |
| 초록/서론의 한정된 관찰·기여 | supported | 초록/§1; 새 detector/최초 benchmark 주장 없음 |
| 사전학습/신규 학습/동결 파라미터 구분 | supported | §2.1/2.3, §4 |
| 데이터/실제 분모/생성 조건 | supported | 500 probe, 4,952 detection, 527/375 spatial; §2.2/2.3 |
| 비교 프로토콜 차이 공개 | supported | §2.2/2.3, 표 1/2 |
| 표/그림 단위와 원본 일치 | supported | 표 1/2, 그림 1, 자동 검사 |
| 불확실성/이미지 의존성/없는 통계 | supported | §2.4; detection/pointing CI not_available |
| 인용 주장에 대한 원문 지지 | supported | citation_audit.md; source scope 안의 attribution만 |
| 한계/대안 설명 | supported | §3.3/§4; 인과 검증 자체는 not_available |
| 재현 command/version/artifact 위치 | supported | README/build_validation; original inference reproduction not_available |
| 사사/양식 상태 | supported | §6; official venue compliance not_available |

## unresolved_checks

1. 공간 정답 유일성/사람 확인, 실제 지각과 annotation-derived rule의 합치.
2. 관계별·후보 수별·category별 추가 균형 및 독립 데이터 재현.
3. context/crop/학습 조건을 맞춘 controlled comparison, surgery+learned CNN.
4. exact released checkpoint identity/hash와 COCO/ObjA 학습 데이터 overlap.
5. 원래 inference run revision/hardware/dependency inventory 및 실제 모델 추론 재현.
6. pointing/detection 이미지 단위 uncertainty. retained summaries만으로 만들지 않음.
7. 신규성 전체 검토, 투고처 공식 형식, 저자/소속, 익명 제출 전환.
8. 독립 인간 동료심사; `chktex` lint 미실행.

위 목록은 제안/미완료 검증이며 이번에 실행하거나 서버 담당에게 전달하지 않았다. 승인 범위 내에서 해결한 작성·수치·형식 문제는 모두 반영하고 bounded review를 종료했다.
