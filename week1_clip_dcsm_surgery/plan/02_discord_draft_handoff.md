# Discord 논문 초안 작성 전달서

작성일: 2026-10-10 (KST)
상태: 작성 준비 확인 및 요청문. 이 파일을 만들면서 봇에 메시지를 보내거나 초안 작성을 실행하지 않았다.

## 작성 담당과 자료 위치

- 작성 담당: `paper` 프로필, 논문 담당 봇 ID `1503942820795388044`.
- 요청 채널: AI논문사일로 / 논문-작업 (`1558119941939535973`).
- 저장소: `/home/happy5700x/work/paper-lab/repos/no-more-slops`.
- 확인한 기준 commit: `c4c560a8a8d4fb0ceb78d5e8363e087ff113a292`. 작성 시작 시 현재 상태를 다시 확인한다.
- 결과와 평가 코드는 이 로컬 저장소에 있다. GPU 서버에 있는 결과라고 표현하지 않는다.
- `paper-server`는 별도 GPU 서버·실험 담당이다. 기존 파일로 초안을 작성하는 데 새 서버 실험은 필요하지 않다.

## Discord에 붙여 넣을 요청문

```text
<@1503942820795388044>
작업 ID: DCSM-DRAFT-001
현재 완료된 DCSM 실험으로 논문 초안 v1을 작성해주세요.

먼저 아래 전달서를 읽고, 거기에 지정한 원본 결과와 평가 코드를 확인해주세요.
/home/happy5700x/work/paper-lab/repos/no-more-slops/week1_clip_dcsm_surgery/plan/02_discord_draft_handoff.md

skill_view로 experiment-to-paper 스킬을 불러오고, 03_paper_pipeline.md의 흐름을 따라주세요.
paper/evidence/와 paper/tables/에는 이미 원본과 대조한 준비 자료가 있습니다. 먼저 확인하고, 변경된 결과가 있을 때만 exporter로 갱신해주세요.

현재 자료로 경험적 분석·재현 연구의 영문 Markdown 초안과 한국어 요약을 작성해주세요.
초안, 주장별 근거표, 참고문헌, 결과표, 검토 메모를 저장소의 week1_clip_dcsm_surgery/paper/에 저장해주세요.
인용은 원문과 근거 위치를 확인하고, 신규성·유의성·인과 해석 중 미확인인 사항은 표시해주세요.
추가 실험이 필요한 항목은 후속 작업 목록에 남기고 현재 자료로 작성 가능한 부분을 완성해주세요.
이번 요청 범위는 로컬 초안 작성과 검토입니다. 실험 실행, 서버 작업 전달, 커밋·푸시, 외부 제출·게시는 포함하지 않습니다.
완료하면 생성 파일 경로와 확인된 주장, 남은 검증 사항을 보고해주세요.
```

## 우선 읽을 원본

경로는 저장소 루트 기준이다.

1. `week1_clip_dcsm_surgery/experiment/README.md`: 방법 및 과제 설명.
2. `week1_clip_dcsm_surgery/experiment/results/full_val2017/summary.json`, `summary.md`, `run_log.txt`: 탐지 결과와 실행 기록.
3. `week1_clip_dcsm_surgery/experiment/results/probe_500/localization_probe.json`: 위치 프로브 결과.
4. `week1_clip_dcsm_surgery/experiment/results/comp_full/summary.json`, `summary.md`, `summary_balanced.md`, `heuristics.md`, `per_item.json`, `run_log.txt`: 조합 쿼리 결과.
5. `week1_clip_dcsm_surgery/experiment/results/comp_full/benchmark/comp_queries.json`, `stats.md`, `spotcheck_*.png`: 평가 항목과 검수 그림.
6. `week1_clip_dcsm_surgery/experiment/scripts/`: 실제 평가 구현. 특히 `dcsm_detection_eval.py`, `compositional_eval.py`, `build_comp_queries.py`, `balanced_view.py`.
7. `week1_clip_dcsm_surgery/experiment/overview.md`(목표·벤치마크·최종 결과)와 `experiment/findings.md`(중간 실측·핵심 발견): 기존 해석과 논문 구상. 수치와 주장은 위 원본 및 코드와 대조한다.
8. `week1_clip_dcsm_surgery/plan/03_paper_pipeline.md`와 `skills/research/experiment-to-paper/`: 봇에 설치한 작성 파이프라인과 결과 내보내기 도구.

## 원고 방향과 주요 수치

잠정 방향: 동결 CLIP의 dense readout 선택과 사전학습 DCSM scorer의 지역 단위 전이를 평가한 경험적 분석.
MaskCLIP·ClearCLIP·DCSM은 선행 방법으로 명시한다. 이들의 적용 자체를 새 방법의 발명으로 표현하지 않는다. 독립적 학술 신규성은 별도 검토 대상으로 남긴다.

- 탐지 평가의 실제 `n_images`는 **4,952**이다. COCO val2017 전체 5,000장을 평가했다고 쓰지 않는다.
- P2, 224px, DCSM token-max AP: naive **1.2**, MaskCLIP **20.3**, ClearCLIP **22.8**.
- P2, 224px, EOS/pooled AP: naive **2.6**, MaskCLIP **26.0**, ClearCLIP **27.3**.
- P1 dense의 최고 AP는 **4.6**이다. 외부 proposal에 대한 의존을 명시한다.
- OWLv2 native AP는 **46.8**, OWL-ViT v1 native는 **28.8**이다.
- 조합 평가 항목은 **1,833**개: size 655, color 142, spatial 527, negation 315, affirm 194.
- 공간 관계의 크기 균형 정확도: DCSM Objaverse scorer **58.1%**, COCO scorer **57.3%**, 무작위 기준 **45.7%**, OWLv2 **43.2%**, CLIP context crop **39.0%**.
- 공간 관계 이득은 좌우에 더 뚜렷하다. 상하를 포함한 관계 이해 전반의 해결로 일반화하지 않는다.
- 크기 원시 정확도는 큰 박스 휴리스틱이 **87.3%**이다. OWL-ViT v1은 원시 **69.5%**, big/small 균형 **44.4%**이다.
- 부정 원시 정확도는 작은 박스 휴리스틱이 **72.7%**이다. DCSM scorer의 균형 부정 정확도는 **37.5~38.7%**, 무작위 기준은 **48.1%**이다.
- `readout/naive/dcsm_tok`의 균형 부정 정확도는 **55.7%**다. 따라서 모든 방법이 무작위 이하라고 쓰지 않는다.
- 색상 균형 정확도: OWLv2 **73.5%**, CLIP crop **69.2%**, DCSM scorer **61.7~65.7%**. DCSM이 색상에서 가장 강하다고 쓰지 않는다.

## 기존 해석에서 수정할 부분

1. **P2 대 OWLv2 native는 순수 분류 능력만의 비교가 아니다.** P2는 objectness 상위 100개 proposal에서 softmax 클래스 점수 × objectness를 쓰고 박스당 3개 클래스를 유지한다. Native는 전체 예측 박스·클래스의 sigmoid 점수 상위 300개를 고른다. P2 변형끼리는 proposal을 공유하지만 native와 최종 후보 선택·점수 규칙은 다르다.
2. **학습 상태를 구분한다.** Readout 변경에는 이번 과제를 위한 학습이 없지만 DCSM scorer는 원 논문의 사전학습 CNN이다. OWL 모델도 학습된 검출기다. 원본 학습 데이터와 평가 이미지의 중복 여부는 확인 전까지 단정하지 않는다.
3. **유의성을 단정하지 않는다.** 현재 요약표에는 신뢰구간·통계 검정이 없다. 필요한 분석을 미완료로 표시한다.
4. **균형 지표의 범위를 명시한다.** 정답이 최대 박스인지 여부(크기 과제는 big/small)로 나눠 평균한 지표다. 모든 크기 편향이나 클래스·문맥 편향을 제거했다고 표현하지 않는다. 실제 휴리스틱 수치는 `summary_balanced.md`를 사용한다.
5. **OWL의 조합 평가는 박스 매칭에 영향받는다.** IoU 0.5 이상인 검출을 후보와 매칭하며, 매칭 실패 시 점수 페널티가 있다. 낮은 점수가 언어 이해만의 실패라고 단정하지 않는다.
6. **부정 해석은 가설로 제한한다.** with/without 성능 차이는 부정 처리의 취약성을 시사하지만, 서로 다른 항목 집합이므로 부정어를 무시한다는 인과 증명은 아니다. 같은 이미지·후보에서 쿼리를 바꾼 통제 비교는 후속 검증이다.
7. **색상 검수는 전체 정확도나 상한이 아니다.** 기존 문서의 20개 spot-check 및 약 16/20 추정은 소규모 검수 기록이다. 자동 라벨 정확도가 확정적으로 80%이거나 모델 정확도의 상한이 80%라고 쓰지 않는다.
8. **이전 프로젝트의 상태를 혼동하지 않는다.** `/home/happy5700x/work/paper-lab/projects/CLIP-FAIL-001/`의 기존 문헌 감사는 다른 설계 E1–E3와 당시 자료 상태를 다룬다. 그 기록을 이번 측정값이 없다는 근거로 사용하지 않는다. 반대로 이번 측정값이 이전 신규성 보류 판단을 자동으로 해소한다고도 쓰지 않는다.

## 예상 산출물과 완료 기준

- `week1_clip_dcsm_surgery/paper/manuscript.md`: 영문 초안. Abstract, Introduction, Related Work, Methods, Evaluation, Results, Discussion/Limitations, Conclusion 및 저장소의 Acknowledgement 조건을 포함한다.
- `week1_clip_dcsm_surgery/paper/summary.ko.md`: 한국어 설명.
- `week1_clip_dcsm_surgery/paper/claim_evidence.md`: 주장별 결과 파일·설정·코드 경로, 관찰과 가설 구분.
- `week1_clip_dcsm_surgery/paper/references.bib`: 확인한 문헌. 원문 검증 위치도 별도 기록한다.
- `week1_clip_dcsm_surgery/paper/tables/`: 원본 결과에서 가져온 표.
- `week1_clip_dcsm_surgery/paper/review_notes.md`: 남은 신규성·프로토콜·통계·라벨 검증과 초안의 한계.

현재 셸 PATH에서 `pdflatex`, `latexmk`, `tectonic`은 확인되지 않았다. Markdown 초안을 먼저 완성한다. LaTeX 소스와 PDF는 별도 산출물로 관리하고 컴파일하지 않은 파일을 검증된 PDF라고 보고하지 않는다.

완료 시 모든 핵심 수치가 원본 파일과 일치하고, 어떤 모델이 무엇을 학습했는지와 후보 박스의 출처가 분명해야 한다. 미확인 내용을 표시한 초안은 작성 가능하지만, 그 상태를 제출 준비 완료로 표현하지 않는다.
