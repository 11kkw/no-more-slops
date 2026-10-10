# DCSM 논문 봇의 작성 파이프라인

설정일: 2026-10-10 (KST). 이 문서는 스킬과 작성 준비 자료를 연결한다. 아직 봇의 초안 작성 요청은 보내지 않았다.

## 설치 및 재사용

| 스킬 | 상태 | 역할 |
|---|---|---|
| `experiment-to-paper` | 이번에 `paper` 프로필에 설치 | 원본 결과 → 확인한 표 → 주장 근거표 → 초안 → 내부 검토 |
| `research-paper-writing` | 기존 설치본 재사용 | 논문의 이야기·구조·문장·학회 템플릿 |
| `grounded-citations` | 기존 설치본 재사용 | 문헌별 원문 근거·인용 ID 검증 |
| `arxiv` | 기존 설치본 재사용 | 문헌 검색·서지 확인 |
| `research-question-audit` | 기존 설치본 재사용 | 필요할 때 신규성·인접 연구 재검토 |

신규 스킬 원본은 저장소의 `skills/research/experiment-to-paper/`, 설치본은 `/home/happy5700x/.hermes/profiles/paper/skills/research/experiment-to-paper/`이다. 일반 개발 프로필과 서버 담당 프로필에는 설치하지 않았다.

## 가져온 파이프라인의 근거

- [Orchestra ML paper writing](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/20-ml-paper-writing/ml-paper-writing/SKILL.md): 저장소와 결과를 읽고, 기여를 정리한 뒤 초안을 쓰고 수정하는 흐름. 기존 Hermes 집필 스킬이 이 계열을 이미 포함한다.
- [SakanaAI AI Scientist](https://github.com/SakanaAI/AI-Scientist/blob/1de1dbc1f4ee2c5f61e9c94348d55eb51d7fa2eb/ai_scientist/perform_review.py): 구조화된 검토와 수정의 설계 참고. 코드나 실행 엔진을 설치한 것은 아니다.
- [NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist): 주장 범위·한계·재현성·실험 설정·불확실성의 확인 항목. 학회가 정해지면 해당 연도의 제출 지침을 다시 확인한다.

외부 파일 원본을 그대로 덮어쓰지 않고, 위 흐름을 이번 프로젝트의 작성 스킬에 연결했다. 상세 출처와 제외한 구성은 스킬의 `references/pipeline-sources.md`에 기록했다.

## 실제 진행 순서

1. **자료 확인:** `02_discord_draft_handoff.md`, 원본 결과, 평가 코드 및 기존 신규성 검토의 적용 범위를 읽는다.
2. **근거 준비:** 생성된 `paper/evidence/measurements.json`, `manifest.json`, `paper/tables/`를 확인한다. 이 파일들은 원본 JSON과 후보별 예측에서 생성했으며, 원시 집계·균형 정확도·휴리스틱 기록을 대조했다.
3. **주장 정리:** `paper/experiment_log.md`, `claim_evidence.md`를 작성한다. 실측과 해석을 구분하고 비교 프로토콜 차이를 기록한다.
4. **문헌 확인:** 프로젝트 전용 `paper/evidence/citations.json`에 출처와 근거 위치를 저장한다. 번호나 서지가 맞는 것과 문헌이 주장을 지지하는 것은 별도로 확인한다.
5. **초안 작성:** 영문 `manuscript.md`, 한국어 `summary.ko.md`, 확인한 참고문헌과 필요 그림을 만든다. 미확인 신규성·통계·학습 데이터 중복 여부를 명시한다.
6. **내부 검토 및 수정:** 원본 수치, 비교 조건, 신규성, 한계, 인용과 표현을 점검하고 `review_notes.md`에 수정과 남은 검증을 남긴다. 이는 독립 동료심사나 학회 채택 판정이 아니다.

## 결과가 바뀌었을 때만 준비 자료 갱신

```bash
python3 /home/happy5700x/.hermes/profiles/paper/skills/research/experiment-to-paper/scripts/export_dcsm_evidence.py \
  --repo /home/happy5700x/work/paper-lab/repos/no-more-slops \
  --out /home/happy5700x/work/paper-lab/repos/no-more-slops/week1_clip_dcsm_surgery/paper \
  --refresh
```

갱신 대상은 exporter의 `evidence/manifest.json`, `evidence/measurements.json`과 네 개의 생성 표뿐이다. 사람이 작성한 초안·근거표·문헌 ledger·검토 메모는 변경하지 않는다. 원본 결과도 수정하지 않는다. 집계나 후보 대응이 불일치하면 산출물을 쓰기 전에 실패한다.

## 검증 범위

설치·스킬 조회, 실제 저장 결과의 집계 및 표 대조를 확인했다. 준비 자료에는 탐지 4,952장과 조합 1,833개 항목, 12개 scorer가 포함된다. 이는 저장된 결과의 산술 일치 확인이며 추론 재실행, 자동 라벨의 정확도 검증, 통계적 유의성 또는 신규성 확정은 아니다.

실제 Discord 메시지 처리와 모델의 초안 생성은 아직 실행하지 않았다. 기존 작성 요청문의 범위대로 초안 작성을 시작하면 되고, 서버 접근이나 새로운 GPU 실험이 선행 조건은 아니다.
