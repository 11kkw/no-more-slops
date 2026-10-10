# DCSM 논문 작성·검토 파이프라인

갱신: 2026-10-10. 범위는 **기존 근거로 논문 작성 → 무료 규칙 검사 → 별도 AI 검토 → 작성자 피드백**이다. 새 실험, GPU 추론, 결과 수집 루프와 제출은 포함하지 않는다.

## 현재 원고

영문 초안 및 2단 shortpaper가 생성되어 있다. 현재 PDF는 본문 4쪽 + 참고문헌 1쪽이다. 최초 Discord 작성 요청도 처리되었으므로 이전 문서의 '아직 요청/작성하지 않음' 상태는 더 이상 유효하지 않다. 원고는 학술 논문 구조를 갖추지만 저자·소속 및 제출 메타데이터는 미정이다. arXiv 제출·승인은 수행하지 않았다.

원고: `../paper/shortpaper/shortpaper.md`, `.tex`, `.pdf`.

## 역할과 필수 검사

| 단계 | 담당 | 결과 |
|---|---|---|
| 기존 근거와 문헌에 기반한 집필 | 기존 `paper` 프로필 | 원고·표·그림·PDF |
| AI 없이 실행하는 필수 규칙 검사 | `paper-review-local` MCP | `local_checks.json` |
| 원고·원문·렌더링에 대한 별도 검토 | 신규 `paper-reviewer` 프로필 | `editorial_review.json`, `feedback.ko.md` |
| 수정 및 최종 판단 | 집필 담당 및 사람 | 변경한 원고, 미해결 사항 |

무료 규칙 검사는 ChkTeX 1.7.9(LaTeX 표기·타이포그래피), pypdf 6.10.0(PDF 텍스트/쪽 수), RapidFuzz 3.14.6(문장 문자열 유사도), 인용 키·그림 파일·저자 자리표시자·입력 해시 검사를 포함한다. **ChkTeX 실행 누락/실패 시 검토 완료를 차단**한다. 린터 경고는 자동으로 논문 오류 판정하거나 문장을 고치지 않고 검토 에이전트가 문맥을 판단한다. 명시적으로 제공한 rc 파일을 사용하며 기본 전역 rc 미발견 메시지는 실행 로그에 보존한다.

영어 원고 본문의 8단어 이상 동일 구절과 12~150단어 문장 구간의 90 이상 문자열 유사도를 검사한다. 이 수치는 문장 쌍의 문자열 점수이며 **논문 표절률이 아니다**. 현재 확보한 참고논문 원문은 4개이며, 인용문헌 8개 전체의 본문을 검사한 것은 아니다. 의미·아이디어의 유사성, 누락된 출처, 웹 및 비공개 논문 DB는 별도 검토 대상이다.

## 자동 실행과 중단 조건

`hermes-paper-reviewer.service`는 `paper/shortpaper`의 원고·PDF·참고문헌·그림/렌더링 변화를 확인한다. 30초 이상 파일 변화가 안정되면 입력 해시를 고정하고 필수 린터를 실행한 뒤 새 `paper-reviewer` 세션을 호출한다. 새 초안 파일을 저장하면 실행되며, 검토 결과 파일 자체의 변경으로 재실행하지 않는다.

- 검토 에이전트는 MCP의 원고 읽기·검토 보고서 저장 도구만 사용하며 검토 스킬은 시작 시 미리 읽는다. 기존 PDF 렌더링 전쪽을 첫 모델 입력에 실제 이미지로 첨부한다. MCP가 반환하는 MEDIA 경로만으로 시각 검토를 완료했다고 표시하지 않는다. 원고 수정용 파일/터미널/Discord 도구는 제공하지 않는다.
- 기존 OpenAI Codex 로그인 재사용을 사용자가 승인했다. 별도 유사도 서비스 비용은 없지만 AI 검토는 기존 계정 사용량을 소비한다.
- 같은 입력 해시는 중복 검토하지 않는다. task ID당 최초 검토 + 수정 후 최대 두 차례 재검토까지만 자동 실행한다.
- 24회 도구 호출/480초 모델 실행 예산, 540초 프로세스 제한을 둔다. 실패한 모델 호출은 자동으로 반복하지 않고 실패 상태를 남긴다.
- 린터가 실행되지 않으면 검토 완료 저장을 거부한다. 원고가 검토 중 바뀌면 이전 해시로 피드백을 확정할 수 없다.
- 최신 상태는 `../paper/shortpaper/reviews/gate.json`, 상세 피드백은 해당 `review_directory`의 파일을 읽는다. 작성자는 최신 원고의 해시와 일치하는 검토를 확인하고 필수 지적 사항을 해결하거나 미해결로 명시한다.

기존 Discord 집필 봇은 유지한다. 별도 내부 검토 에이전트의 완료 결과는 `publish_review.py`가 논문-작업 채널(1558119941939535973)에 요약과 전체 피드백 Markdown 첨부로 게시한다. 사용자 승인으로 자동 게시를 활성화했고 첫 피드백은 기존 Hermes-Agent 계정으로 실제 게시/첨부 확인했다. 별도 Discord 봇 Paper Reviewer(1558443383176044556)를 AI논문사일로 서버에 초대하고 paper-reviewer 전용 토큰을 권한 0600으로 연결했다. 새 봇 이름으로 논문-작업 채널에 연결 메시지를 게시/확인했다. 이후 완료되는 검토 결과는 이 토큰을 우선 사용해 Paper Reviewer 이름으로 게시한다. 현재 원고의 최초 전체 피드백은 기존 Hermes-Agent 게시글에 보존한다. 원고 해시·필수 린터 실행·목적지 서버/채널을 확인하고 전송 영수증, 잠금, 동일 nonce, 게시 기록 확인으로 중복을 방지한다. 결과가 불확실하면 자동 재게시하지 않는다. 검토 모델에는 Discord 도구를 제공하지 않으며 게시기는 고정 채널만 사용한다. 기존 gateway는 재시작하지 않았다.

## 무료 외부 검사와 아직 연결하지 않은 기능

Paperpal 공식 무료 웹 검사는 월 7,000단어 및 전체 유사도 점수/상위 3개 출처를 제공한다. 무료 API/MCP는 확인하지 못했으므로 자동 연결 상태로 표시하지 않는다. 선택적인 수동 보조 검사로 사용할 수 있다. 원고 업로드는 수행하지 않았다.

TeXtidote/LanguageTool은 무료 로컬 문법·표현 검사 후보이나 현재 서버에 Java가 없어 미설치다. ChkTeX가 영어 문법이나 arXiv 승인 여부를 검사한다고 주장하지 않는다. Copyleaks 공식 MCP는 유료 잔액/계정이 필요하여 비활성이다. iThenticate/Turnitin 역시 미연결이다.

공식 출처와 제공 범위: `../../skills/research/paper-review-local/references/provider-options.md`.

## arXiv 준비 항목

arXiv에는 모든 논문에 동일한 2단/4쪽 템플릿을 요구하는 규정이 없다. 학술 구조와 실제 제출 준비는 구분한다. 저자·소속, 제목/초록 메타데이터, 분야/라이선스, 그림·참고문헌을 포함한 소스 묶음, 제출 환경에서의 컴파일 및 사람의 판단이 남아 있다. PDF의 표·그림 간격은 실제 렌더링으로 검토한다. 기존 컴파일 성공만으로 arXiv 제출 준비가 완료되었다고 판정하지 않는다.

## 구현 및 검증 기록

스킬/도구 원본: `skills/research/paper-review-local/`. 설치 프로필: `paper`, `paper-reviewer`.
Python·ChkTeX 의존성은 사용자 전용 캐시에 설치했으며 공유 Hermes 가상환경 및 시스템 TeX 환경을 변경하지 않았다. MCP 초기화/6개 도구 발견, 실제 원고 검사, PDF 페이지 이미지 읽기, 경로 탈출 차단, 동일 입력 재사용, 변경 입력의 오래된 검토 거부를 시험했다. 원본 원고와 실험 결과는 수정하지 않는다.


## Automated writing feedback cycle (2026-10-10)

The user explicitly requested persistent Discord connection, mutual feedback and automatic handoff to the paper writer. `hermes-paper-review-presence.service` maintains Paper Reviewer Gateway presence with no chat-content intents. `hermes-paper-reviewer.service` freezes each manuscript version, runs mandatory ChkTeX and local source-overlap checks, then invokes a fresh independent reviewer with native PDF page images. `hermes-paper-feedback-loop.service` hands the completed structured findings to a fresh existing paper writer session using only `paper_revision_local` MCP tools.

The writer stages exact prose patches and a response to every finding; it cannot directly edit originals or run commands. Number tokens in each patch, table numeric cells, figure paths and verified citations are protected. Existing evidence remains immutable. The coordinator builds and validates a separate candidate using the existing CPU-only paper builder, reruns ChkTeX, rejects concurrent source changes, and applies only validated manuscript/compiled outputs with recoverable before-copies. It posts the writer response and revised PDF through the existing writing bot; Paper Reviewer then independently rechecks the changed manuscript and compares the prior findings/response.

The latest completed review is the baseline for this newly authorized cycle. Earlier manual review history is preserved in `reviews/watcher_state_before_feedback_loop.json`. `reviews/feedback_cycle.json` records one baseline review plus at most two writer revisions and two rechecks. This cycle is not automatically reset. Failed validation, missing evidence, exhausted rounds and unresolved author choices are recorded and announced. No experiments, model-result fabrication, checkpoint/author invention, manuscript similarity-provider uploads or arXiv submission are authorized by this cycle. Gateway/publisher presence does not imply a chat-command interface.

### First automated writer handoff verification

The first writer response applied 13 prose patches after CPU compilation, 17 document/evidence checks, mandatory ChkTeX, and preservation of 44 scientific evidence/result files. The revised PDF retains four body pages and one reference page. Linter/spacing warnings remain for independent PDF review.

Writer response and revised PDF: https://discord.com/channels/1558103834788765766/1558119941939535973/1558456850339336222

Recovery retained the initial proposal and failed candidate, provisioned the missing cached Python 3.10 document runtime, and compressed redundant prose after a six-page candidate was rejected. The original scientific snapshot excludes only the separately authorized pipeline workflow document, with baseline/current hashes recorded. No experiment, prediction, arXiv submission, or invented author/checkpoint metadata was performed.
