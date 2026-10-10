# Hermes Discord 숏페이퍼 작성·검토 요청

작업 ID: `DCSM-SHORT-20261010-001`
사용자 승인: 2026-10-10 KST, 기존 초안의 논문 형식을 디스코드 Hermes 봇이 확인하고 숏페이퍼로 작성하도록 요청. 투고처는 미정이며 **본문 4쪽, 2단 형식**을 선택함. 이어서 사용자가 “디스코드 헤르메스 봇한테 이야기해야하는거아닌가?”라고 명시함.

## 담당과 범위

- 담당: 논문 담당 Hermes (`paper` 프로필), 봇 `1503942820795388044`.
- 사용자 논문 채널: AI논문사일로 / 논문-작업 `1558119941939535973`, guild `1558103834788765766`.
- 실제 봇 간 전달·완료 보고 채널: 진행-현황 `1558120989332734112`. 기존 `discord-peer-gate`가 서버 담당→논문 담당 전달을 이 채널에서 허용하므로 이 경로를 사용한다. 첫 논문-작업 메시지는 게이트에서 처리되지 않았고 중복 실행이 아니다. 권한 설정은 변경하지 않았다.
- Codex가 사용자의 요청을 기존 서버 담당 봇 계정으로 전달한다. 새로운 서버 실험 요청이 아니다. 추가 봇을 멘션하거나 서버 작업을 전달하지 말 것.
- 저장소: `/home/happy5700x/work/paper-lab/repos/no-more-slops`.
- 기존 초안: `week1_clip_dcsm_surgery/paper/manuscript.md` (약 5,300 단어, 본문 표 5개, 그림 2개와 부록).
- **새 산출물은 `week1_clip_dcsm_surgery/paper/shortpaper/` 안에 작성**한다. 기존 장문 초안·원본 결과·코드·스킬·다른 미커밋 작업을 보존한다.

## 해야 할 일

1. `skill_view`로 `experiment-to-paper`, `research-paper-writing`을 읽고 기존 원고·근거표·검토 메모를 확인한다. 전체 연구 파이프라인이나 새 실험을 시작하지 않는다.
2. 현재 원고가 학술 논문 구성과 숏페이퍼 형식에 각각 맞는지 **실제 파일 기준**으로 평가해 `format_review.ko.md`에 적는다. Markdown에 논문 섹션이 있다는 것과 4쪽으로 조판된 논문이라는 것을 구분한다.
3. 주장을 하나로 집중해 영어 숏페이퍼를 작성한다. 핵심은 “평가한 CLIP readout의 위치·검출 이득이 관계 기반 후보 선택 개선으로 일관되게 전이되지는 않았다”는 경험적 관찰이다. 원래의 장문 표·실험 설명을 그대로 줄여 넣지 말고 논문 서사에 맞게 선택한다.
4. `shortpaper.md`, `shortpaper.tex`, **실제 컴파일한 `shortpaper.pdf`**를 만든다. 본문 4쪽, 2단, 정상적인 10pt 이상 본문 글꼴·일반적 여백을 사용한다. 참고문헌은 별도 페이지로 둘 수 있다. 임의 글꼴 축소, 음수 간격, 내용을 채우기 위한 공백으로 4쪽을 맞추지 않는다. 특정 학회 공식 양식에 부합한다고 주장하지 않는다. 저자·소속은 미정이며 저자를 지어내지 않는다.
5. 제목, 초록, 서론(관련 연구 통합 가능), 방법, 실험·결과, 한계·결론, 참고문헌을 갖춘다. 핵심 표와 그림만 사용하고 기존 상세 초안을 보충 자료로 연결한다.
6. 컴파일·페이지 수·인용키·표 수치·그림과 넘침을 확인하고 `build_validation.json`에 실제 수행한 검사를 적는다. PDF를 직접 읽거나 렌더링해 조판을 검토한다. 이용 가능한 방법으로 못한 검사는 미확인으로 표시한다.
7. 결과를 내부 검토하고 수정한 뒤 `discord_review.ko.md`에 검토 근거와 수정 사항을 적는다. 이는 Hermes에 의한 검토이며 독립적인 인간 동료 심사가 아니다.
8. **완료되면 이 Discord 채널에** 파일 경로, 실제 페이지 수(본문/참고문헌), 논문 형식 판정, 수정한 핵심 사항과 남은 검증을 한국어로 보고한다. PDF 업로드가 가능하면 함께 첨부한다. 다른 봇 멘션 없이 사용자가 확인할 수 있게 답한다.

## 바로 사용할 근거

- `paper/evidence/measurements.json`, `paper/evidence/manifest.json`
- `paper/evidence/cluster_bootstrap.json` (기존 예측의 CPU 분석; 새 모델 실험이 아님)
- `paper/claim_evidence.md`, `paper/experiment_log.md`, `paper/review_notes.md`
- `paper/references.bib`, `paper/evidence/citations.json` (검증한 8개 문헌; 서지 정보 추측 금지)
- `paper/figures/` (PNG/SVG/PDF)
- 원본 `experiment/results/`와 평가 코드. 필요할 때 주장 범위를 확인한다.

반드시 유지할 조건:

- 검출 평가 4,952장; AP는 0–100. P2의 기본 EOS 2.6 → ClearCLIP 방식 EOS 27.3(224 입력).
- P2 변형끼리는 조건을 공유하지만 native OWLv2 46.8과는 최종 제안 박스·점수 규칙이 다르다. 순수 분류 차이로 표현 금지.
- 공간 관계 527문항/375이미지, DCSM ObjA 보정 58.1% [53.7,62.4], COCO 57.3% [53.2,61.2], 기준 확률 45.7%.
- `dcsm_tok`는 토큰 최댓값이며 학습된 DCSM CNN과 다름. Surgery + 학습 CNN 조합은 미실험.
- 조합 과제는 정답 후보 박스가 제공되는 조건부 선택. 사람 라벨 검증과 학습 중복·신규성 검토는 미완료.
- with/without은 생성 분기·후보 수가 달라 matched control이 아님.
- 사사: `This work was created as part of the No-More-Slops group activities at Pseudo Lab.`

## 조판 도구

Codex가 공식 릴리스의 SHA256을 대조해 준비한 Tectonic 0.17.0:

`/home/happy5700x/.cache/paper-tools/tectonic-0.17.0/tectonic`

`tectonic --keep-logs --keep-intermediates shortpaper.tex`로 LaTeX/BibTeX를 컴파일할 수 있다. 최초 사용 시 TeX 패키지 번들을 내려받을 수 있다. PDF 페이지 수와 렌더링 도구가 없으면 `uv run --no-project --python 3.10 --with pymupdf python ...` 같은 격리된 CPU 도구를 사용해도 된다. 환경 부족을 컴파일 성공으로 보고하지 않는다.

허용 범위: 로컬 원고 작성·편집, 기존 예측 검토, 소규모 CPU 분석/조판 도구 설치, 컴파일·검사, 이 요청에 대한 Discord 완료 보고 및 PDF 첨부.
제외: GPU/서버 실험, 추가 에이전트·봇 호출, commit/push, 외부 투고·게시, 기존 원본 수정. 동일 작업 요청/보고는 각각 1회로 마무리한다.
