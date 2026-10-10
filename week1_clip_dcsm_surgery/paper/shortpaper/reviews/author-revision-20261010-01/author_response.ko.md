# 최신 리뷰에 대한 저자 수정 및 재검토

## 범위와 판정

사용자 요청: “검토 다시 진행해줘 논문 담당은 그것읽고 패치 진행해주세요”.
입력 리뷰: `../auto-380a8d9c312d980928e5/feedback.ko.md` 및 최신 원고.
한 차례의 원고 패치와 bounded 재검사를 수행했다. 과거 원고·PDF·생성기·검사 기록은 이 폴더의 `before-*`에 보존했다. 수정 차이는 `manuscript.patch`에 있다.

판정: **검토 기반 패치 완료, 제출 준비 미완료**. 실험·GPU 작업·외부 업로드·commit/push는 하지 않았다. 이번 재검토는 저자 측 확인 및 로컬 결정론적 검사이며 새 독립 동료심사가 아니다.

## 13개 리뷰 항목 처리

| ID | 처리 | 변경·근거·남은 사항 |
|---|---|---|
| R01 | 부분 반영 / 사용자 결정 필요 | TeX의 작업용 date 라벨 제거. 저자·소속은 추정하지 않고 placeholder 유지. 명단·순서·소속·연락처·공개 동의는 사용자 확정 필요. |
| R02 | 반영 | §2.1에서 MaskCLIP ViT 원형과 달리 value-only variant는 최종 residual/FFN을 생략하고 key smoothing/prompt denoising을 쓰지 않음을 명시. `sources/zhou2021maskclip.pdf.txt` §3.3, 특히 residual 설명 343–347행 및 refinement 설명 70–72행. 표의 MaskCLIP-style은 원형 재현이 아닌 variant 표기를 유지. |
| R03 | 부분 반영 / 미확인 유지 | §2.3에 DCSM 직접 인용, 행렬 방향 197 image positions × 30 text positions, image CLS 및 text BOS/EOS/padding, 후보 전체 텐서 normalization 축 추가. 코드 근거 `../../experiment/scripts/compositional_eval.py:81–109`; whitespace functional embedding 삽입은 101–106행. checkpoint 기대와의 정렬, 배포 버전·해시는 미확인. 로컬 기본 이름은 `data/dcsm_ckpt/dcsm_obja_ckpt.pt`, `data/dcsm_ckpt/dcsm_coco_ckpt.pt` (176행). 실행 당시 파일 해시가 아니라 소스의 기본 경로만 확인한 것임. |
| R04 | 반영 / 통계 미확인 유지 | 보존된 `../evidence/cluster_bootstrap.json` spatial/paired_differences에 ordinary-versus-modified 대비가 없음을 확인. §3.2에 descriptive ranking reversal이며 paired interval unavailable라고 명시. 주변 CI를 빼서 대비 CI를 만들거나 유의성을 주장하지 않음. |
| R05 | 부분 반영 / 분모 미확인 유지 | pointing은 image–category trial의 micro 평균이며 500은 이미지 수이지 trial 분모가 아님을 명시. 정확한 trial 수는 summary에 없어서 추정하지 않음. 코드 `probe_localization.py:223–245`. P1 bilinear upsample, align_corners=False, 입력 square 크기와 four-connectivity 기재(138–155행, scipy ndimage.label 기본 연결 구조). P2 빈 membership은 box center가 속한 grid patch로 clamped fallback임을 기재(`dcsm_detection_eval.py:170–180`). |
| R06 | 반영 | 생성기가 표 헤더 (%)를 버리던 문제 수정: 실제 PDF Table 1 Pointing (%). 초록·Table 2·Figure 1의 exploratory percentile 95% image-cluster bootstrap 의미를 명시. 수치는 변경하지 않음. |
| R07 | 반영 | Figure 1에서 MaskCLIP-style을 생략한 이유(보고 정밀도에서 ClearCLIP-style과 같은 추정치), map/context/matched 정의 추가. `build.py:17–35`가 `../evidence/cluster_bootstrap.json#/subsets/spatial/balanced`를 읽어 그림을 생성하며 `figure_data.json`에 키·값·오차를 저장. factorial ablation이 아님을 유지. |
| R08 | 반영 | 핵심 주장·조건부 선택·CNN 미평가의 반복 bold를 일반 문장으로 변경. 내부 신규성/저자/투고 준비 상태 문구는 학술 본문에서 제거하고 이 기록과 README에서 관리. AI 지원, saved-artifact 분석, 비공개 supplement와 inference 비재실행 공개는 유지. |
| R09 | 부분 반영 / minor 유지 | 날짜 제거와 문단·캡션 수정 후 전 5쪽을 실제 재렌더링해 확인. 표·그림·캡션·본문 잘림이나 겹침은 발견하지 않음. 2→3쪽 및 3쪽 단 전환에서 문장/단어 분할, 4쪽 그림 아래 짧은 diagnostics 결말은 여전히 있어 흐름 개선은 완전 해결로 주장하지 않음. 본문 4쪽·2단과 참고문헌 별도 쪽은 기존 요청에 따라 유지. |
| R10 | 재실행 및 분류 / 경고 유지 | ChkTeX 1.7.9 실제 재실행: exit 2, 34진단. 추출한 배포 resource를 -l로 지정했으나 global resource discovery 경고도 남음. rule 9는 Pandoc의 `{[}`/`{]}` 구간 출력, rule 8은 image--text 등 기술적 연결/Appendices A--B 범위, rule 1은 raggedbottom/tightlist/balance/nobalance 선언, rule 13은 AP/CLS/CNN/EOS 및 약어 뒤 문장 간격, rule 24는 caption 뒤 label/본문 끝 anchor의 공백 제안. 모두 문맥과 실제 PDF를 확인했으나 warning-free나 완전 lint 환경으로 주장하지 않음. 심각한 TeX 오류·undefined citation·수평 overflow는 이번 compile에서 발견되지 않음. |
| R11 | 범위 기록 / 미검증 유지 | 로컬 source 4개 대상으로 exact 8-word와 RapidFuzz 3.14.6 sentence comparison을 재실행하여 후보 없음. 네 source는 MaskCLIP/ClearCLIP/DCSM/OWLv2 추출문이며 나머지 4개 인용 전문·웹 DB·의미 중복 검사는 포함하지 않음. 표절률/독창성 인증 아님. |
| R12 | 부분 반영 / 제출 미확인 | 최신 공식 arXiv Submission Overview 및 TeX Submissions 원문을 조회함. main TeX, bib, bbl, 실제 사용 PDF figure만의 로컬 archive를 생성하고 별도 clean-build 디렉터리에서 Tectonic 실제 컴파일(exit 0), 5쪽 및 페이지별 추출 텍스트 동일성 확인. arXiv TeX Live compilation, 배포 package-version 호환성, author/license/category/endorsement/moderation은 미검증. 업로드 안 함. |
| R13 | 반영 / 재현 미확인 유지 | 저장 수치와 이번 컴파일·렌더 검증을 구분. checkpoint/원실행 revision/환경/훈련 중복/인간 라벨/일반화/inference 재현은 여전히 unavailable 또는 unverified. 그림 데이터 입력 및 생성기 provenance 제공. 이번 renders는 build.py가 이번 PDF에서 직접 생성한 것으로 전쪽 픽셀 검사. |

## 실제 검사 결과

- Tectonic 재컴파일: exit 0, 본문 4쪽 + 참고문헌 1쪽 = 5쪽.
- `validate.py`: 17개 검사 중 16개 통과. 수치, 두 표, 기존 paired 차이, 그림 입력, 인용·참고문헌, 페이지·영역·2단 검사는 통과.
- 유일한 실패: 과거 `input_snapshot.json`과 현재 `plan/03_paper_pipeline.md`의 해시 불일치. 이 파일은 이번 패치 명령의 대상이 아니었다. 누가 언제 바꿨는지는 검증하지 않았다. 과거 snapshot을 덮어쓰거나 해당 계획서를 되돌리지 않았으며 `build_validation.json`의 needs_revision을 유지한다.
- TeX log: underfull 15건; overfull 0건. 따라서 warning-free는 아님.
- 실제 1–5쪽 PNG 픽셀 검사: 초록 완전성, 표 단위, 수식, 그림과 캡션, 감사·결론, 참고문헌 8건 확인. 심각한 겹침·잘림 없음. 자연스럽지만 다소 불편한 페이지/단 전환은 R09로 남긴다.
- 별도 로컬 recheck: `../author-recheck-20261010-01/local_checks.json`의 구조·인용키·figure 존재·PDF 판독·PDF hash 검사 통과. 새 독립 editorial review가 아니라 기존 리뷰 반영 후의 author-side recheck다.
- clean source archive: `source-bundle.tar.gz`; 검증은 `source_bundle_validation.json`, 실행 로그는 `clean-build.log.txt`.

## 공식 arXiv 근거

2026-10-10 실제 조회:
- https://info.arxiv.org/help/submit/index.html — Formats for text of submission, File names and case sensitivity, Verify and correct your submission.
- https://info.arxiv.org/help/submit_tex.html — Submissions are automatically processed, Considerations for (La)TeX submissions, Figure inclusion, We do not have your style files or macros.

원문은 TeX 원본과 사용 figure의 완전성, extraneous 파일 제외, 제출 시스템의 PDF 직접 확인을 요구한다. 로컬 source archive는 제출 가능한 후보 패키지일 뿐 arXiv 승인·수락을 보증하지 않는다. 본문 4쪽·2단은 사용자의 작업 형식이며 보편적 arXiv 규칙으로 해석하지 않는다.

## 미정·다음 제안

저자 정보, checkpoint provenance, tokenizer 정렬 및 통계/라벨의 독립 검증, 공개 supplement 정책은 미정이다. 별도 실험이나 서버 작업은 추가로 시작하지 않는다. 제출 결정을 내리기 전에 저자 정보 확정과 위 미확인 항목에 대한 사용자 판단이 필요하다.
