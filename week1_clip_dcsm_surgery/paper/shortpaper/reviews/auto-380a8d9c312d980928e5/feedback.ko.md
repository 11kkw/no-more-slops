# 논문 검토 피드백

판정: 수정 후 저자 결정 필요; 현재 arXiv 업로드 준비 완료로 판단하지 않음. 필수 run_local_review를 먼저 실행했고 ChkTeX 1.7.9 실제 실행은 warnings_found, exit_code 2, 진단 33건이며 global resource file 누락도 기록됨. 원문 shortpaper.md와 shortpaper.tex 전체, references.bib, README, figure_data.json, 기존 build_validation.json 일부를 읽음. writer self-review는 읽지 않음. 사용자 최초 입력의 실제 native page-1부터 page-5까지 모두 시각 검사했으며 MEDIA 문자열을 픽셀로 간주하지 않음. 1쪽 제목/초록/본문, 2쪽 수식/방법, 3쪽 표/그림/결과, 4쪽 논의/결론/감사, 5쪽 참고문헌은 대체로 가독성이 좋고 심각한 잘림은 보이지 않음. 표 1 단위, exploratory 구간 표기, float 문장 단절, 여백 및 내부 진행 문구 수정 권고. 장점은 conditional candidate selection, 비대응 입력/훈련, 비인과적 해석과 주석 라벨 한계를 솔직히 밝힌 점. 핵심 관찰은 해당 저장 결과와 scalar-map 프로토콜 범위에서 서술적으로 지지되나 보편적 관계 추론 불가능성 또는 modified-feature learned DCSM 효과를 입증하지 않음. 문자열 비교는 kang2025dcsm.html.txt, lan2024clearclip.pdf.txt, minderer2023owlv2.txt, zhou2021maskclip.pdf.txt 네 로컬 출처에 한정, 참고문헌 제외. exact 최소 8단어 및 RapidFuzz 3.14.6 normalized sentence fuzz.ratio 임계값 90/12–150단어 분절 후보 각각 없음; 표절률/독창성 인증이 아님. 네 출처의 관련 초록/서론/방법 문맥을 읽었으나 전문 전체 수동 검사는 아님. 미검증: CLIP/COCO/OWL-ViT/FG-OVD 원출처 전문, 외부 웹/출판 DB 및 전역 의미 중복; 최신 공식 arXiv 지침; 제출 archive/모든 의존 파일 완전성/clean 및 arXiv-side compilation; 저자/소속/라이선스/카테고리/endorsement/moderation/acceptance; checkpoint 해시/배포 버전/훈련-평가 중복/원실행 revision/하드웨어/전체 환경; adapter 토큰 정렬/행렬 방향/normalization 축; 인간 라벨 검증/독립 일반화; inference 재현/item-level 통계 독립 재계산; figure upstream bootstrap와 생성 코드까지의 독립 재현; 렌더-PDF 재렌더링 기반 동일성. 기존 build_validation은 기록으로만 검토했으며 이번 재실행으로 주장하지 않음. 원본 수정/실험/업로드/Discord 연락은 수행하지 않음.

이 피드백은 현재 원고와 확보한 참고자료의 검토 결과입니다. 전체 문헌에 대한 표절 판정이나 arXiv 승인 판정이 아닙니다.

## R01_AUTHOR_METADATA — required

위치: PDF p.1 제목 아래; shortpaper.tex author/date

저자·소속 미정으로 책임 소재와 제출 메타데이터가 확정되지 않았다. 학술 구조와 PDF 존재는 업로드 준비 완료를 뜻하지 않는다.

수정 제안: 저자가 명단·순서·소속·연락처와 공개 동의를 확정하고 PDF/제출 메타데이터를 일치시키며 작업용 date 라벨을 제거한다.

## R02_MASKCLIP_VARIANT_ATTRIBUTION — required

위치: §2.1 PDF pp.1–2; Table 1/2 MaskCLIP-style

원논문의 ViT residual 설명과 로컬 value-only/no-residual 구성의 차이를 직접 밝혀야 원래 MaskCLIP 결과로 오인되지 않는다. '-style' 단서는 적절하며 부정행위 판단은 아니다.

수정 제안: §2.1에 원형과 달리 이번 value-only variant는 최종 residual/FFN을 제외한다고 원출처 §3.3 인용과 함께 명시한다. value-only (MaskCLIP-inspired) 명칭과 refinement 미사용 여부를 일관되게 표기한다.

## R03_DCSM_IMPLEMENTATION_PROVENANCE — required

위치: §2.3 PDF p.2 DCSM adapter

행렬 방향, tokenizer 위치 정렬 및 normalization 축이 checkpoint 기대와 일치하는지는 미검증이다. 표현상의 전치 차이를 구현 오류라고 단정할 수 없다.

수정 제안: §2.3에 원출처를 직접 인용하고 checkpoint 배포 이름/URL/버전 또는 확보 가능한 해시, 입력 축과 CLS/EOS/padding, functional embedding 삽입 및 normalization 축을 명시한다. 원형과 로컬 adapter 차이 및 미확인 정렬을 유지한다.

## R04_MAIN_CONTRAST_UNCERTAINTY — recommended

위치: §3.2 PDF p.3; Table 2

핵심 ordinary-versus-modified 대비의 paired uncertainty가 없어 marginal interval로 차이의 불확실성을 추론하게 된다. 수치 순위 역전은 서술적으로 지지되나 유의성을 추가 단정할 수 없다.

수정 제안: 이미 저장된 핵심 paired contrast 구간이 있으면 artifact 경로와 함께 추가한다. 없으면 descriptive ranking reversal이며 paired uncertainty unavailable라고 기록한다. marginal interval을 빼서 contrast interval을 만들지 않는다.

## R05_PROBE_DENOMINATOR_AND_PROTOCOL — recommended

위치: §2.2 PDF p.2; Table 1 p.3

500 images는 pointing 성공률 직접 분모가 아니며 집계 규칙이 불명확하다. component pixel 조건 해상도/연결성과 fallback 위치도 short paper만으로 재현하기 어렵다.

수정 제안: 보존된 자료에서 가능한 trial 수, micro/macro 집계와 eligible 조건을 기재하고 upsample 크기/보간/연결성 및 빈 box fallback 규칙을 supplement 또는 본문에 명시한다. 없는 값은 추정하지 않는다.

## R06_TABLE_UNITS_AND_INTERVAL_LABELS — recommended

위치: Abstract p.1; Table 1/2 p.3; TeX 표 정의

실제 PDF에서 단위와 탐색적 구간 성격이 원문보다 덜 명확하다. AP 0–100 단위는 캡션에 있어 적절하다.

수정 제안: PDF header를 Pointing (%)로 하고 Table 2/초록에 exploratory percentile 95% image-cluster bootstrap interval을 명시한다. 생성 과정에서 header 의미가 보존되도록 점검한다.

## R07_FIGURE_SELECTION_AND_LABELS — recommended

위치: Figure 1 및 Table 2 p.3; figure_data.json

selected scorers 단서가 있어 허위 전수표시는 아니지만 핵심 modified arm 생략 이유가 없다. 축약 label은 pooled/context/matched 차이를 그림 단독으로 설명하지 못한다.

수정 제안: MaskCLIP-style을 추가하거나 유사 결과여서 표에만 남겼다는 선택 근거를 캡션에 기재한다. pooled map/context crop/matched 의미와 artifact source/생성 코드 연결을 제공하고 factorial ablation 아님을 유지한다.

## R08_PROSE_INTERNAL_STATUS_AND_EMPHASIS — recommended

위치: §1 p.1; §2 pp.2; §4/6 p.4

범위 제한은 적절하지만 반복 방어 문구와 내부 진행 상태는 학술 서사를 약화한다. AI 지원 및 saved-artifact 분석 공개는 투명성에 중요하다.

수정 제안: 제한사항 중복과 bold를 줄이고 운영 상태를 README/review record로 이동한다. AI 지원 및 비재실행 공개는 유지하고 증거는 접근 가능한 supplement 식별자로 연결하거나 비공개 상태를 명확히 한다.

## R09_PAGE_FLOW_WHITESPACE — minor

위치: PDF pp.3–5; TeX float/clearpage

5쪽 대체로 가독성이 좋고 clipping은 보이지 않지만 float가 문장/구간을 갈라 읽기 흐름을 방해한다. 여백이나 5쪽 자체를 arXiv 위반으로 판단할 근거는 없다.

수정 제안: float/문단 경계를 조정해 단독 문장 조각을 피한다. 목표 형식이 허용하면 참고문헌 강제 새 쪽/balancing을 재검토하고 저자 수정 후 전쪽을 재검사한다. 보편적 2단/4쪽 규칙을 적용하지 않는다.

## R10_CHKTEX_TRIAGE — recommended

위치: shortpaper.tex; 로컬 검사 latex_lint

필수 lint 실행은 충족했으나 warning-free는 아니다. escaped bracket rule 9는 생성기 관용구 오탐 가능성이 있어 문법 오류로 단정하면 안 된다. resource 누락/input traversal disabled로 검사 범위가 제한된다.

수정 제안: command space/dash/sentence spacing/label 주변 경고를 rule별 검토하고 실제 오류는 수정, 오탐은 근거를 남긴다. resource 설정을 갖춘 bounded lint 재검사를 저자 수정 단계에서 실시하고 PDF 표기 보존을 확인한다.

## R11_LOCAL_INTEGRITY_COVERAGE — recommended

위치: §1/2 인용; References p.5; local_similarity/near_similarity

무단 복사 구절을 지목할 문자열 근거는 없지만 네 추출문으로 전역 표절/의미 중복 부재나 모든 인용 정확성을 인증할 수 없다. CLIP/COCO/OWL-ViT/FG-OVD 원출처 전문은 비교에 미포함.

수정 제안: 네 파일 coverage, extraction/string 한계를 기록한다. 추가 원출처가 확보되면 로컬 bounded 비교와 해당 주장 문맥 검토로 다룬다. 무검출/AI-text score를 originality 인증에 사용하지 않는다.

## R12_ARXIV_SOURCE_BUNDLE — required

위치: README build/supporting material; TeX preamble; §6 p.4

학술 구조와 로컬 PDF는 자기완결 source bundle이나 arXiv-side compilation을 보증하지 않는다. 최신 공식 지침 조회가 불가능해 최종 형식 적합성은 미검증이다.

수정 제안: 저자가 main TeX/bib/필요시 bbl/figure/필수 style 의존성을 갖춘 archive와 supplement 공개 계획, 절대 경로 없는 build 절차를 준비한다. 현재 공식 https://info.arxiv.org/help/submit/index.html 및 https://info.arxiv.org/help/submit_tex.html에서 요구를 확인한다. 저자/라이선스/카테고리/endorsement/moderation은 별도 결정이며 acceptance는 인증하지 않는다.

## R13_UNVERIFIED_REPRODUCIBILITY_AND_GENERALIZATION — recommended

위치: §2.3–2.4 pp.2–3; §4 p.4; figure_data/build_validation

한계를 정직하게 유지해야 한다. 기존 기록과 그림 가독성을 inference 재현, 라벨 진실성, 통계 독립 검증 또는 일반화로 확장할 수 없다. 렌더-PDF 동일성도 재렌더링으로 검증하지 않았다.

수정 제안: 증거 manifest에서 확보 가능한 checkpoint/코드/입력 해시, split/중복 상태, figure source와 생성 코드, 이번 verified와 previously-recorded를 분리한다. 없는 정보는 unavailable로 남기고 관찰을 해당 구성 샘플/scorer 범위로 제한한다.
