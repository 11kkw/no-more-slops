# 논문 검토 피드백

판정: 수정의 효과는 확인되었으나 필수 근거 및 제출 메타데이터가 남아 있어 수정 필요. 현재 shortpaper.md/tex와 시작 입력에 네이티브 이미지로 첨부된 PDF 1–5쪽을 먼저 독립 검토한 뒤 reviews/latest_writer_response.json 및 그 파일이 지목한 auto-c322eb7a99a934eda26c/editorial_review.json을 읽었다. 저자 응답의 addressed를 그대로 채택하지 않고 현재 본문·픽셀·출처·측정 JSON으로 대조했다. METRIC-01과 PROSE-01의 문장 수정은 확인되며, METHOD-01의 released 표현 제거 및 로컬 어댑터 범위 제한도 확인된다. METHOD-02는 미확인 규칙을 공개했지만 프로토콜 재구성 문제는 남는다. STAT-01은 서술적 순위 역전 및 저장 대조 경로를 명료화했지만 계산 검증은 남는다. VISUAL-01은 그림이 현재 p.3에 있으며 p.3 chance 정의가 완전한 문장으로 재개되어 이전 지적 일부가 개선되었다. 제목 간격·캡션 길이·참고문헌 여백은 남는다. INTEGRITY-01의 기능 임베딩과 Functional Rows의 명시적 연결은 저자 응답의 주장만큼 본문에 분명하지 않아 부분 해결로 판정한다. META-01, LINT-01, INTEGRITY-02는 미해결이며 ARXIV-01은 한계 공개만 개선되었다. 필수 run_local_review 및 ChkTeX 1.7.9 실제 실행 완료: exit_code 2, 진단 34건, global resource file 미발견. 섹션·인용 키·그림 파일·PDF 가독성 검사 통과; 경고 없는 원고로 판정하지 않는다. PDF 전체 픽셀에서 눈에 띄는 잘림·겹침은 없고 표 값, AP/정확도 단위, 그림 축·오차막대·chance 선은 읽힌다. 자동 exact 검사는 8단어 이상, near 검사는 RapidFuzz 문장 ratio 임계값 90이며 제공 로컬 출처 4개 모두 후보가 검출되지 않았다. 후보 점수는 표절 판정이 아니며 미검출은 독창성 보증이 아니다. 비교 범위는 Kang DCSM, Lan ClearCLIP, Minderer OWLv2, Zhou MaskCLIP의 제공 텍스트뿐이다. 직접 문맥을 읽은 범위는 DCSM 도입부/§5.1–5.3/부록 B.1–B.2, ClearCLIP 도입부 및 §3.1–3.3 일부, MaskCLIP 도입부 및 §3.3, OWLv2 원문 pp.1–4 일부이다. 로컬 출처 전체를 문장별 의미 비교한 것은 아니다. 주요 결과의 값과 raw/balanced 구분은 evidence_export/evidence/measurements.json 및 figure_data.json과 대조했으며 독립 수치 재계산·모델 추론은 수행하지 않았다. 미검증 목록: 외부 DB/웹 비교 및 전역 의미 유사도·독창성; CLIP/COCO/OWL-ViT/FG-OVD 원문과 최신 공식 서지정보; DCSM 배포 신원·체크포인트 해시/버전·원 실행 리비전·학습평가 중복·기능어 토크나이저 정렬·정규화 동등성; 생성 코드/원 쿼리의 정확한 분리·reference·후보 순서·동률·템플릿 규칙; 사람 라벨/목표 유일성 검증; 원 per-item 자료·bootstrap 구현/paired 구간의 독립 계산과 provenance; pointing 시행 수 및 AP/pointing 불확실성; 독립 새 평가·미지 범주 일반화·속도; 빌드 스크립트 실행·소스 번들 압축 내용/해시/현재 입력 대응 및 독립 clean-directory 빌드; PDF 내부 메타데이터·폰트 포함·접근성·축소 인쇄 가독성; 최신 공식 arXiv 지침(현재 도구 network_access=false), 저자/소속·라이선스·분류·공개 증거 접근 결정·arXiv측 컴파일·moderation·승인. ../evidence/cluster_bootstrap.json 직접 읽기는 경로 제한으로 거절되었으며 build_validation의 과거 검증 기록은 이번 독립 실행으로 간주하지 않았다. 학술 구조는 적절하나 arXiv 업로드 준비 완료가 아니다. 2단/본문 4쪽은 보편 arXiv 의무가 아니다. 원고 변경·실험·외부 업로드·Discord 연락은 하지 않았다. 후속 자동 실험이나 무한 수정 루프는 제안하지 않으며 남은 근거와 저자 결정을 기록한다.

이 피드백은 현재 원고와 확보한 참고자료의 검토 결과입니다. 전체 문헌에 대한 표절 판정이나 arXiv 승인 판정이 아닙니다.

## META-01 — required

위치: PDF p.1 저자행; shortpaper.md 첫머리; shortpaper.tex author

이전 지적 미해결. 저자·소속 placeholder가 실제 PDF에 남아 있어 공개 및 제출 메타데이터를 확정할 수 없다. 제목과 초록의 존재만으로 제출 준비가 완료되지 않는다.

수정 제안: 저자가 이름·순서·소속을 확정한 뒤 원문·PDF·제출 메타데이터를 일치시킨다. working short paper 표시는 공개 목적에 따라 결정한다.

## METHOD-01 — required

위치: 초록; §2.3 PDF p.2; 표 2/그림 1 PDF p.3; 결론 p.4

이전 released DCSM 표현은 제거되어 귀속 범위가 개선되었다. 그러나 배포 신원·해시·버전·정렬·정규화 동등성은 여전히 미확인이다. 제한을 공개한 것은 해결의 증거가 아니며 로컬 어댑터 결과를 원 방법의 재현으로 일반화할 수 없다. 행렬 전치 자체를 오류라고 판정하지 않는다.

수정 제안: 가능한 기존 기록으로 실제 체크포인트 신원·파일 해시·어댑터 및 실행 리비전을 기록한다. 기록이 없으면 현재 local adapter 표기를 유지하고 공식 DCSM 재현 주장과 분리한다. 원 방법 대비 축 순서·padding·기능어 처리·정규화 범위를 프로토콜 표에 정리한다.

## METHOD-02 — required

위치: §2.3 PDF p.2 공간 쿼리 구성

이전 재구성 부족은 미해결이며 새로 미검증 목록을 명시한 점만 개선되었다. 0.05 center margin과 seed/cap만으로 표본 생성과 reference 유일성을 재구성할 수 없다. 저자 응답의 partial 판정과 일치한다.

수정 제안: 기존 생성 코드/쿼리 자료를 제공할 수 있을 때 실제 분리 임계값·reference 선택·중복/동률·후보 순서·정확한 템플릿을 접근 가능한 부록에 발췌한다. 자료가 없으면 추정하지 말고 재구성 불가 범위를 명시하고 construction audit라는 기여 표현을 그 범위에 맞춘다.

## STAT-01 — recommended

위치: §2.4 PDF pp.2–3; §3.2 PDF pp.3–4; figure_data.json

주장 강도와 경로 공개는 개선되었다. 그러나 paired 구간의 정의·원 계산 provenance·정확성은 독립 검증되지 않았고 핵심 ordinary 대 modified 대조 구간은 없다. 그림의 marginal interval 비겹침으로 paired 검정을 대체해서는 안 된다.

수정 제안: 보존된 두 ObjA 대조의 정확한 키와 정의를 현재 패키지에서 읽을 수 있는 증거 목록에 연결한다. 핵심 대조가 없다면 현재 descriptive 표기를 유지한다. 기록된 결과 일치와 독립 계산 검증을 구별한다.

## METRIC-01 — minor

위치: §3.3 PDF p.4 without/with 진단; evidence_export/evidence/measurements.json

이전 지적의 문장·메트릭 구분은 실제로 해결되었다. without의 균형 층을 size의 big/small 및 raw 정확도와 구분하여 독자 혼동을 줄였다. 저장 값의 문맥 대응 확인이며 원 per-item 계산 재현 인증은 아니다.

수정 제안: 현재 balanced 표기와 층 정의를 유지하고 향후 표/부록에서도 raw와 balanced를 분리한다.

## PROSE-01 — minor

위치: §1 기여 목록; §2.1 pooled-text map 정의; §4 및 Evidence Availability

이전 지적의 주요 편집은 해결되었다. 연구 질문→프로토콜→관찰 흐름과 용어 정의가 명료하며 AI assistance 공개도 유지되었다. 미확인 release와 인과 한계의 반복은 다소 남지만 근거상 필요한 제한이며 삭제로 강한 주장처럼 만들면 안 된다.

수정 제안: 긍정형 기여·pooled-text 정의를 유지한다. 필요하면 동일한 release 미확인 문구만 한 번의 방법 설명과 짧은 표 주석으로 압축하되 초록/결론의 주장 범위를 보존한다.

## VISUAL-01 — minor

위치: 네이티브 첨부 PDF pp.1–5; 특히 p.3 표/그림, p.4 §6, p.5 참고문헌

이전 리뷰의 페이지 경계 문장 문제와 그림 배치는 실제 개선되었다. 5쪽 모두 잘림이나 겹침은 눈에 띄지 않는다. 긴 캡션·양끝 맞춤 제목·참고문헌의 큰 여백은 남아 있으며 원고 형식 선택의 문제다. 그림에서 MaskCLIP 생략 이유는 캡션에 명시되어 있어 선택 은폐로 보지 않는다.

수정 제안: 제목을 줄이거나 heading 정렬을 조정하고 캡션에서 본문에 이미 있는 설명을 줄인다. 강제 clearpage 및 참고문헌 독립 페이지는 목표 형식에 따라 결정한다. 그림 라벨에 local adapter를 직접 넣으면 단독 재사용 시 범위가 더 분명하다. 변경 시 전체 페이지 픽셀을 재확인한다.

## LINT-01 — recommended

위치: 현재 shortpaper.tex; 자동 보고서 latex_lint

필수 linter 실행은 충족했지만 이전 경고 문제는 미해결이다. Pandoc 구간 escaping의 경고는 픽셀상 정상이라 컴파일 실패로 취급할 근거가 없다. 설정 파일 부재 및 input traversal 비활성화는 검사 범위의 제한이다.

수정 제안: 현재 줄 번호로 경고별 실제 오류·타이포그래피 개선·정당한 false positive를 분류한다. 생성 단계에서 label 인접 공백과 dash/spacing을 문맥별로 교정하고 escaping에는 근거 있는 제한적 suppression만 적용한다. 동일 설정으로 재검사하며 warning-free로 과장하지 않는다.

## INTEGRITY-01 — recommended

위치: §1/§2.1–2.3 출처 문맥; 로컬 exact/near 결과

읽은 문맥에서 인용된 기술적 재서술과 style/inspired 구분은 적절하며 무인용 장문 복제 후보는 발견되지 않았다. 다만 저자 응답의 addressed를 완전 해결로 채택할 수 없다. 체크포인트 기능 임베딩과 원 Functional Rows의 연결이 실제 본문에서는 여전히 암시적이다. 미검출은 표절 부재 보증이 아니다.

수정 제안: §2.3에 원 DCSM의 Functional Rows는 기능어 위치의 유사도 행을 고정 벡터로 대체한다는 짧은 인용 설명을 추가하고, 로컬 inherited functional embeddings와 위치 정렬의 미검증을 구분한다. 구현 동등성을 확인한 것처럼 쓰지 않는다.

## INTEGRITY-02 — recommended

위치: PDF p.5 References; references.bib 및 원문 References

이전 출처 커버리지 부족은 미해결이다. 키와 서지 레코드가 존재하는 것과 해당 원문의 주장 지원·공식 저자/연도/URL 확인은 다르다. 현재 verified라는 내부 표현을 이번 독립 심사의 확인으로 간주하지 않는다.

수정 제안: 나머지 원문 또는 공식 메타데이터와 확인 기록을 제공해 특히 FG-OVD 및 OWL-ViT 주장을 대조한다. verified 표현은 검증 기록에 연결하거나 stated first-submission-year policy로 바꾼다. 비교 커버리지 4개라는 제한을 유지한다.

## ARXIV-01 — recommended

위치: Evidence Availability PDF p.4; README Build and validate/Supporting material

한계 공개는 실제 개선되었으나 현재 입력에 대응하는 최소 번들 내용·해시·이식성·독립 빌드는 미검증이다. 제목·초록·TeX·bib·그림 파일과 5쪽 PDF는 존재하지만 업로드 준비 완료를 뜻하지 않는다. README의 과거 공식 지침 확인과 빌드 성공은 이번 심사의 실행 증거가 아니다.

수정 제안: 저자가 현재 제출 번들의 파일 목록·해시·상대 경로·엔진/의존성 및 공개 증거 접근 방법을 정리한다. 제출 직전 공식 https://info.arxiv.org/help/submit_tex.html 및 https://info.arxiv.org/help/submit/index.html을 직접 확인하고 저자·라이선스·분류를 결정한다. 로컬 형식이나 빌드를 arXiv측 컴파일/승인으로 표시하지 않는다.
