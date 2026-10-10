# 논문 검토 피드백

판정: 수정 후 재검토 권고. 필수 로컬 검사 및 ChkTeX 1.7.9 실행 완료(exit_code 2, 경고 34건, global resource file 미발견 경고). 원문 shortpaper.md/tex와 references.bib를 먼저 읽었으며 writer self-review 파일은 읽지 않았다. 시작 메시지에 첨부된 실제 PNG 픽셀로 PDF 1–5쪽 전부를 검토했다(MCP MEDIA 경로를 시각 증거로 사용하지 않음). 표와 그림의 숫자·단위는 읽을 수 있고 뚜렷한 잘림이나 겹침은 보이지 않는다. 학술 구조와 인과·일반화 제한은 대체로 적절하며, 주요 기술적 순위 역전은 저장된 측정값과 부합한다. 비교 검사는 로컬 텍스트 4개(Kang DCSM, Lan ClearCLIP, Minderer OWLv2, Zhou MaskCLIP)만 대상으로 했다. 8단어 이상 exact 및 RapidFuzz 문장 ratio 임계값 90의 near-string 후보는 모두 미검출이며 이는 독창성이나 표절 부재의 증명이 아니다. 직접 읽은 출처 범위는 DCSM 도입부 및 §5.1–5.3/부록 B.1–B.2, ClearCLIP 도입부 및 §3, MaskCLIP 도입부 및 §3.3/실험 설정, OWLv2 원문 pp.1–4의 관련 문맥이다. CLIP, COCO, OWL-ViT, FG-OVD의 원문은 로컬 비교·직접 검토 범위 밖이다. 필수 수정은 저자 메타데이터, 미확인 DCSM 어댑터/체크포인트와 원 방법의 구분, 공간 쿼리 생성 규칙의 명시이다. 통계·메트릭 명료화, 편집 및 제출 패키지 문서 보완을 권고한다. 미검증: 외부 웹/데이터베이스 유사도, 전역 의미 유사도, 로컬 출처 전체의 문장별 의미 검토, 나머지 네 인용 원문과 최신 서지 메타데이터, 최신 공식 arXiv 지침(도구 network_access=false), 제출 메타데이터·라이선스·분류·저자 결정·moderation·arXiv측 컴파일/승인, 독립 clean-directory 소스 번들 빌드/압축 내용, build.py 실행 및 내용(텍스트 도구가 .py 읽기를 거부), 원 실행 코드·환경·리비전과 모델 추론 재현, 체크포인트 해시/배포 버전/학습평가 중복, 토크나이저 및 기능 임베딩 정렬, 사람에 의한 라벨/목표 유일성 검증, 원 per-item 자료와 bootstrap의 독립 재계산·paired 구간, pointing 시행 수와 AP/pointing 불확실성, 새 독립 평가·미지 카테고리 일반화·속도 비교, PDF 내부 메타데이터/폰트 포함·접근성 및 인쇄 축소 가독성. README의 이전 빌드 성공 서술은 독립 실행 검증으로 간주하지 않았다. 원고 수정, 실험, 외부 업로드, Discord 연락은 하지 않았다. arXiv 준비는 학술 구조와 별개이며 현재 제출 준비 완료로 판정하지 않는다.

이 피드백은 현재 원고와 확보한 참고자료의 검토 결과입니다. 전체 문헌에 대한 표절 판정이나 arXiv 승인 판정이 아닙니다.

## META-01 — required

위치: PDF p.1 저자행; shortpaper.md 첫머리; shortpaper.tex author

저자·소속이 미정이므로 배포 및 제출 책임과 메타데이터를 확정할 수 없다. 제목·초록·참고문헌 구조가 있다는 사실은 업로드 준비 완료를 뜻하지 않는다.

수정 제안: 저자가 이름·순서·소속을 확정하고 PDF와 제출 메타데이터를 일치시킨다. working short paper 표시는 공개 버전 목적에 따라 제거하거나 명확히 유지한다.

## METHOD-01 — required

위치: 초록; §2.3 PDF p.2 DCSM 단락; §3.2/그림 1

원 방법의 개념적 귀속은 있지만 실제 평가 대상은 로컬 어댑터와 신원이 미확인된 체크포인트의 조합이다. 독자가 released DCSM의 검증된 재현 결과로 읽을 위험이 있다. 전치 자체가 오류라는 판단은 하지 않으며, 정렬·정규화의 구현 동등성은 미검증이다.

수정 제안: 초록과 표/그림에 local DCSM-checkpoint adapter pipelines라는 범위를 일관되게 표시한다. 기존 자료로 공식 배포 경로·버전·파일 해시·어댑터 리비전을 확인할 수 있으면 기록하고, 없으면 미확인을 유지한다. 기능어 위치 매핑·행렬 축 순서·padding·정규화 범위를 원 방법과 구분해 문서화한다. 새 실험은 이 검토에서 요구하거나 실행하지 않는다.

## METHOD-02 — required

위치: §2.3 공간 쿼리 생성; PDF p.2

조건부 선택이라는 범위는 명확하지만 핵심 데이터 구성 규칙 일부가 이름만 제시되어 재구성과 목표 유일성 평가가 어렵다. 참고 인스턴스가 같은 범주의 다른 인스턴스와 어떻게 식별되는지도 텍스트만으로 충분히 드러나지 않는다.

수정 제안: 저장된 생성 코드/자료에서 분리 조건, reference 선택·식별 규칙, 중복·동률 처리, 후보 순서, 정확한 문장 템플릿을 발췌해 짧은 프로토콜 또는 접근 가능한 부록에 명시한다. 실제로 남지 않은 규칙은 추정하지 말고 미확인으로 표시한다.

## STAT-01 — recommended

위치: §2.4 및 §3.2; PDF pp.2–3

서술적 순위 역전과 인과 효과를 구분한 것은 적절하다. 다만 핵심 ordinary-versus-modified 대조 구간은 없고 다른 paired 대조 구간은 있어 통계 근거 접근성이 비대칭적이다. 이 세션은 원 bootstrap/per-item 자료를 독립 재계산하지 않았으므로 구간 계산 정확성을 인증하지 않는다.

수정 제안: 이미 저장된 bootstrap 자료의 정확한 JSON pointer와 대조 정의를 각 paired 구간에 연결한다. 핵심 대조가 보존되지 않았다면 descriptive reversal 표기를 초록·결론에도 유지하고 개별 구간의 비겹침을 paired 검정으로 대체하지 않는다.

## METRIC-01 — recommended

위치: §3.3 PDF p.3 without/with 진단

앞 단락은 raw accuracy와 big/small reweighting을 다루는데 without 단락은 균형 정확도라는 단어와 균형 층 정의가 빠져 있어 raw와 balanced를 혼동할 수 있다. 공간 균형식이 다른 하위집합에도 같은 층으로 적용되는 것으로 오해할 수 있다.

수정 제안: 세 without 수치와 chance를 balanced without accuracy라고 명시하고 이 하위집합의 균형 층과 가중 방식을 정의한다. 기존 표 또는 보충자료에서 raw 값과 분리하여 표기한다.

## PROSE-01 — recommended

위치: §1 마지막 related-work 단락; §3.1/§4/§6

주장의 범위를 신중하게 제한한 점은 장점이다. 그러나 부정형 면책 서술이 반복되어 측정 질문→프로토콜→관찰의 흐름을 약화한다. 학술적 한계와 내부 작성·검토 과정 기록을 구분할 필요가 있다. 과도한 본문 bold 강조는 보이지 않는다.

수정 제안: 기여를 관찰·프로토콜 범위의 긍정형 한 문장으로 정리하고 중복 인과/novelty 면책은 §4에 모은다. saved-artifact 설계와 AI assistance 공개는 유지하되 재실행하지 않았다는 동일 문구의 반복은 줄인다. mean-map, pooled map, Ordinary map 명칭을 한 정의로 통일한다.

## VISUAL-01 — minor

위치: PDF p.3 표 1–2와 §2.4 이어짐; PDF p.4 그림 1/§6; PDF p.5 참고문헌

표 수치·단위·오차막대·캡션은 읽을 수 있고 눈에 띄는 잘림이나 겹침은 없다. 그러나 방법 설명의 페이지 단절, 긴 캡션과 일부 양끝 맞춤 제목, 참고문헌 페이지의 큰 여백은 읽기 흐름을 개선할 여지가 있다. 2단 또는 본문 4쪽을 arXiv의 보편 요구로 보지는 않는다.

수정 제안: chance 정의 문장이 페이지 경계에서 잘리지 않도록 float 배치를 조정한다. 그림 캡션의 방법 차이는 본문으로 옮겨 줄이고 짧은 제목을 사용한다. 참고문헌 단독 페이지는 저자 형식 선택으로 결정하되 불필요한 강제 clearpage 여부를 점검한다. 변경 후 5쪽 전체를 다시 픽셀 검토한다.

## LINT-01 — recommended

위치: shortpaper.tex; 로컬 보고서 latex_lint

필수 linter 실행은 충족했으나 warning-free가 아니다. 신뢰구간의 Pandoc 이스케이프는 실제 렌더에서 정상으로 보이므로 경고를 컴파일 오류로 간주하면 안 된다. global resource file 미발견과 input traversal 비활성화로 설정·검사 범위에 제한이 있다.

수정 제안: 각 경고를 실제 오류/타이포그래피 개선/정당한 false positive로 분류해 기록한다. label은 caption·제목에 붙이고 간격·대시는 문맥별로 교정한다. 중괄호로 보호된 구간 표기에는 근거 있는 제한적 suppression을 사용하며 전체 규칙을 일괄 끄지 않는다. 저자 수정 후 같은 설정으로 재검사한다.

## INTEGRITY-01 — minor

위치: §1/§2.1–2.3 인용 문맥; local_similarity/near_similarity

직접 확인한 방법 설명은 출처를 명시한 기술적 재서술이며, MaskCLIP-inspired 변형과 원 방법 차이도 설명한다. 검토 범위에서 무인용 장문 복제 후보는 없었으나 이것을 표절 부재 또는 전역 독창성 보증으로 해석하지 않는다. 상용·웹 DB 검사, 추출 손실과 의미적 중복은 이 검사로 해결되지 않는다.

수정 제안: 현재 인용과 inspired/style 구분을 유지한다. DCSM functional embeddings가 원 논문의 Functional Rows에 대응한다는 연결을 명시하고 정확한 절 번호를 붙이면 귀속을 더 분명히 할 수 있다. 새 유사도 후보가 생기면 점수 대신 원문과 인용 문맥을 검토한다.

## INTEGRITY-02 — recommended

위치: References PDF p.5; 참고문헌 중 로컬 원문 미제공 항목

참고문헌 키 완전성과 해당 논문 내용·연도·저자·URL 정확성은 다른 검사다. 로컬로 제공되지 않은 네 원문의 주장 지원 및 최신 공식 서지정보는 이번 세션에서 미검증이다.

수정 제안: 저자가 원문 사본 또는 공식 메타데이터를 제공한 뒤 특히 FG-OVD의 fine-grained 평가 범위와 OWL-ViT 관련 서술을 대조한다. 현재 제공된 first-submission year 정책은 일관되게 설명하되 미검증을 verified로 바꾸지 않는다.

## ARXIV-01 — recommended

위치: README Build and validate / source bundle; §6 Evidence Availability

TeX, bib, bbl, 사용 그림의 존재와 로컬 PDF 가독성은 확인했지만 독립 재현 빌드나 압축 번들 내용은 검증하지 못했다. 최신 공식 arXiv 지침에 직접 접근할 수 없으므로 이전 저자의 지침 확인 서술을 이 리뷰의 확인으로 대체하지 않는다. 학술 형식과 실제 업로드·승인 준비는 분리해야 한다.

수정 제안: 저자가 제출용 최소 번들의 파일 목록·해시·상대 경로·엔진/TeX 의존성을 제공하고 사용자 홈 절대경로 없는 빌드 방법을 문서화한다. 공개 가능한 증거·부록의 접근 경로를 정한다. 제출 직전 공식 https://info.arxiv.org/help/submit_tex.html 및 https://info.arxiv.org/help/submit/index.html의 최신 지침을 직접 대조하고 저자·라이선스·분류를 결정한다. 로컬 빌드를 arXiv측 컴파일 또는 acceptance로 표시하지 않는다.
