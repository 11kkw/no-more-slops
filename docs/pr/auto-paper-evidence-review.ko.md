완료된 실험 결과와 원고 사이의 수치·주장 근거를 추적하고, 작성된 문서를 로컬에서 검토할 수 있는 두 skill과 실행 도구를 제공합니다. 저장된 JSON에서 선택한 값을 표로 옮기고 원고를 수정했을 때, 이전 입력에 대한 검토를 현재 원고의 검토로 재사용하지 않도록 입력 해시를 함께 기록합니다.

## 논문 예제

[실제 DCSM 공간관계 논문 PDF · 본문 4쪽 + 참고문헌 1쪽](https://github.com/11kkw/no-more-slops/blob/c19a2ccd434dfd81617e58e7defec997c61072a6/examples/auto-paper/dcsm-spatial/shortpaper.pdf)를 함께 제공합니다. 저자·소속과 독립 검토가 남아 있는 연구 초안입니다.

[한국어 사용 안내](https://github.com/11kkw/no-more-slops/blob/feat/auto-paper-evidence-review/docs/auto-paper-evidence-review.ko.md) · [공유·설치 및 검증 기록](https://github.com/11kkw/no-more-slops/blob/feat/auto-paper-evidence-review/examples/auto-paper/README.md)

## 변경 사항

- `experiment-to-paper`: 측정값·단위·분모·비교 조건에서 주장과 원고를 구성하는 skill 및 프로젝트 구조에 의존하지 않는 JSON evidence exporter
- `paper-review-local`: 섹션·인용 키·그림·PDF·ChkTeX·제공된 로컬 텍스트의 overlap 후보 검사, 입력 해시에 연결된 검토 기록, 로컬 CLI와 선택적 MCP
- 실제 DCSM 논문 예제: 5쪽 PDF, 문서 소스·그림·측정값, 78개 표 수치의 export spec, 원본 해시, 주장-근거표, 새 디렉터리의 문서 재빌드 명령
- 가상 결과 예제, 한국어 사용 안내, 한국어·영어 README 진입 링크, Hermes 설치 안내
- Python 3.10/3.12용 GitHub Actions와 복사한 skill을 다른 프로젝트에서 실행하는 테스트

각 skill 폴더 전체를 복사해 사용할 수 있습니다. 개인 경로·agent 프로필·Discord·systemd 설정에 의존하지 않습니다. 실험 실행, 외부 업로드와 논문 제출은 수행하지 않습니다.

## 검증

- 로컬 Python 3.10/3.12 각각 동작 테스트 24개 통과: 결측값/0, 잘못된 값·selector, 기존 출력 보존, 경로·symlink 범위, 입력 변경 후 검토 저장 거부, 별도 프로젝트와 실제 MCP stdio 실행
- 설치된 Hermes의 frontmatter parser와 보조 파일 탐색 검사 통과; 임시 Hermes 홈에서 실제 목록 처리기로 두 skill 조회 확인
- SKILL.md에 직접 링크한 파일만 복사한 설치 번들에서도 가상 예제와 로컬 검토 실행 확인
- 실제 예제의 표 수치 78개와 원본 파일 해시 일치, bibliography keys·figure references·5쪽 PDF 읽기 검사 통과
- 로컬 설정의 ChkTeX 종료 코드 0·진단 0개; stderr의 설정 환경 안내는 보고서에 보존
- 새 디렉터리에서 pdfLaTeX → BibTeX → pdfLaTeX → pdfLaTeX 모두 종료 코드 0; 저장된 PDF와 5쪽 전체 추출 텍스트 일치

[Upstream PR의 GitHub Actions 실행](https://github.com/Pseudo-Lab/no-more-slops/actions/runs/38225864365)에서 Python 3.10/3.12 검사 모두 통과했습니다. 예제에 대한 저자·소속과 독립 editorial review는 미완료로 표시했습니다. 문서 빌드와 정적 검사 결과를 실험 재현, scientific validity, originality 또는 투고 승인으로 보고하지 않습니다.

## 재현

```sh
python -m pip install -r skills/paper-review-local/scripts/requirements-mcp.txt
python -B -m unittest discover -s tests/auto-paper -v
python skills/experiment-to-paper/scripts/export_evidence.py \
  --root examples/auto-paper/dcsm-spatial --spec export-spec.json \
  --out /tmp/auto-paper-evidence-new
python skills/paper-review-local/scripts/review_mcp.py \
  --root . --package examples/auto-paper/dcsm-spatial --review-id initial-01
```

ChkTeX와 문서 재빌드 설치 조건, 출력 상태의 해석은 [예제 README](https://github.com/11kkw/no-more-slops/blob/feat/auto-paper-evidence-review/examples/auto-paper/README.md)에 있습니다. Exporter는 입력값 전사와 해시를 검사하며 집계 재계산은 하지 않습니다. Overlap 검사는 제공된 영어 source text에 한정된 검토 후보이며 global plagiarism percentage가 아닙니다.

이 결과물은 가짜연구소(Pseudo Lab) No-More-Slops 그룹 활동의 일환으로 제작되었습니다.
