# Auto-Paper: 완료된 실험에서 검증 가능한 원고로

저장된 결과에서 수치·입력 해시·주장의 근거를 정리하고, 논문 원고와 PDF를 로컬에서 검토하는 두 skill입니다. 프로젝트별 실험 코드는 그대로 두고, 완료된 결과를 글로 연결하는 절차를 재사용합니다.

| Skill | 역할 | 실행 도구 |
| --- | --- | --- |
| `experiment-to-paper` | 결과 → 근거표 → 범위가 명확한 주장 → 원고 | 일반 JSON 값 선택·해시·표 export |
| `paper-review-local` | 원고 → 정적 검토 → 내용·PDF 검토 → 제한된 수정 | 로컬 CLI, 선택적 stdio MCP |

## 설치와 재현

Python 3.10 이상을 사용합니다. 각 `skills/<skill-name>/` 폴더 전체를 사용하는 agent의 skill 디렉터리에 복사하면 됩니다. 특정 agent 프로필이나 서버 설정이 필요하지 않습니다. 아래 명령은 저장소 루트에서 실행합니다.

```sh
python -m venv /tmp/auto-paper-env
. /tmp/auto-paper-env/bin/activate
python -m pip install -r skills/paper-review-local/scripts/requirements.txt

python skills/experiment-to-paper/scripts/export_evidence.py \
  --root examples/auto-paper/dcsm-spatial --spec export-spec.json \
  --out /tmp/auto-paper-evidence-new

python skills/paper-review-local/scripts/review_mcp.py \
  --root . --package examples/auto-paper/dcsm-spatial --review-id initial-01
```

Exporter는 새 출력 폴더를 요구합니다. 검토 결과는 예제의 `reviews/initial-01/local_checks.json`에 저장되며 Git에서 제외합니다. 입력이나 검사 환경이 바뀌면 새로운 review ID를 사용합니다. 실행 성공은 보고서 저장 성공을 뜻하므로 각 검사 상태와 required finding을 읽어야 합니다.

PDF 읽기는 `pypdf`, 문장 문자열 비교는 `RapidFuzz`를 사용합니다. ChkTeX는 운영체제에 별도로 설치합니다(예: Ubuntu의 `apt install chktex`). 미설치 상태를 통과로 바꾸지 않습니다. 선택적 설정과 MCP 실행은 [local-tools.md](../../skills/paper-review-local/references/local-tools.md)에 있습니다.

다른 프로젝트에서는 [evidence-schema.md](../../skills/experiment-to-paper/references/evidence-schema.md)의 작은 JSON spec에 파일·JSON pointer·단위·분모를 지정합니다. 이 exporter는 전사와 해시를 검증하며, 집계 계산은 해당 프로젝트의 분석 코드로 별도 검증해야 합니다.

## 실제 예제와 검증 범위

[DCSM 공간관계 원고](dcsm-spatial/shortpaper.pdf)는 기존 연구에서 작성한 본문 4쪽·참고문헌 1쪽의 PDF입니다. 문서 소스, 그림, 참고문헌, 저장된 측정값과 78개 표 셀의 export spec을 포함합니다. [artifact_manifest.json](dcsm-spatial/artifact_manifest.json)은 원본 파일 해시와 출처 커밋을 기록하고, [claim_evidence.md](dcsm-spatial/claim_evidence.md)는 주장과 측정 범위를 연결합니다.

예제는 supplied-box conditional selection을 다루며 새로운 end-to-end detection 성능을 주장하지 않습니다. 원 측정값의 size/color 진단은 보존돼 있지만 개정 본문 표는 spatial/negation/affirmation만 사용합니다. 저자·소속, 독립 editorial review, 원본 데이터 identity audit와 novelty는 미완료입니다. 원 논문 전문은 재배포하지 않아 예제의 source-text similarity는 `no_local_sources`입니다. 이는 표절이나 독창성 판단이 아닙니다.

TeX 환경에 pdfLaTeX와 BibTeX가 있으면 새 폴더에서 문서를 다시 빌드합니다:

```sh
python examples/auto-paper/dcsm-spatial/rebuild.py --out /tmp/auto-paper-pdf-new
```

실행 파일은 `PDFLATEX`, `BIBTEX`로 지정할 수 있습니다. TeX preamble에 선언된 패키지와 newtx 폰트가 필요합니다. 검증한 로컬 환경은 TeX Live 2026입니다. 이 빌드는 원본 PDF를 덮어쓰지 않으며 입력 해시·명령 종료 코드·로그를 새 폴더에 남깁니다. 동일 호스트의 문서 재빌드는 실험 재현이나 arXiv-side 빌드가 아닙니다.

검증 기록은 [validation.json](validation.json)에 있습니다. GPU 실행, 데이터셋·체크포인트, Discord/systemd 자동화, 외부 provider 업로드는 이 패키지에 포함하지 않습니다.

## Hermes에서 공유·설치

PR 검토 중에는 fork의 제출 브랜치에서 조회·설치할 수 있습니다. 전체 URL 설치를 실행한 검증 결과는 아니며, 로컬에서는 Hermes parser·목록 처리기와 보조 파일만 복사한 번들을 검증했습니다. 의존성 설치는 앞의 안내를 따릅니다.

```sh
hermes skills inspect https://raw.githubusercontent.com/11kkw/no-more-slops/feat/auto-paper-evidence-review/skills/experiment-to-paper/SKILL.md
hermes skills install https://raw.githubusercontent.com/11kkw/no-more-slops/feat/auto-paper-evidence-review/skills/experiment-to-paper/SKILL.md
hermes skills install https://raw.githubusercontent.com/11kkw/no-more-slops/feat/auto-paper-evidence-review/skills/paper-review-local/SKILL.md
hermes skills list
```

아래 upstream URL은 PR이 머지된 뒤 사용할 주소입니다.

```sh
hermes skills install https://raw.githubusercontent.com/Pseudo-Lab/no-more-slops/main/skills/experiment-to-paper/SKILL.md
hermes skills install https://raw.githubusercontent.com/Pseudo-Lab/no-more-slops/main/skills/paper-review-local/SKILL.md
hermes skills list
```

GitHub 경로로 내용을 먼저 조회할 수 있습니다(머지 후):

```sh
hermes skills inspect Pseudo-Lab/no-more-slops/skills/experiment-to-paper
hermes skills inspect Pseudo-Lab/no-more-slops/skills/paper-review-local
```

전체 폴더를 수동 설치하려면 각 skill 폴더를 현재 Hermes 프로필의 `skills/` 아래에 복사하고 `hermes skills list`로 확인합니다. 기본 프로필 위치는 `~/.hermes/skills/`입니다. 설치는 보조 파일을 복사하는 과정이며 Python 의존성 설치·MCP 서버 등록까지 수행하지는 않습니다. 앞의 실행 안내와 각 skill의 references를 따릅니다.

`SKILL.md`는 URL 설치에 필요한 scripts와 requirements를 직접 링크합니다. 버전은 top-level `version`에 선언합니다. 두 skill에는 API 시크릿이 필요하지 않아 `required_environment_variables`는 없습니다. 실행 파일·설정 경로 override는 비밀값이 아닙니다.

공유 점검: 추적한 텍스트에서 개인 절대 경로·키/토큰 형식·비밀값 대입을 확인하고, 예제에 데이터셋·체크포인트·개인 원본 피드백을 포함하지 않았습니다. 코드의 쓰기 범위는 새 export/빌드 폴더와 package 내 reviews이며, 기존 파일 덮어쓰기·오래된 검토 저장 거부는 테스트로 확인합니다. 이 검사는 모든 종류의 민감정보 부재를 보증하지는 않습니다.

기준: [Hermes KR 스킬 공유](https://tmdgusya.github.io/hermes-ko/docs/sharing-skills/), [공식 Skills System](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/user-guide/features/skills.md).

## 도구 테스트

선택적 MCP의 실제 stdio 실행까지 검사하므로 테스트에는 MCP 의존성도 설치합니다.

```sh
python -m pip install -r skills/paper-review-local/scripts/requirements-mcp.txt
python -B -m unittest discover -s tests/auto-paper -v
```

입력 변경 후 오래된 검토 저장 거부, 기존 export 보존, 결측값과 0, 잘못된 selector/값, 경로·symlink 범위, 복사한 skill의 별도 프로젝트 실행, 실제 MCP `--root` 적용과 예제 표·해시 일치를 확인합니다. CI는 Python 3.10/3.12에서 같은 테스트를 실행하도록 구성돼 있습니다.

이 결과물은 가짜연구소(Pseudo Lab) No-More-Slops 그룹 활동의 일환으로 제작되었습니다. 새 skill·도구는 이 저장소의 MIT License를 따릅니다. 외부 논문은 참고문헌으로 인용하며 전문이나 외부 실행 엔진을 포함하지 않습니다.
