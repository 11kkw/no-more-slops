# Auto-Paper: 저장된 결과에서 원고와 검토 기록까지

완료된 실험 결과를 논문으로 정리하거나, 수정한 원고를 다시 검토할 때 사용하는 두 스킬입니다.
각 스킬 폴더를 에이전트의 스킬 디렉터리에 복사하거나, 저장소의 `SKILL.md`를 읽도록 요청합니다.
Hermes 설치와 의존성 안내는 [공유·설치 안내](../examples/auto-paper/README.md#hermes에서-공유설치)에 있습니다.

## 사용 요청 예시

> experiment-to-paper 스킬을 사용해 저장된 결과 JSON을 확인해줘.
> 지표·단위·분모와 원본 파일을 근거표로 연결하고, 확인된 결과로 원고를 작성해줘.
> 결측값은 0으로 바꾸지 말고, 추가 실험이 필요한 주장은 미확인으로 표시해줘.

> paper-review-local 스킬로 현재 원고와 PDF를 검토해줘.
> 구조·인용 키·그림·PDF와 ChkTeX 결과를 확인하고, 각 지적에 위치와 근거를 남겨줘.
> 원고를 수정했다면 새 입력 해시와 review ID로 다시 검토해줘.

논문 형식과 쪽수는 프로젝트 요구에 맞춥니다. 스킬 자체에는 특정 학회나 4쪽 제한이 없습니다.

## 작업 흐름

1. [experiment-to-paper](../skills/experiment-to-paper/SKILL.md)로 완료된 결과와 연구 질문을 확인합니다.
2. JSON selector로 수치·결측값·입력 해시를 추출하고 주장과 근거를 연결합니다.
3. 요청한 형식으로 원고와 PDF를 작성합니다.
4. [paper-review-local](../skills/paper-review-local/SKILL.md)로 정적 검사를 실행하고 내용과 실제 PDF 화면을 검토합니다.
5. 수정한 입력에는 새 검토 기록을 남기고, 남은 지적과 산출물을 함께 전달합니다.

## 다른 프로젝트에서 먼저 실행하기

Python 3.10 이상에서 가상 결과를 사용하는 다음 명령을 실행합니다. 추가 패키지는 필요 없습니다.

```sh
python3 skills/experiment-to-paper/scripts/export_evidence.py \
  --root skills/experiment-to-paper/assets/example --spec export-spec.json \
  --out /tmp/paper-evidence-demo-new
```

3개 측정값과 입력 해시·표를 출력합니다. 실제 프로젝트에서는
[설정 규약](../skills/experiment-to-paper/references/evidence-schema.md)에 따라 입력 파일과 JSON pointer를 바꿉니다.
로컬 검토의 Python 의존성, ChkTeX, 선택적 MCP는
[실행 안내](../skills/paper-review-local/references/local-tools.md)를 따릅니다.

## 논문 예제와 기여 범위

[실제 DCSM 논문 PDF](../examples/auto-paper/dcsm-spatial/shortpaper.pdf)와 문서 소스,
측정값, 주장-근거표, 재빌드 명령을 함께 제공합니다.
[예제 안내와 검증 기록](../examples/auto-paper/README.md)은 도구 실행과 논문의 미확인 사항을 구분합니다.
저자·소속과 독립 검토가 아직 완료되지 않은 연구 초안입니다.

JSON의 지정한 값을 근거표로 추출하고, 원고 입력이 바뀌면 이전 검토를 재사용하지 않도록 합니다.
실험 집계 재계산, 통계적 타당성, 독창성, 투고 승인에 대한 자동 판정은 제공하지 않습니다.
