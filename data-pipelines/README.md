# Data Pipelines (Frieren Chatbot)

이 디렉토리는 캐릭터 챗봇(프리렌) 학습을 위한 데이터셋을 생성하는 파이프라인을 포함하고 있습니다.
기본적인 질문(Instruction)과 학습 포인트(Training Point)가 담긴 TSV 파일을 입력받아, OpenAI API(GPT-4o)를 이용해 캐릭터의 페르소나가 반영된 답변을 생성하고 JSON 형식으로 저장합니다.

## 기능

- **페르소나 적용**: '장송의 프리렌'의 프리렌 캐릭터 성격(냉철함, 마법 오타쿠, 시간 관념 등)을 반영한 답변 생성.
- **Thinking Process**: 답변 생성 전 `<think>` 블록을 통해 캐릭터의 내적 사고 과정을 포함.
- **데이터 포맷 변환**: CSV/TSV 입력을 LLM 학습에 사용되는 Alpaca 스타일의 JSON 포맷으로 변환.

## 필요 조건 (Prerequisites)

- Python 3.13 이상
- OpenAI API Key

## 설치 (Installation)

이 프로젝트는 `uv`를 사용하여 패키지를 관리하고 있습니다.

### uv 사용 시 (권장)
```bash
# 의존성 설치 및 동기화
uv sync
```

### pip 사용 시
```bash
pip install openai python-dotenv tqdm
```
*(참고: `tqdm`은 진행률 표시를 위해 선택적으로 사용됩니다.)*

## 환경 설정 (Configuration)

루트 디렉토리 또는 `data-pipelines/` 내에 `.env` 파일을 생성하고, OpenAI API 키를 설정해야 합니다.

```bash
# .env 파일
OPENAI_KEY=sk-proj-...
# 또는
OPENAI_API_KEY=sk-proj-...
```

## 사용 방법 (Usage)

1. `data/frieren_question.tsv` 파일이 존재하는지 확인합니다.
2. 아래 명령어로 파이프라인 스크립트를 실행합니다.

```bash
# uv 사용 시
uv run make-chat-dataset.py

# 가상환경 활성화 후 실행 시
source .venv/bin/activate
python make-chat-dataset.py
```

스크립트가 실행되면:
1. `data/frieren_question.tsv`를 읽어옵니다.
2. 각 질문에 대해 GPT-4o를 호출하여 프리렌 페르소나로 답변을 생성합니다.
3. 결과물은 `data/frieren_chat_dataset.json`에 저장됩니다.

## 데이터 구조 (Data Structure)

### 입력 데이터 (`data/frieren_question.tsv`)
탭으로 구분된 텍스트 파일로, 다음 컬럼을 포함해야 합니다:
- `no`: 번호
- `category`: 카테고리
- `instruction`: 사용자 질문 또는 지시사항
- `training_point`: 답변 시 중점을 둬야 할 캐릭터 특성이나 의도

### 출력 데이터 (`data/frieren_chat_dataset.json`)
Alpaca 데이터셋 형식을 따르는 JSON 배열입니다.
```json
[
  {
    "instruction": "사용자 질문",
    "input": "",
    "output": "<think>내적 사고 과정...</think>\n\n캐릭터 답변 (한국어)",
    "system": "시스템 프롬프트 (프리렌 페르소나 정의)"
  },
  ...
]
```
