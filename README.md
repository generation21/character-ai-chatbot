# Character Chatbot AI (Eternal Journey)

'장송의 프리렌'의 프리렌 캐릭터 페르소나를 가진 AI 챗봇 프로젝트입니다.
FastAPI 기반의 추론 백엔드와 학습 데이터 생성을 위한 데이터 파이프라인으로 구성되어 있습니다.

## 📂 프로젝트 구조

이 프로젝트는 크게 두 가지 주요 컴포넌트로 나뉩니다:

- **`backend/`**: 실시간 채팅 서비스를 위한 FastAPI 서버입니다. vLLM (Qwen3-4B-Instruct) 모델을 사용하여 프리렌의 말투와 성격을 반영한 답변을 제공합니다.
- **`data-pipelines/`**: 챗봇 학습 및 평가를 위한 데이터셋 생성 도구입니다. GPT-4o를 활용하여 프리렌의 사고 과정(Thinking Process)이 포함된 고품질 대화 데이터를 생성합니다.

## 🚀 시작하기 (Getting Started)

### 사전 요구 사항 (Prerequisites)

- **Python 3.13+**
- **uv** (권장 패키지 매니저)
- **vLLM** (로컬 추론 시 필요, NVIDIA GPU 권장)

### 1. 백엔드 실행 (Backend)

백엔드 서버는 `backend/` 디렉토리 내의 코드를 사용합니다.

```bash
# 의존성 설치
uv sync

# FastAPI 서버 실행
uv run uvicorn backend.main:app --reload
```
서버가 실행되면 `http://localhost:8000/docs`에서 API 문서를 확인할 수 있습니다.

### 2. 데이터 파이프라인 (Data Pipelines)

데이터 생성 스크립트는 `data-pipelines/` 디렉토리에 있습니다. 자세한 내용은 [`data-pipelines/README.md`](data-pipelines/README.md)를 참고하세요.

# vLLM 서버 실행
```bash
python -m vllm.entrypoints.openai.api_server \
    --model ./models/llm/Qwen3-4B-Instruct-2507 \
    --enable-lora \
    --lora-modules frieren-lora=./models/llm/frieren-lora-adapter \
    --max-lora-rank 64 \
    --gpu-memory-utilization 0.5 \
    --max-model-len 16384 \
    --port 8000
```

```bash
# 데이터 생성 스크립트 실행
cd data-pipelines
uv run make-chat-dataset.py
```

## 🛠️ 기술 스택 (Tech Stack)

- **Backend**: FastAPI, Uvicorn
- **AI Model**: vLLM, Qwen/Qwen3-4B-Instruct
- **Orchestration**: LangChain
- **Data Generation**: OpenAI API (GPT-4o)
- **Package Manager**: uv

## 📝 라이선스

이 프로젝트는 개인 학습 및 연구 목적으로 개발되었습니다.
