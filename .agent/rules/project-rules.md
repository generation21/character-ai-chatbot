---
trigger: always_on
---

# Eternal Journey AI Project Rules

## Project Context
- 이 프로젝트는 '장송의 프리렌' 캐릭터 AI 엔진입니다.
- 주요 기술 스택: Streamlit(Frontend), FastAPI (Backend), LangChain (Orchestration), vLLM (Qwen3-4B-Instruct), ComfyUI API (Image), GPT-SoVITS (Audio).
- 하드웨어: 단일 NVIDIA RTX 4090 (24GB VRAM).

## Coding Standards
- 모든 Backend 로직은 `async/await`를 사용하는 비동기 방식으로 작성할 것.
- Python Type Hinting을 엄격하게 적용할 것 (Pydantic 모델 활용).
- 모델 추론 서버(vLLM, ComfyUI, TTS)와의 통신은 `httpx` 비동기 클라이언트를 사용할 것.

## VRAM Management (Critical)
- 메모리 누수 방지를 위해 사용하지 않는 텐서는 즉시 정리하는 로직을 포함할 것.

## Character Persona: Frieren
- 대화 생성 프롬프트 작성 시 프리렌의 말투(나른함, 초연함, ~일지도 모르지)를 유지할 것.
- 답변 결과에 항상 [Emotion], [Saturation] 태그를 포함하여 이미지 생성이 연동되도록 할 것.