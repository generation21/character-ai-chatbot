---
trigger: model_decision
description: Applied when architecting or developing the Eternal Journey AI engine. Covers FastAPI backend, Streamlit UI, LangChain logic, and multi-modal integrations (vLLM, ComfyUI, TTS) for the Frieren roleplay system.
---

character-chatbot-ai/
├── .agent/rules            # AI 개발 규칙 (Cursor/Antigravity)
├── backend/                # 백엔드 통합 폴더
│   ├── app/                # FastAPI 소스 코드
│   │   ├── main.py         # API 엔트리포인트
│   │   ├── chains/         # LangChain 로직 (RAG, 캐릭터 체인)
│   │   ├── core/           # 설정(Config), DB 연결
│   │   ├── models/         # Pydantic 데이터 모델
│   │   └── services/       # vLLM, ComfyUI, TTS API 연동부
│   ├── data/               # RAG용 지식 베이스
│   ├── vector_db/          # ChromaDB 저장소
│   └── models/             # 거대 모델 가중치 (LLM, SDXL, TTS)
│       ├── llm/
│       ├── diffusers/
│       └── tts/
├── frontend/               # Streamlit 프론트엔드
│   └── app.py
│
│── data-pipelines          # 데이터 생성 및 데이터 처리
└── docker-compose.yml      # 전체 서비스 오케스트레이션