"""
Frieren Chatbot API

FastAPI 기반 백엔드 서버
- 대화 생성 API (세션 기반)
- 세션 관리 API
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import (ChatRequest, ChatResponseWithImage, ChatResponseWithSession, SessionInfo, SessionListResponse,
                     SessionMessagesResponse)
from services.session_manager import session_manager
from services.vllm_client import (generate_response, generate_response_with_image)

app = FastAPI(title="Frieren Chatbot API", description="프리렌 캐릭터 AI 챗봇 API - 대화 기록 관리 포함", version="0.2.0")

# CORS Configuration
origins = ["http://localhost:5173", "http://localhost:3000", "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Chat Endpoints
# ============================================================


@app.post("/chat", response_model=ChatResponseWithSession)
async def chat(request: ChatRequest):
    """
    대화 생성 엔드포인트

    - session_id가 없으면 새 세션 자동 생성
    - session_id가 있으면 해당 세션의 대화 기록을 사용
    """
    response = await generate_response(request.message, request.session_id)
    return response


@app.post("/chat-with-image", response_model=ChatResponseWithImage)
async def chat_with_image(request: ChatRequest):
    """
    대화 생성 + 이미지 생성 엔드포인트

    - session_id가 없으면 새 세션 자동 생성
    - 대화 응답과 함께 프리렌 이미지 생성
    - ComfyUI 서버가 없으면 이미지 없이 대화만 반환
    """
    response = await generate_response_with_image(request.message, request.session_id)
    return response


# ============================================================
# Session Management Endpoints
# ============================================================


@app.get("/sessions", response_model=SessionListResponse)
async def list_sessions():
    """모든 세션 목록 조회 (최근순)"""
    sessions = await session_manager.list_sessions()
    return SessionListResponse(sessions=sessions)


@app.post("/sessions", response_model=SessionInfo)
async def create_session():
    """새 세션 생성"""
    session_id = await session_manager.create_session()
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=500, detail="Failed to create session")
    return session


@app.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """특정 세션 정보 조회"""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """세션 삭제 (관련 메시지도 함께 삭제)"""
    deleted = await session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted successfully", "session_id": session_id}


@app.get("/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
async def get_session_messages(session_id: str, limit: int = 50):
    """
    특정 세션의 대화 기록 조회

    Args:
        session_id: 세션 ID
        limit: 조회할 메시지 개수 (기본 50)
    """
    # 세션 존재 확인
    exists = await session_manager.session_exists(session_id)
    if not exists:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = await session_manager.get_messages(session_id, limit=limit)
    return SessionMessagesResponse(session_id=session_id, messages=messages)


# ============================================================
# Knowledge Management Endpoints
# ============================================================


@app.post("/knowledge/reindex")
async def reindex_knowledge(force: bool = False):
    """
    PDF 지식 베이스 재인덱싱

    Args:
        force: 기존 데이터 삭제 후 강제 재인덱싱
    """
    from config import settings
    from services.knowledge_manager import knowledge_manager

    try:
        count = knowledge_manager.index_pdf(settings.PDF_PATH, force=force)
        return {"message": "Reindexing complete", "chunks_indexed": count, "pdf_path": settings.PDF_PATH}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge/stats")
async def get_knowledge_stats():
    """ChromaDB 컬렉션 상태 조회"""
    from services.knowledge_manager import knowledge_manager
    return knowledge_manager.get_collection_stats()


@app.get("/knowledge/search")
async def search_knowledge(query: str, top_k: int = 3):
    """
    지식 베이스 검색 테스트

    Args:
        query: 검색 쿼리
        top_k: 반환할 문서 수
    """
    from services.knowledge_manager import knowledge_manager

    results = await knowledge_manager.search(query, top_k)
    return {"query": query, "results": results, "count": len(results)}


# ============================================================
# Health Check
# ============================================================


@app.get("/")
async def root():
    """헬스 체크 엔드포인트"""
    return {"message": "Eternal Journey AI Backend is running", "version": "0.3.0"}
