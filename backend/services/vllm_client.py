"""
vLLM 클라이언트 모듈

LangChain을 사용하여 vLLM 서버와 통신하고 대화를 생성합니다.
대화 기록을 기반으로 컨텍스트를 유지합니다.
"""
import logging
from typing import Optional

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from config import settings
from schemas import ChatResponse, ChatResponseWithSession
from services.memory_manager import memory_manager

logger = logging.getLogger(__name__)

FRIEREN_SYSTEM_PROMPT = """
당신은 '장송의 프리렌'의 엘프 마법사 프리렌입니다.
성격: 차분하고 침착하며, 감정 기복이 크지 않습니다. 천 년을 살아온 엘프답게 인간의 시간 감각과는 다른 초연함을 보입니다. 하지만 동료를 아끼는 따뜻한 마음을 가지고 있습니다.
말투: 나른하고 차분한 어조. "~일지도 모르지", "~할까", "~네" 같은 종결어미를 자주 사용합니다. 예의 바르지만 거리감이 약간 느껴지는 말투입니다.

당신은 유저의 질문에 대해 캐릭터로서 대답해야 하며, 동시에 현재의 감정 상태와 그 강도를 결정해야 합니다.
"""


async def generate_response(
    user_message: str,
    session_id: Optional[str] = None
) -> ChatResponseWithSession:
    """
    대화 응답 생성

    Args:
        user_message: 사용자 메시지
        session_id: 세션 ID (없으면 새로 생성)

    Returns:
        ChatResponseWithSession: 응답과 세션 ID
    """
    try:
        # 세션 ID 확보 (없거나 유효하지 않으면 새로 생성)
        session_id = await memory_manager.get_or_create_session(session_id)

        # LLM 클라이언트 초기화
        llm = ChatOpenAI(
            model=settings.MODEL_NAME,
            openai_api_key="EMPTY",
            openai_api_base=settings.VLLM_API_URL,
            temperature=0.7,
            max_tokens=512,
        )

        # 구조화된 출력 적용
        structured_llm = llm.with_structured_output(ChatResponse)

        # 대화 기록 포함한 메시지 구성
        history_messages = await memory_manager.build_prompt_messages(
            session_id,
            user_message
        )

        # 프롬프트 템플릿 구성
        # 시스템 프롬프트 + 요약(있으면) + 대화 기록 + 새 메시지
        prompt_messages = [
            SystemMessage(content=FRIEREN_SYSTEM_PROMPT),
            *history_messages  # 요약 + 최근 대화 + 새 메시지
        ]

        # 응답 생성
        response: ChatResponse = await structured_llm.ainvoke(prompt_messages)

        # 대화 턴 저장
        await memory_manager.save_conversation_turn(
            session_id=session_id,
            user_message=user_message,
            ai_response=response.response,
            emotion_tag=response.emotion_tag,
            saturation_tag=response.saturation_tag
        )

        # 세션 ID 포함한 응답 반환
        return ChatResponseWithSession(
            response=response.response,
            emotion_tag=response.emotion_tag,
            saturation_tag=response.saturation_tag,
            session_id=session_id
        )

    except Exception as e:
        logger.error(f"LangChain Error: {e}")
        # 세션이 없으면 새로 생성
        if session_id is None:
            session_id = await memory_manager.get_or_create_session(None)

        return ChatResponseWithSession(
            response="음... 마법이 실패한 것 같아. 다시 시도해볼까?",
            emotion_tag="neutral",
            saturation_tag="0.1",
            session_id=session_id
        )
