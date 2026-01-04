"""
대화 기억 관리 모듈

최근 메시지와 요약을 조합하여 대화 컨텍스트를 관리합니다.
"""
import logging
from typing import List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings
from schemas import MessageInfo
from services.session_manager import session_manager

logger = logging.getLogger(__name__)

# 요약을 위한 시스템 프롬프트
SUMMARIZE_SYSTEM_PROMPT = """
당신은 대화 요약 전문가입니다. 주어진 대화 내용을 간결하게 요약해주세요.
요약 시 다음 사항을 포함하세요:
- 대화의 주요 주제
- 중요한 정보나 사실
- 사용자가 언급한 개인적인 정보 (이름, 선호도 등)
- 대화의 감정적 흐름

요약은 한국어로 작성하고, 200자 이내로 간결하게 작성하세요.
"""


class MemoryManager:
    """대화 기억 관리 - 최근 메시지 + 요약"""

    def __init__(self):
        self.max_recent_messages = settings.MAX_RECENT_MESSAGES
        self.summarize_threshold = settings.SUMMARIZE_THRESHOLD

    async def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """세션 ID가 없거나 유효하지 않으면 새로 생성"""
        if session_id:
            exists = await session_manager.session_exists(session_id)
            if exists:
                return session_id
            logger.warning(f"Session {session_id} not found, creating new session")

        return await session_manager.create_session()

    async def should_summarize(self, session_id: str) -> bool:
        """요약이 필요한지 확인"""
        message_count = await session_manager.get_message_count(session_id)
        return message_count >= self.summarize_threshold

    async def summarize_old_messages(self, session_id: str) -> Optional[str]:
        """오래된 메시지를 요약하여 저장"""
        # 기존 요약 가져오기
        existing_summary = await session_manager.get_summary(session_id)

        # 요약할 오래된 메시지 가져오기
        old_messages = await session_manager.get_old_messages_for_summary(
            session_id,
            keep_recent=self.max_recent_messages
        )

        if not old_messages:
            return existing_summary

        # 요약을 위한 대화 텍스트 구성
        conversation_text = self._format_messages_for_summary(old_messages, existing_summary)

        try:
            # vLLM을 사용하여 요약 생성
            llm = ChatOpenAI(
                model=settings.MODEL_NAME,
                openai_api_key="EMPTY",
                openai_api_base=settings.VLLM_API_URL,
                temperature=0.3,
                max_tokens=256,
            )

            messages = [
                SystemMessage(content=SUMMARIZE_SYSTEM_PROMPT),
                HumanMessage(content=f"다음 대화를 요약해주세요:\n\n{conversation_text}")
            ]

            response = await llm.ainvoke(messages)
            new_summary = response.content.strip()

            # 요약 저장
            await session_manager.update_summary(session_id, new_summary)
            logger.info(f"Generated new summary for session {session_id}")

            return new_summary

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return existing_summary

    def _format_messages_for_summary(
        self,
        messages: List[MessageInfo],
        existing_summary: Optional[str] = None
    ) -> str:
        """메시지를 요약용 텍스트로 포맷팅"""
        parts = []

        if existing_summary:
            parts.append(f"[이전 요약]\n{existing_summary}\n")

        parts.append("[새로운 대화]")
        for msg in messages:
            role_name = "사용자" if msg.role == "human" else "프리렌"
            parts.append(f"{role_name}: {msg.content}")

        return "\n".join(parts)

    async def build_prompt_messages(
        self,
        session_id: str,
        new_message: str
    ) -> List[BaseMessage]:
        """프롬프트에 포함할 메시지 목록 구성"""
        messages: List[BaseMessage] = []

        # 1. 요약이 필요하면 먼저 요약 수행
        if await self.should_summarize(session_id):
            await self.summarize_old_messages(session_id)

        # 2. 세션 요약 가져오기
        summary = await session_manager.get_summary(session_id)

        # 3. 최근 메시지 가져오기
        recent_messages = await session_manager.get_messages(
            session_id,
            limit=self.max_recent_messages
        )

        # 4. 요약이 있으면 시스템 메시지로 추가
        if summary:
            messages.append(SystemMessage(
                content=f"[이전 대화 요약]\n{summary}"
            ))

        # 5. 최근 메시지 추가
        for msg in recent_messages:
            if msg.role == "human":
                messages.append(HumanMessage(content=msg.content))
            else:
                messages.append(AIMessage(content=msg.content))

        # 6. 새 메시지 추가
        messages.append(HumanMessage(content=new_message))

        return messages

    async def save_conversation_turn(
        self,
        session_id: str,
        user_message: str,
        ai_response: str,
        emotion_tag: Optional[str] = None,
        saturation_tag: Optional[str] = None
    ) -> None:
        """대화 턴 저장 (사용자 메시지 + AI 응답)"""
        # 사용자 메시지 저장
        await session_manager.add_message(
            session_id=session_id,
            role="human",
            content=user_message
        )

        # AI 응답 저장
        await session_manager.add_message(
            session_id=session_id,
            role="ai",
            content=ai_response,
            emotion_tag=emotion_tag,
            saturation_tag=saturation_tag
        )


# 싱글톤 인스턴스
memory_manager = MemoryManager()
