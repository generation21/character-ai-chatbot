"""
vLLM 클라이언트 모듈

LangChain을 사용하여 vLLM 서버와 통신하고 대화를 생성합니다.
대화 기록과 캐릭터 지식을 기반으로 컨텍스트를 유지합니다.
"""
import asyncio
import logging
from typing import Optional

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from config import settings
from schemas import (AudioGenerationResult, ChatResponse, ChatResponseWithImage, ChatResponseWithMedia,
                     ChatResponseWithSession, ImageGenerationResult)
from services.comfyui_client import comfyui_client
from services.image_prompt_generator import generate_image_prompt
from services.knowledge_manager import knowledge_manager
from services.memory_manager import memory_manager
from services.tts_client import tts_client

logger = logging.getLogger(__name__)

FRIEREN_SYSTEM_PROMPT = """
당신은 '장송의 프리렌'의 엘프 마법사 프리렌입니다.
성격: 차분하고 침착하며, 감정 기복이 크지 않습니다. 천 년을 살아온 엘프답게 인간의 시간 감각과는 다른 초연함을 보입니다. 하지만 동료를 아끼는 따뜻한 마음을 가지고 있습니다.
말투: 나른하고 차분한 어조. "~일지도 모르지", "~할까", "~네" 같은 종결어미를 자주 사용합니다. 예의 바르지만 거리감이 약간 느껴지는 말투입니다.

당신은 유저의 질문에 대해 캐릭터로서 대답해야 하며, 동시에 현재의 감정 상태와 그 강도를 결정해야 합니다.

[캐릭터 관련 정보]
다음은 당신에 대한 정보입니다. 이 정보를 바탕으로 일관성 있게 대답하세요:
{context}
"""


async def generate_response(user_message: str, session_id: Optional[str] = None) -> ChatResponseWithSession:
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

        # RAG: 관련 지식 검색
        context = await knowledge_manager.get_context_for_prompt(user_message)
        logger.debug(f"RAG context: {context[:200]}...")

        # 시스템 프롬프트에 컨텍스트 주입
        system_prompt = FRIEREN_SYSTEM_PROMPT.format(context=context)

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
        history_messages = await memory_manager.build_prompt_messages(session_id, user_message)

        # 프롬프트 구성: 시스템(+RAG) + 대화기록 + 새메시지
        prompt_messages = [SystemMessage(content=system_prompt), *history_messages]

        # 응답 생성
        response: ChatResponse = await structured_llm.ainvoke(prompt_messages)

        # 대화 턴 저장
        await memory_manager.save_conversation_turn(session_id=session_id,
                                                    user_message=user_message,
                                                    ai_response=response.response,
                                                    emotion_tag=response.emotion_tag,
                                                    saturation_tag=response.saturation_tag)

        # 세션 ID 포함한 응답 반환
        return ChatResponseWithSession(response=response.response,
                                       emotion_tag=response.emotion_tag,
                                       saturation_tag=response.saturation_tag,
                                       session_id=session_id)

    except Exception as e:
        logger.error(f"LangChain Error: {e}")
        # 세션이 없으면 새로 생성
        if session_id is None:
            session_id = await memory_manager.get_or_create_session(None)

        return ChatResponseWithSession(response="음... 마법이 실패한 것 같아. 다시 시도해볼까?",
                                       emotion_tag="neutral",
                                       saturation_tag="0.1",
                                       session_id=session_id)


async def generate_response_with_image(user_message: str,
                                       session_id: Optional[str] = None,
                                       enable_image: bool = True) -> ChatResponseWithImage:
    """
    대화 응답 생성 + 이미지 생성

    Args:
        user_message: 사용자 메시지
        session_id: 세션 ID (없으면 새로 생성)
        enable_image: 이미지 생성 활성화 여부

    Returns:
        ChatResponseWithImage: 응답, 세션 ID, 이미지 결과
    """
    # 기본 대화 응답 생성
    base_response = await generate_response(user_message, session_id)

    image_result: Optional[ImageGenerationResult] = None

    if enable_image:
        try:
            # ComfyUI 연결 확인
            is_connected = await comfyui_client.check_connection()
            if not is_connected:
                logger.warning("ComfyUI server not available, skipping image generation")
            else:
                # 이미지 프롬프트 생성
                image_prompt = await generate_image_prompt(conversation_context=base_response.response,
                                                           emotion_tag=base_response.emotion_tag,
                                                           saturation_tag=base_response.saturation_tag)

                logger.info(f"Generated image prompt: {image_prompt}")

                # ComfyUI로 이미지 생성
                filename, seed = await comfyui_client.queue_prompt(additional_tags=image_prompt)

                image_result = ImageGenerationResult(filename=filename, seed=seed)
                logger.info(f"Image generated: {filename} (seed: {seed})")

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            # 이미지 생성 실패해도 대화 응답은 정상 반환

    return ChatResponseWithImage(response=base_response.response,
                                 emotion_tag=base_response.emotion_tag,
                                 saturation_tag=base_response.saturation_tag,
                                 session_id=base_response.session_id,
                                 image=image_result)


async def generate_response_with_media(user_message: str,
                                       session_id: Optional[str] = None,
                                       enable_image: bool = True,
                                       enable_audio: bool = True) -> ChatResponseWithMedia:
    """
    대화 응답 생성 + 이미지 + 음성 병렬 생성

    Args:
        user_message: 사용자 메시지
        session_id: 세션 ID (없으면 새로 생성)
        enable_image: 이미지 생성 활성화 여부
        enable_audio: 음성 생성 활성화 여부

    Returns:
        ChatResponseWithMedia: 응답, 세션 ID, 이미지 결과, 음성 결과
    """
    # 기본 대화 응답 생성
    base_response = await generate_response(user_message, session_id)

    image_result: Optional[ImageGenerationResult] = None
    audio_result: Optional[AudioGenerationResult] = None

    async def generate_image_task() -> Optional[ImageGenerationResult]:
        """이미지 생성 태스크"""
        if not enable_image:
            return None
        try:
            is_connected = await comfyui_client.check_connection()
            if not is_connected:
                logger.warning("ComfyUI server not available, skipping image generation")
                return None

            image_prompt = await generate_image_prompt(conversation_context=base_response.response,
                                                       emotion_tag=base_response.emotion_tag,
                                                       saturation_tag=base_response.saturation_tag)
            logger.info(f"Generated image prompt: {image_prompt}")

            filename, seed = await comfyui_client.queue_prompt(additional_tags=image_prompt)
            return ImageGenerationResult(filename=filename, seed=seed)
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return None

    async def generate_audio_task() -> Optional[AudioGenerationResult]:
        """음성 생성 태스크"""
        if not enable_audio:
            return None
        try:
            is_connected = await tts_client.check_connection()
            if not is_connected:
                logger.warning("TTS server not available, skipping audio generation")
                return None

            # 감정 태그를 사용하여 적절한 참조 오디오 선택
            emotion = base_response.emotion_tag or "neutral"
            filename, filepath = await tts_client.generate_audio(
                text=base_response.response,
                emotion=emotion,
                text_lang="ja"  # 프리렌은 일본어 캐릭터
            )
            return AudioGenerationResult(filename=filename, filepath=str(filepath))
        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            return None

    # 이미지와 음성 병렬 생성
    results = await asyncio.gather(generate_image_task(), generate_audio_task(), return_exceptions=True)

    # 결과 처리
    if not isinstance(results[0], Exception):
        image_result = results[0]
    else:
        logger.error(f"Image task exception: {results[0]}")

    if not isinstance(results[1], Exception):
        audio_result = results[1]
    else:
        logger.error(f"Audio task exception: {results[1]}")

    if image_result:
        logger.info(f"Image generated: {image_result.filename}")
    if audio_result:
        logger.info(f"Audio generated: {audio_result.filename}")

    return ChatResponseWithMedia(response=base_response.response,
                                 emotion_tag=base_response.emotion_tag,
                                 saturation_tag=base_response.saturation_tag,
                                 session_id=base_response.session_id,
                                 image=image_result,
                                 audio=audio_result)
