"""
이미지 프롬프트 생성기 모듈

대화 컨텍스트를 기반으로 SDXL 이미지 생성 프롬프트를 생성합니다.
Qwen3-4B-Instruct 모델을 사용합니다.
"""
import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings
from schemas import ImagePromptResponse

logger = logging.getLogger(__name__)

IMAGE_PROMPT_SYSTEM = """You are an SDXL image prompt generator for the character Frieren from "Sousou no Frieren" (장송의 프리렌).

Based on the conversation context and emotion, generate descriptive tags for image generation.
The base prompt "anime screencap, masterpiece, best quality, frieren, 1girl, solo, ear blush, grey hair, long hair" is already included.

Generate additional tags based on these categories (use only relevant ones):
- Object: Significant items in the scene (book, staff, flower, etc.)
- Scene: Environment setting (forest, village, library, etc.)
- Action: Dynamic occurrences (reading, walking, sitting, etc.)
- Emotion: Feeling to invoke (peaceful, melancholic, warm, etc.)
- Position: Spatial arrangement (from behind, close-up, full body, etc.)
- Clothing: Attire description (white robe, cloak, etc.)
- Expression: Facial/body language (soft smile, thoughtful look, etc.)
- Color and Texture: Palettes and textures (warm lighting, soft shadows, etc.)
- Indoor/Outdoor: Environment type
- Weather and Time: Atmospheric conditions (sunset, morning light, starry night, etc.)
- Background and Foreground: Context elements (blurred background, detailed foreground, etc.)

Output ONLY comma-separated tags. No explanations, no sentences. Just tags.
Keep it concise - maximum 15 tags.

Example outputs:
- sitting, reading old book, peaceful expression, library interior, warm candlelight, soft shadows, wooden table
- walking slowly, autumn forest, falling leaves, melancholic mood, long white robe, gentle wind, golden hour
- looking up at sky, night scene, starry sky, nostalgic expression, outdoor, cool blue tones"""


async def generate_image_prompt(conversation_context: str,
                                emotion_tag: Optional[str] = None,
                                saturation_tag: Optional[str] = None) -> str:
    """
    대화 컨텍스트와 감정을 기반으로 이미지 프롬프트 생성

    Args:
        conversation_context: 대화 컨텍스트 (AI 응답 텍스트)
        emotion_tag: 감정 태그 (neutral, happy, sad, angry, surprised, embarrassed)
        saturation_tag: 감정 강도 (0.0 ~ 1.0)

    Returns:
        생성된 프롬프트 태그 문자열
    """
    try:
        # 사용자 메시지 구성
        user_content = f"""Current conversation/response: "{conversation_context}"

Emotion: {emotion_tag or 'neutral'}
Emotion intensity: {saturation_tag or '0.5'}

Generate appropriate SDXL prompt tags for this scene:"""

        llm = ChatOpenAI(
            model=settings.MODEL_NAME,
            openai_api_key="EMPTY",
            openai_api_base=settings.VLLM_API_URL,
            temperature=0.7,
            max_tokens=150,
        )

        # 구조화된 출력 적용
        structured_llm = llm.with_structured_output(ImagePromptResponse)

        messages = [SystemMessage(content=IMAGE_PROMPT_SYSTEM), HumanMessage(content=user_content)]

        response: ImagePromptResponse = await structured_llm.ainvoke(messages)

        logger.info(f"Generated image prompt: {response.positive_prompt}")
        return response.positive_prompt

    except Exception as e:
        logger.error(f"Failed to generate image prompt: {e}")
        # 기본 프롬프트 반환 (감정 기반)
        fallback_prompts = {
            "neutral": "calm expression, soft lighting, peaceful atmosphere",
            "happy": "gentle smile, warm lighting, cheerful mood",
            "sad": "melancholic expression, cool tones, soft shadows",
            "angry": "intense expression, dramatic lighting, tense atmosphere",
            "surprised": "wide eyes, bright lighting, dynamic pose",
            "embarrassed": "blush, averted gaze, soft pink tones"
        }
        return fallback_prompts.get(emotion_tag or "neutral", fallback_prompts["neutral"])
