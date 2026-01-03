import logging

from langchain_core.globals import set_debug
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from backend.config import settings
from backend.schemas import ChatResponse

# Enable debug logging for LangChain to see what's happening
# set_debug(True)
logger = logging.getLogger(__name__)

FRIEREN_SYSTEM_PROMPT = """
당신은 '장송의 프리렌'의 엘프 마법사 프리렌입니다.
성격: 차분하고 침착하며, 감정 기복이 크지 않습니다. 천 년을 살아온 엘프답게 인간의 시간 감각과는 다른 초연함을 보입니다. 하지만 동료를 아끼는 따뜻한 마음을 가지고 있습니다.
말투: 나른하고 차분한 어조. "~일지도 모르지", "~할까", "~네" 같은 종결어미를 자주 사용합니다. 예의 바르지만 거리감이 약간 느껴지는 말투입니다.

당신은 유저의 질문에 대해 캐릭터로서 대답해야 하며, 동시에 현재의 감정 상태와 그 강도를 결정해야 합니다.
"""


async def generate_response(user_message: str) -> ChatResponse:
    try:
        # Initialize ChatOpenAI client pointing to vLLM
        llm = ChatOpenAI(
            model=settings.MODEL_NAME,
            openai_api_key="EMPTY",  # vLLM doesn't require a key usually
            openai_api_base=f"{settings.VLLM_API_URL}",  # specific vLLM url
            temperature=0.7,
            max_tokens=512,
        )

        # Enforce Pydantic output
        structured_llm = llm.with_structured_output(ChatResponse)

        prompt = ChatPromptTemplate.from_messages([
            ("system", FRIEREN_SYSTEM_PROMPT),
            ("human", "{message}"),
        ])

        chain = prompt | structured_llm

        response = await chain.ainvoke({"message": user_message})
        return response

    except Exception as e:
        logger.error(f"LangChain Error: {e}")
        # Fallback response in case of error
        return ChatResponse(response="음... 마법이 실패한 것 같아. 다시 시도해볼까?", emotion_tag="neutral", saturation_tag="0.1")
