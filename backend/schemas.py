from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # None이면 새 세션 자동 생성


class ChatResponse(BaseModel):
    response: str = Field(description="The spoken text of the character, in Frieren's tone.")
    emotion_tag: Optional[str] = Field(
        None,
        description="Current emotion of the character. One of: [neutral, happy, sad, angry, surprised, embarrassed]")
    saturation_tag: Optional[str] = Field(None, description="The intensity of the emotion, from 0.0 to 1.0.")


class ChatResponseWithSession(ChatResponse):
    """ChatResponse with session_id included"""
    session_id: str = Field(description="The session ID for this conversation")


class ImagePromptResponse(BaseModel):
    """SDXL 프롬프트 생성 결과"""
    positive_prompt: str = Field(description="Generated SDXL positive prompt tags")


class ImageGenerationResult(BaseModel):
    """이미지 생성 결과"""
    filename: str = Field(description="Generated image filename")
    seed: int = Field(description="Seed used for generation")


class AudioGenerationResult(BaseModel):
    """TTS 음성 생성 결과"""
    filename: str = Field(description="Generated audio filename")
    filepath: str = Field(description="Path to the generated audio file")


class ChatResponseWithImage(ChatResponseWithSession):
    """이미지가 포함된 대화 응답"""
    image: Optional[ImageGenerationResult] = Field(
        None, description="Generated image result, None if image generation failed or disabled")


class ChatResponseWithMedia(ChatResponseWithSession):
    """이미지 + 음성이 포함된 대화 응답"""
    image: Optional[ImageGenerationResult] = Field(None, description="Generated image result")
    audio: Optional[AudioGenerationResult] = Field(None, description="Generated audio result")


class SessionInfo(BaseModel):
    """Session information for listing and management"""
    session_id: str
    created_at: datetime
    last_message_at: Optional[datetime] = None
    message_count: int = 0
    summary: Optional[str] = None


class SessionListResponse(BaseModel):
    """Response for session list endpoint"""
    sessions: List[SessionInfo]


class MessageInfo(BaseModel):
    """Message information for history endpoint"""
    role: str  # 'human' or 'ai'
    content: str
    emotion_tag: Optional[str] = None
    saturation_tag: Optional[str] = None
    created_at: datetime


class SessionMessagesResponse(BaseModel):
    """Response for session messages endpoint"""
    session_id: str
    messages: List[MessageInfo]
