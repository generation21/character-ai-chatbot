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
