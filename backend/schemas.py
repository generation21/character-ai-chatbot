from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str = Field(description="The spoken text of the character, in Frieren's tone.")
    emotion_tag: Optional[str] = Field(
        None,
        description="Current emotion of the character. One of: [neutral, happy, sad, angry, surprised, embarrassed]")
    saturation_tag: Optional[str] = Field(None, description="The intensity of the emotion, from 0.0 to 1.0.")
