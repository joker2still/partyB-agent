from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str | None = Field(default=None)
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    debug: dict[str, Any]
