from pydantic import BaseModel, Field


class AgentMessage(BaseModel):
    role: str
    content: str


class AgentState(BaseModel):
    session_id: str
    task_type: str | None = None
    stage: str = "start"
    slots: dict = Field(default_factory=dict)
    missing_slots: list[str] = Field(default_factory=list)
    confirmed: dict = Field(default_factory=dict)
    history: list[AgentMessage] = Field(default_factory=list)
