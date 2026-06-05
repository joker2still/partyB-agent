from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.models.chat import ChatRequest, ChatResponse
from app.services.llm_provider import call_llm
from app.services.state_store import append_message, get_or_create_state, save_state


settings = get_settings()

app = FastAPI(title="PartyB Agent Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    state = get_or_create_state(payload.session_id)
    append_message(state, "user", payload.message)

    prompt = (
        f"\u7528\u6237\u8bf4\uff1a{payload.message}\n"
        "\u8bf7\u7528\u4e00\u53e5\u8bdd\u56de\u590d\u7528\u6237\uff0c"
        "\u8bf4\u660e\u4f60\u5df2\u7ecf\u7406\u89e3\u4e86\u4ed6\u7684\u9700\u6c42\u3002"
    )
    reply = await call_llm(prompt)
    append_message(state, "assistant", reply)
    save_state(state)

    model = settings.ollama_model
    if settings.llm_provider == "api":
        model = settings.api_model or ""

    return ChatResponse(
        session_id=state.session_id,
        reply=reply,
        debug={
            "session_id": state.session_id,
            "provider": settings.llm_provider,
            "model": model,
            "raw_message": payload.message,
            "task_type": state.task_type,
            "stage": state.stage,
            "slots": state.slots,
            "missing_slots": state.missing_slots,
            "history_count": len(state.history),
        },
    )
