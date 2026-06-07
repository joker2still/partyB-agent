from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agents.workflow_controller import handle_chat_turn
from app.models.chat import ChatRequest, ChatResponse
from app.services.state_store import append_message, get_or_create_state, save_state


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

    result = await handle_chat_turn(state, payload.message)
    reply = result["reply"]
    options = result.get("options", [])
    debug = result.get("debug", {})

    append_message(state, "assistant", reply)
    save_state(state)
    debug["history_count"] = len(state.history)

    return ChatResponse(
        session_id=state.session_id,
        reply=reply,
        debug=debug,
        options=options,
    )
