from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.agents.workflow_controller import handle_chat_turn
from app.models.chat import ChatRequest, ChatResponse, ResumeExportRequest
from app.services.docx_exporter import export_structured_resume_docx
from app.services.state_store import append_message, get_or_create_state, get_state, save_state


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


@app.post("/export/resume")
def export_resume(payload: ResumeExportRequest) -> FileResponse:
    state = get_state(payload.session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="session not found")

    markdown_text = state.slots.get("resume_draft")
    if not isinstance(markdown_text, str) or not markdown_text.strip():
        raise HTTPException(status_code=400, detail="resume draft not found")

    file_path = export_structured_resume_docx(state.slots, state.session_id)
    filename = Path(file_path).name
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
