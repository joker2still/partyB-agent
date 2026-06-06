from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agents.task_router import detect_task_type
from app.core.config import get_settings
from app.models.chat import ChatRequest, ChatResponse
from app.services.llm_provider import call_llm
from app.services.state_store import append_message, get_or_create_state, save_state
from app.services.template_loader import load_task_template


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
    task_router_result: dict | None = None
    task_template_loaded = False
    required_slots: list[str] = []

    if state.stage == "start":
        task_router_result = await detect_task_type(payload.message)

        if task_router_result["task_type"] == "resume":
            state.task_type = "resume"
            state.stage = "collect_info"
            template = load_task_template("resume")
            task_template_loaded = bool(template)
            required_slots = template.get("required_slots", []) if template else []

            slot_questions = template.get("slot_questions", {}) if template else {}
            first_round_slots = required_slots[:3]
            question_lines = []
            for index, slot_name in enumerate(first_round_slots, start=1):
                question = slot_questions.get(slot_name)
                if isinstance(question, str) and question.strip():
                    question_lines.append(f"{index}. {question}")

            if question_lines:
                reply = (
                    "\u597d\u7684\uff0c\u6211\u7406\u89e3\u4f60\u60f3\u5199\u7b80\u5386\u3002"
                    "\u4e3a\u4e86\u5148\u642d\u597d\u7b80\u5386\u9aa8\u67b6\uff0c"
                    "\u6211\u9700\u8981\u4e86\u89e3\u4e0b\u9762\u51e0\u9879\uff1a\n\n"
                    + "\n".join(question_lines)
                    + "\n\n"
                    + "\u4f60\u53ef\u4ee5\u4e00\u6b21\u6027\u56de\u7b54\uff0c"
                    "\u4e5f\u53ef\u4ee5\u5148\u56de\u7b54\u5176\u4e2d\u4e00\u90e8\u5206\u3002"
                )
            else:
                reply = (
                    "\u597d\u7684\uff0c\u6211\u7406\u89e3\u4f60\u60f3\u5199\u7b80\u5386\u3002"
                    "\u63a5\u4e0b\u6765\u6211\u4f1a\u5148\u6536\u96c6\u4f60\u7684\u57fa\u672c\u4fe1\u606f\uff0c"
                    "\u4f60\u53ef\u4ee5\u5148\u544a\u8bc9\u6211\u6c42\u804c\u65b9\u5411\u3001"
                    "\u6559\u80b2\u7ecf\u5386\u548c\u5de5\u4f5c/\u9879\u76ee\u7ecf\u5386\u3002"
                )
        else:
            state.task_type = "unknown"
            state.stage = "clarify_task"
            reply = (
                "\u6211\u8fd8\u4e0d\u786e\u5b9a\u4f60\u60f3\u5b8c\u6210\u54ea\u7c7b\u4efb\u52a1\u3002"
                "\u4f60\u53ef\u4ee5\u7b80\u5355\u8bf4\u4e00\u4e0b\u4f60\u60f3\u8ba9\u6211\u5e2e\u4f60"
                "\u4ea7\u51fa\u4ec0\u4e48\uff0c\u6bd4\u5982\u7b80\u5386\u3001PPT\u3001\u65b9\u6848\u3001"
                "\u6587\u6848\u6216\u4ee3\u7801\u9879\u76ee\u3002"
            )
    else:
        if state.task_type:
            template = load_task_template(state.task_type)
            task_template_loaded = bool(template)
            required_slots = template.get("required_slots", []) if template else []

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
            "task_router_result": task_router_result,
            "task_template_loaded": task_template_loaded,
            "required_slots": required_slots,
        },
    )
