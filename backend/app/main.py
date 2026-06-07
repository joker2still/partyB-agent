from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agents.draft_generator import generate_resume_draft
from app.agents.resume_reviser import revise_resume_draft
from app.agents.slot_checker import check_missing_slots
from app.agents.slot_extractor import extract_slots
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

FINAL_CONFIRM_WORDS = [
    "\u786e\u8ba4\u6700\u7ec8\u7248",
    "\u6700\u7ec8\u7248",
    "\u53ef\u4ee5\u4e86",
    "\u6ca1\u95ee\u9898",
    "\u786e\u8ba4",
]


def _build_question_lines(slot_names: list[str], slot_questions: dict) -> list[str]:
    question_lines: list[str] = []
    for index, slot_name in enumerate(slot_names, start=1):
        question = slot_questions.get(slot_name)
        if isinstance(question, str) and question.strip():
            question_lines.append(f"{index}. {question}")
    return question_lines


def _normalize_text(text: str) -> str:
    return text.strip().casefold()


def _match_style_option(message: str, style_options: list[dict]) -> dict | None:
    normalized_message = _normalize_text(message)
    for option in style_options:
        label = option.get("label")
        value = option.get("value")
        if isinstance(label, str) and _normalize_text(label) == normalized_message:
            return option
        if isinstance(value, str) and _normalize_text(value) == normalized_message:
            return option
    return None


def _build_draft_reply(style_label: str, draft: str) -> str:
    return (
        "\u597d\u7684\uff0c\u6211\u4f1a\u6309\u300c"
        f"{style_label}"
        "\u300d\u98ce\u683c\u751f\u6210\u3002\u4e0b\u9762\u662f\u7b2c\u4e00\u7248\u7b80\u5386\u521d\u7a3f\uff1a\n\n"
        f"{draft}\n\n"
        "\u4f60\u53ef\u4ee5\u544a\u8bc9\u6211\u54ea\u91cc\u9700\u8981\u4fee\u6539\uff0c"
        "\u6bd4\u5982\uff1a\u66f4\u7a81\u51fa\u9879\u76ee\u3001\u538b\u7f29\u5de5\u4f5c\u7ecf\u5386\u3001"
        "\u589e\u52a0\u6280\u80fd\u6808\u3001\u8c03\u6574\u8bed\u6c14\u7b49\u3002"
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
    extracted_slots: dict = {}
    slot_extractor_called = False
    slot_extractor_raw_output = ""
    slot_extractor_parse_success = False
    slots_before = dict(state.slots)
    slots_after = dict(state.slots)
    options: list[dict] = []
    selected_style = ""
    draft_generated = False
    draft_length = 0
    revise_called = False
    final_confirmed = False

    if state.stage == "start":
        task_router_result = await detect_task_type(payload.message)

        if task_router_result["task_type"] == "resume":
            state.task_type = "resume"
            state.stage = "collect_info"
            template = load_task_template("resume")
            task_template_loaded = bool(template)
            required_slots = template.get("required_slots", []) if template else []

            slot_questions = template.get("slot_questions", {}) if template else {}
            question_lines = _build_question_lines(required_slots[:3], slot_questions)

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
    elif state.stage == "collect_info":
        template = load_task_template(state.task_type or "")
        task_template_loaded = bool(template)
        required_slots = template.get("required_slots", []) if template else []

        slot_extractor_called = True
        extraction_result = await extract_slots(payload.message, template, state.slots)
        slot_extractor_raw_output = extraction_result.get("raw_llm_output", "")
        slot_extractor_parse_success = bool(extraction_result.get("parse_success", False))
        extracted_slots = extraction_result.get("updated_slots", {})
        state.slots = extracted_slots if isinstance(extracted_slots, dict) else dict(state.slots)
        slots_after = dict(state.slots)
        state.missing_slots = check_missing_slots(template, state.slots)

        if state.missing_slots:
            slot_questions = template.get("slot_questions", {}) if template else {}
            question_lines = _build_question_lines(state.missing_slots[:2], slot_questions)

            if question_lines:
                reply = (
                    "\u6211\u5df2\u7ecf\u8bb0\u5f55\u4e86\u4f60\u521a\u624d\u63d0\u4f9b\u7684\u4fe1\u606f\u3002"
                    "\u73b0\u5728\u8fd8\u7f3a\u5c11\u4e0b\u9762\u51e0\u9879\uff1a\n\n"
                    + "\n".join(question_lines)
                    + "\n\n"
                    + "\u4f60\u53ef\u4ee5\u7ee7\u7eed\u8865\u5145\u3002"
                )
            else:
                reply = (
                    "\u6211\u5df2\u7ecf\u8bb0\u5f55\u4e86\u4f60\u521a\u624d\u63d0\u4f9b\u7684\u4fe1\u606f\u3002"
                    "\u8fd8\u6709\u4e9b\u5fc5\u586b\u9879\u672a\u8865\u5168\uff0c\u4f60\u53ef\u4ee5\u7ee7\u7eed\u8865\u5145\u3002"
                )
        else:
            state.stage = "confirm_style"
            options = template.get("style_options", []) if template else []
            reply = (
                "\u57fa\u7840\u4fe1\u606f\u5df2\u7ecf\u591f\u4e86\u3002"
                "\u63a5\u4e0b\u6765\u8bf7\u786e\u8ba4\u7b80\u5386\u98ce\u683c\uff0c"
                "\u4f60\u53ef\u4ee5\u9009\u62e9\u4e0b\u9762\u4e00\u79cd\uff1a"
            )
    elif state.stage == "confirm_style":
        template = load_task_template(state.task_type or "")
        task_template_loaded = bool(template)
        required_slots = template.get("required_slots", []) if template else []
        options = template.get("style_options", []) if template else []

        matched_style_option = _match_style_option(payload.message, options)
        if matched_style_option:
            selected_style = matched_style_option.get("value", "")
            state.slots["style"] = selected_style
            state.stage = "draft"
            slots_after = dict(state.slots)

            draft_result = await generate_resume_draft(state.slots, template)
            draft = draft_result.get("draft", "")
            state.slots["resume_draft"] = draft
            state.stage = "revise"
            slots_after = dict(state.slots)
            draft_generated = True
            draft_length = len(draft)
            options = []
            reply = _build_draft_reply(matched_style_option.get("label", ""), draft)
        else:
            reply = (
                "\u6211\u9700\u8981\u5148\u786e\u8ba4\u7b80\u5386\u98ce\u683c\uff0c"
                "\u4f60\u53ef\u4ee5\u9009\u62e9\u4e0b\u9762\u4e00\u79cd\uff1a"
            )
    elif state.stage == "draft":
        template = load_task_template(state.task_type or "")
        task_template_loaded = bool(template)
        required_slots = template.get("required_slots", []) if template else []

        draft_result = await generate_resume_draft(state.slots, template)
        draft = draft_result.get("draft", "")
        state.slots["resume_draft"] = draft
        state.stage = "revise"
        slots_after = dict(state.slots)
        draft_generated = True
        draft_length = len(draft)

        style_options = template.get("style_options", []) if template else []
        style_label = next(
            (
                option.get("label", "")
                for option in style_options
                if option.get("value") == state.slots.get("style")
            ),
            "\u9ed8\u8ba4",
        )
        reply = _build_draft_reply(style_label, draft)
    elif state.stage == "revise":
        template = load_task_template(state.task_type or "")
        task_template_loaded = bool(template)
        required_slots = template.get("required_slots", []) if template else []

        normalized_message = payload.message.strip()
        if any(keyword in normalized_message for keyword in FINAL_CONFIRM_WORDS):
            state.stage = "final"
            final_confirmed = True
            reply = (
                "\u597d\u7684\uff0c\u5f53\u524d\u7b80\u5386\u5df2\u786e\u8ba4\u4f5c\u4e3a\u6700\u7ec8\u7248\u3002"
                "\u4e0b\u4e00\u6b65\u53ef\u4ee5\u8fdb\u5165 Word \u7b80\u5386\u5bfc\u51fa\u529f\u80fd\u3002"
            )
        else:
            current_draft = state.slots.get("resume_draft", "")
            revise_result = await revise_resume_draft(current_draft, payload.message, state.slots)
            revised_draft = revise_result.get("revised_draft", current_draft)
            state.slots["resume_draft"] = revised_draft
            slots_after = dict(state.slots)
            revise_called = True
            draft_length = len(revised_draft)
            reply = (
                "\u6211\u5df2\u7ecf\u6839\u636e\u4f60\u7684\u610f\u89c1\u4fee\u6539\u4e86\u7b80\u5386\uff0c\u4e0b\u9762\u662f\u65b0\u7248\uff1a\n\n"
                f"{revised_draft}\n\n"
                "\u4f60\u8fd8\u53ef\u4ee5\u7ee7\u7eed\u4fee\u6539\uff0c\u6216\u8005\u8f93\u5165\uff1a\u786e\u8ba4\u6700\u7ec8\u7248\u3002"
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
            "extracted_slots": extracted_slots,
            "slot_extractor_called": slot_extractor_called,
            "slot_extractor_raw_output": slot_extractor_raw_output,
            "slot_extractor_parse_success": slot_extractor_parse_success,
            "slots_before": slots_before,
            "slots_after": slots_after,
            "history_count": len(state.history),
            "task_router_result": task_router_result,
            "task_template_loaded": task_template_loaded,
            "required_slots": required_slots,
            "options_count": len(options),
            "selected_style": selected_style,
            "draft_generated": draft_generated,
            "draft_length": draft_length,
            "revise_called": revise_called,
            "final_confirmed": final_confirmed,
        },
        options=options,
    )
