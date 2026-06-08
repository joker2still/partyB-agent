from app.agents.draft_generator import generate_resume_draft
from app.agents.langgraph_workflow import run_langgraph_preview
from app.agents.resume_reviser import revise_resume_draft
from app.agents.slot_checker import check_missing_slots
from app.agents.slot_extractor import extract_slots
from app.agents.task_router import detect_task_type
from app.core.config import get_settings
from app.models.state import AgentState
from app.services.template_loader import load_task_template


settings = get_settings()

FINAL_CONFIRM_WORDS = [
    "确认最终版",
    "最终版",
    "可以了",
    "没问题",
    "确认",
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
        f"好的，我会按「{style_label}」风格生成。下面是第一版简历初稿：\n\n"
        f"{draft}\n\n"
        "你可以告诉我哪里需要修改，比如：更突出项目、压缩工作经历、增加技能栈、调整语气等。"
    )


def _build_base_debug(state: AgentState, user_message: str) -> dict:
    model = settings.ollama_model
    if settings.llm_provider == "api":
        model = settings.api_model or ""

    return {
        "session_id": state.session_id,
        "provider": settings.llm_provider,
        "model": model,
        "raw_message": user_message,
        "task_type": state.task_type,
        "stage": state.stage,
        "slots": dict(state.slots),
        "missing_slots": list(state.missing_slots),
        "extracted_slots": {},
        "slot_extractor_called": False,
        "slot_extractor_raw_output": "",
        "slot_extractor_parse_success": False,
        "slots_before": dict(state.slots),
        "slots_after": dict(state.slots),
        "task_router_result": None,
        "task_template_loaded": False,
        "required_slots": [],
        "options_count": 0,
        "selected_style": "",
        "draft_generated": False,
        "draft_length": 0,
        "revise_called": False,
        "final_confirmed": False,
        "workflow_stage_before": state.stage,
        "workflow_stage_after": state.stage,
        "workflow_action": "",
        "langgraph_enabled": True,
        "langgraph_preview": {},
    }


async def handle_chat_turn(state: AgentState, user_message: str) -> dict:
    debug = _build_base_debug(state, user_message)
    debug["langgraph_preview"] = run_langgraph_preview(state.stage, state.task_type, user_message)
    options: list[dict] = []
    reply = ""
    workflow_action = ""

    if state.stage == "start":
        task_router_result = await detect_task_type(user_message)
        debug["task_router_result"] = task_router_result

        if task_router_result["task_type"] == "resume":
            state.task_type = "resume"
            state.stage = "collect_info"
            template = load_task_template("resume")
            debug["task_template_loaded"] = bool(template)
            debug["required_slots"] = template.get("required_slots", []) if template else []

            slot_questions = template.get("slot_questions", {}) if template else {}
            question_lines = _build_question_lines(debug["required_slots"][:3], slot_questions)

            if question_lines:
                reply = (
                    "好的，我理解你想写简历。为了先搭好简历骨架，我需要了解下面几项：\n\n"
                    + "\n".join(question_lines)
                    + "\n\n"
                    + "你可以一次性回答，也可以先回答其中一部分。"
                )
            else:
                reply = (
                    "好的，我理解你想写简历。接下来我会先收集你的基础信息，"
                    "你可以先告诉我求职方向、教育经历和工作/项目经历。"
                )
            workflow_action = "detect_task"
        else:
            state.task_type = "unknown"
            state.stage = "clarify_task"
            reply = (
                "我还不确定你想完成哪类任务。你可以简单说一下你想让我帮你产出什么，"
                "比如简历、PPT、方案、文案或代码项目。"
            )
            workflow_action = "clarify_task"

    elif state.stage == "collect_info":
        template = load_task_template(state.task_type or "")
        debug["task_template_loaded"] = bool(template)
        debug["required_slots"] = template.get("required_slots", []) if template else []
        debug["slot_extractor_called"] = True

        extraction_result = await extract_slots(user_message, template, state.slots)
        debug["slot_extractor_raw_output"] = extraction_result.get("raw_llm_output", "")
        debug["slot_extractor_parse_success"] = bool(extraction_result.get("parse_success", False))

        extracted_slots = extraction_result.get("updated_slots", {})
        if isinstance(extracted_slots, dict):
            state.slots = extracted_slots
        debug["extracted_slots"] = dict(state.slots)
        debug["slots_after"] = dict(state.slots)

        state.missing_slots = check_missing_slots(template, state.slots)
        debug["missing_slots"] = list(state.missing_slots)

        if state.missing_slots:
            slot_questions = template.get("slot_questions", {}) if template else {}
            question_lines = _build_question_lines(state.missing_slots[:2], slot_questions)

            if question_lines:
                reply = (
                    "我已经记录了你刚才提供的信息。现在还缺少下面几项：\n\n"
                    + "\n".join(question_lines)
                    + "\n\n"
                    + "你可以继续补充。"
                )
            else:
                reply = "我已经记录了你刚才提供的信息。还有些必填项未补全，你可以继续补充。"
        else:
            state.stage = "confirm_style"
            options = template.get("style_options", []) if template else []
            reply = "基础信息已经够了。接下来请确认简历风格，你可以选择下面一种："
        workflow_action = "collect_slots"

    elif state.stage == "confirm_style":
        template = load_task_template(state.task_type or "")
        debug["task_template_loaded"] = bool(template)
        debug["required_slots"] = template.get("required_slots", []) if template else []
        options = template.get("style_options", []) if template else []

        matched_style_option = _match_style_option(user_message, options)
        if matched_style_option:
            selected_style = matched_style_option.get("value", "")
            state.slots["style"] = selected_style
            state.stage = "draft"
            debug["selected_style"] = selected_style
            debug["slots_after"] = dict(state.slots)

            draft_result = await generate_resume_draft(state.slots, template)
            draft = draft_result.get("draft", "")
            state.slots["resume_draft"] = draft
            state.stage = "revise"
            debug["slots_after"] = dict(state.slots)
            debug["draft_generated"] = True
            debug["draft_length"] = len(draft)
            options = []
            reply = _build_draft_reply(matched_style_option.get("label", ""), draft)
            workflow_action = "generate_draft"
        else:
            reply = "我需要先确认简历风格，你可以选择下面一种："
            workflow_action = "confirm_style"

    elif state.stage == "draft":
        template = load_task_template(state.task_type or "")
        debug["task_template_loaded"] = bool(template)
        debug["required_slots"] = template.get("required_slots", []) if template else []

        draft_result = await generate_resume_draft(state.slots, template)
        draft = draft_result.get("draft", "")
        state.slots["resume_draft"] = draft
        state.stage = "revise"
        debug["slots_after"] = dict(state.slots)
        debug["draft_generated"] = True
        debug["draft_length"] = len(draft)

        style_options = template.get("style_options", []) if template else []
        style_label = next(
            (
                option.get("label", "")
                for option in style_options
                if option.get("value") == state.slots.get("style")
            ),
            "默认",
        )
        reply = _build_draft_reply(style_label, draft)
        workflow_action = "generate_draft"

    elif state.stage == "revise":
        template = load_task_template(state.task_type or "")
        debug["task_template_loaded"] = bool(template)
        debug["required_slots"] = template.get("required_slots", []) if template else []

        normalized_message = user_message.strip()
        if any(keyword in normalized_message for keyword in FINAL_CONFIRM_WORDS):
            state.stage = "final"
            debug["final_confirmed"] = True
            reply = "好的，当前简历已确认作为最终版。下一步可以进入 Word 简历导出功能。"
            workflow_action = "final_confirmed"
        else:
            current_draft = state.slots.get("resume_draft", "")
            revise_result = await revise_resume_draft(current_draft, user_message, state.slots)
            revised_draft = revise_result.get("revised_draft", current_draft)
            state.slots["resume_draft"] = revised_draft
            debug["slots_after"] = dict(state.slots)
            debug["revise_called"] = True
            debug["draft_length"] = len(revised_draft)
            reply = (
                "我已经根据你的意见修改了简历，下面是新版：\n\n"
                f"{revised_draft}\n\n"
                "你还可以继续修改，或者输入：确认最终版。"
            )
            workflow_action = "revise_draft"

    elif state.stage == "final":
        reply = "好的，当前简历已确认作为最终版。下一步可以进入 Word 简历导出功能。"
        workflow_action = "final_idle"

    elif state.stage == "clarify_task":
        reply = (
            "我还不确定你想完成哪类任务。你可以简单说一下你想让我帮你产出什么，"
            "比如简历、PPT、方案、文案或代码项目。"
        )
        workflow_action = "clarify_task"

    else:
        state.stage = "start"
        reply = "当前流程状态异常，我会重新回到需求确认阶段。"
        workflow_action = "clarify_task"

    debug["task_type"] = state.task_type
    debug["stage"] = state.stage
    debug["slots"] = dict(state.slots)
    debug["missing_slots"] = list(state.missing_slots)
    debug["options_count"] = len(options)
    debug["workflow_stage_after"] = state.stage
    debug["workflow_action"] = workflow_action

    return {
        "reply": reply,
        "options": options,
        "debug": debug,
    }
