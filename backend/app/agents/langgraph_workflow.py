import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.draft_generator import generate_resume_draft
from app.agents.resume_reviser import revise_resume_draft
from app.agents.slot_checker import check_missing_slots
from app.agents.slot_extractor import extract_slots
from app.agents.task_router import detect_task_type
from app.core.config import get_settings
from app.models.state import AgentState
from app.services.template_loader import load_task_template


settings = get_settings()

FINAL_CONFIRM_WORDS = [
    "\u786e\u8ba4\u6700\u7ec8\u7248",
    "\u6700\u7ec8\u7248",
    "\u53ef\u4ee5\u4e86",
    "\u6ca1\u95ee\u9898",
    "\u786e\u8ba4",
]
EXPORT_REQUIRED_SLOTS = ["name", "phone", "email", "age", "location"]
EXPORT_SLOT_LABELS = {
    "name": "\u59d3\u540d",
    "phone": "\u624b\u673a",
    "email": "\u90ae\u7bb1",
    "age": "\u5e74\u9f84",
    "location": "\u73b0\u5c45\u5730",
}
KNOWN_STAGES = {
    "start",
    "collect_info",
    "confirm_style",
    "draft",
    "revise",
    "final",
    "clarify_task",
    "collect_export_info",
}


class GraphState(TypedDict):
    stage: str
    task_type: str | None
    state: AgentState
    message: str
    reply: str
    options: list[dict[str, Any]]
    debug: dict[str, Any]
    workflow_action: str


class PreviewState(TypedDict):
    stage: str
    task_type: str | None
    message: str
    reply: str
    workflow_action: str


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


def _build_base_debug(state: AgentState, user_message: str) -> dict[str, Any]:
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
        "langgraph_real_workflow": True,
        "langgraph_preview": {},
        "invalid_stage": False,
        "export_info_missing": [],
        "export_info_collected": False,
    }


def _finalize_debug(graph_state: GraphState) -> GraphState:
    agent_state = graph_state["state"]
    debug = graph_state["debug"]
    options = graph_state["options"]

    debug["task_type"] = agent_state.task_type
    debug["stage"] = agent_state.stage
    debug["slots"] = dict(agent_state.slots)
    debug["missing_slots"] = list(agent_state.missing_slots)
    debug["options_count"] = len(options)
    debug["workflow_stage_after"] = agent_state.stage
    debug["workflow_action"] = graph_state["workflow_action"]

    graph_state["stage"] = agent_state.stage
    graph_state["task_type"] = agent_state.task_type
    return graph_state


def _check_missing_export_info(slots: dict) -> list[str]:
    missing: list[str] = []
    for key in EXPORT_REQUIRED_SLOTS:
        value = slots.get(key, "")
        if not isinstance(value, str) or not value.strip():
            missing.append(key)
    return missing


def _missing_export_info_labels(slot_names: list[str]) -> str:
    return "\u3001".join(EXPORT_SLOT_LABELS.get(slot_name, slot_name) for slot_name in slot_names)


def _extract_export_info(message: str, current_slots: dict) -> dict[str, str]:
    updated = dict(current_slots)

    name_match = re.search(r"(?:\u6211\u53eb|\u59d3\u540d(?:\u662f)?[:\uff1a]?|\u540d\u5b57(?:\u662f)?[:\uff1a]?)([\u4e00-\u9fffA-Za-z\u00b7]{2,20})", message)
    if name_match:
        updated["name"] = name_match.group(1).strip()

    phone_patterns = [
        r"(?:\u624b\u673a(?:\u53f7)?[:\uff1a]?\s*)(1\d{10})",
        r"(?:\u7535\u8bdd[:\uff1a]?\s*)(1\d{10})",
        r"(1\d{10})",
    ]
    for pattern in phone_patterns:
        phone_match = re.search(pattern, message)
        if phone_match:
            updated["phone"] = phone_match.group(1)
            break

    email_match = re.search(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", message)
    if email_match:
        updated["email"] = email_match.group(1)

    age_patterns = [
        r"(?:\u5e74\u9f84(?:\u662f)?[:\uff1a]?\s*)(\d{1,2})(?:\u5c81)?",
        r"(?:\u4eca\u5e74\s*)(\d{1,2})\u5c81",
        r"(\d{1,2})\u5c81",
    ]
    for pattern in age_patterns:
        age_match = re.search(pattern, message)
        if age_match:
            updated["age"] = age_match.group(1)
            break

    location_patterns = [
        r"(?:\u73b0\u5c45(?:\u5730)?[:\uff1a]?)([^\uff0c\u3002\uff1b\n]{2,20})",
        r"(?:\u5c45\u4f4f\u5728)([^\uff0c\u3002\uff1b\n]{2,20})",
        r"(?:\u6240\u5728\u5730[:\uff1a]?)([^\uff0c\u3002\uff1b\n]{2,20})",
    ]
    for pattern in location_patterns:
        location_match = re.search(pattern, message)
        if location_match:
            updated["location"] = location_match.group(1).strip()
            break

    return updated


def route_stage_node(graph_state: GraphState) -> GraphState:
    agent_state = graph_state["state"]
    stage = agent_state.stage
    if stage not in KNOWN_STAGES:
        graph_state["debug"]["invalid_stage"] = True
        graph_state["stage"] = "start"
    else:
        graph_state["stage"] = stage
    graph_state["task_type"] = agent_state.task_type
    return graph_state


async def start_node(graph_state: GraphState) -> GraphState:
    agent_state = graph_state["state"]
    message = graph_state["message"]
    debug = graph_state["debug"]

    if debug.get("invalid_stage"):
        agent_state.stage = "start"
        graph_state["reply"] = "\u5f53\u524d\u6d41\u7a0b\u72b6\u6001\u5f02\u5e38\uff0c\u6211\u4f1a\u91cd\u65b0\u56de\u5230\u9700\u6c42\u786e\u8ba4\u9636\u6bb5\u3002"
        graph_state["workflow_action"] = "clarify_task"
        return _finalize_debug(graph_state)

    task_router_result = await detect_task_type(message)
    debug["task_router_result"] = task_router_result

    if task_router_result["task_type"] == "resume":
        agent_state.task_type = "resume"
        agent_state.stage = "collect_info"
        template = load_task_template("resume")
        debug["task_template_loaded"] = bool(template)
        debug["required_slots"] = template.get("required_slots", []) if template else []

        slot_questions = template.get("slot_questions", {}) if template else {}
        question_lines = _build_question_lines(debug["required_slots"][:3], slot_questions)

        if question_lines:
            graph_state["reply"] = (
                "\u597d\u7684\uff0c\u6211\u7406\u89e3\u4f60\u60f3\u5199\u7b80\u5386\u3002"
                "\u4e3a\u4e86\u5148\u642d\u597d\u7b80\u5386\u9aa8\u67b6\uff0c"
                "\u6211\u9700\u8981\u4e86\u89e3\u4e0b\u9762\u51e0\u9879\uff1a\n\n"
                + "\n".join(question_lines)
                + "\n\n"
                + "\u4f60\u53ef\u4ee5\u4e00\u6b21\u6027\u56de\u7b54\uff0c\u4e5f\u53ef\u4ee5\u5148\u56de\u7b54\u5176\u4e2d\u4e00\u90e8\u5206\u3002"
            )
        else:
            graph_state["reply"] = (
                "\u597d\u7684\uff0c\u6211\u7406\u89e3\u4f60\u60f3\u5199\u7b80\u5386\u3002"
                "\u63a5\u4e0b\u6765\u6211\u4f1a\u5148\u6536\u96c6\u4f60\u7684\u57fa\u672c\u4fe1\u606f\uff0c"
                "\u4f60\u53ef\u4ee5\u5148\u544a\u8bc9\u6211\u6c42\u804c\u65b9\u5411\u3001\u6559\u80b2\u7ecf\u5386\u548c\u5de5\u4f5c/\u9879\u76ee\u7ecf\u5386\u3002"
            )
        graph_state["workflow_action"] = "detect_task"
    else:
        agent_state.task_type = "unknown"
        agent_state.stage = "clarify_task"
        graph_state["reply"] = (
            "\u6211\u8fd8\u4e0d\u786e\u5b9a\u4f60\u60f3\u5b8c\u6210\u54ea\u7c7b\u4efb\u52a1\u3002"
            "\u4f60\u53ef\u4ee5\u7b80\u5355\u8bf4\u4e00\u4e0b\u4f60\u60f3\u8ba9\u6211\u5e2e\u4f60\u4ea7\u51fa\u4ec0\u4e48\uff0c"
            "\u6bd4\u5982\u7b80\u5386\u3001PPT\u3001\u65b9\u6848\u3001\u6587\u6848\u6216\u4ee3\u7801\u9879\u76ee\u3002"
        )
        graph_state["workflow_action"] = "clarify_task"

    return _finalize_debug(graph_state)


async def collect_info_node(graph_state: GraphState) -> GraphState:
    agent_state = graph_state["state"]
    message = graph_state["message"]
    debug = graph_state["debug"]

    template = load_task_template(agent_state.task_type or "")
    debug["task_template_loaded"] = bool(template)
    debug["required_slots"] = template.get("required_slots", []) if template else []
    debug["slot_extractor_called"] = True

    extraction_result = await extract_slots(message, template, agent_state.slots)
    debug["slot_extractor_raw_output"] = extraction_result.get("raw_llm_output", "")
    debug["slot_extractor_parse_success"] = bool(extraction_result.get("parse_success", False))

    extracted_slots = extraction_result.get("updated_slots", {})
    if isinstance(extracted_slots, dict):
        agent_state.slots = extracted_slots
    debug["extracted_slots"] = dict(agent_state.slots)
    debug["slots_after"] = dict(agent_state.slots)

    agent_state.missing_slots = check_missing_slots(template, agent_state.slots)
    debug["missing_slots"] = list(agent_state.missing_slots)

    if agent_state.missing_slots:
        slot_questions = template.get("slot_questions", {}) if template else {}
        question_lines = _build_question_lines(agent_state.missing_slots[:2], slot_questions)
        if question_lines:
            graph_state["reply"] = (
                "\u6211\u5df2\u7ecf\u8bb0\u5f55\u4e86\u4f60\u521a\u624d\u63d0\u4f9b\u7684\u4fe1\u606f\u3002"
                "\u73b0\u5728\u8fd8\u7f3a\u5c11\u4e0b\u9762\u51e0\u9879\uff1a\n\n"
                + "\n".join(question_lines)
                + "\n\n"
                + "\u4f60\u53ef\u4ee5\u7ee7\u7eed\u8865\u5145\u3002"
            )
        else:
            graph_state["reply"] = "\u6211\u5df2\u7ecf\u8bb0\u5f55\u4e86\u4f60\u521a\u624d\u63d0\u4f9b\u7684\u4fe1\u606f\u3002\u8fd8\u6709\u4e9b\u5fc5\u586b\u9879\u672a\u8865\u5168\uff0c\u4f60\u53ef\u4ee5\u7ee7\u7eed\u8865\u5145\u3002"
    else:
        agent_state.stage = "confirm_style"
        graph_state["options"] = template.get("style_options", []) if template else []
        graph_state["reply"] = "\u57fa\u7840\u4fe1\u606f\u5df2\u7ecf\u591f\u4e86\u3002\u63a5\u4e0b\u6765\u8bf7\u786e\u8ba4\u7b80\u5386\u98ce\u683c\uff0c\u4f60\u53ef\u4ee5\u9009\u62e9\u4e0b\u9762\u4e00\u79cd\uff1a"

    graph_state["workflow_action"] = "collect_slots"
    return _finalize_debug(graph_state)


async def confirm_style_node(graph_state: GraphState) -> GraphState:
    agent_state = graph_state["state"]
    message = graph_state["message"]
    debug = graph_state["debug"]

    template = load_task_template(agent_state.task_type or "")
    debug["task_template_loaded"] = bool(template)
    debug["required_slots"] = template.get("required_slots", []) if template else []
    graph_state["options"] = template.get("style_options", []) if template else []

    matched_style_option = _match_style_option(message, graph_state["options"])
    if matched_style_option:
        selected_style = matched_style_option.get("value", "")
        agent_state.slots["style"] = selected_style
        agent_state.stage = "draft"
        debug["selected_style"] = selected_style
        debug["slots_after"] = dict(agent_state.slots)

        draft_result = await generate_resume_draft(agent_state.slots, template)
        draft = draft_result.get("draft", "")
        agent_state.slots["resume_draft"] = draft
        agent_state.stage = "revise"
        debug["slots_after"] = dict(agent_state.slots)
        debug["draft_generated"] = True
        debug["draft_length"] = len(draft)
        graph_state["options"] = []
        graph_state["reply"] = _build_draft_reply(matched_style_option.get("label", ""), draft)
        graph_state["workflow_action"] = "generate_draft"
    else:
        graph_state["reply"] = "\u6211\u9700\u8981\u5148\u786e\u8ba4\u7b80\u5386\u98ce\u683c\uff0c\u4f60\u53ef\u4ee5\u9009\u62e9\u4e0b\u9762\u4e00\u79cd\uff1a"
        graph_state["workflow_action"] = "confirm_style"

    return _finalize_debug(graph_state)


async def draft_node(graph_state: GraphState) -> GraphState:
    agent_state = graph_state["state"]
    debug = graph_state["debug"]

    template = load_task_template(agent_state.task_type or "")
    debug["task_template_loaded"] = bool(template)
    debug["required_slots"] = template.get("required_slots", []) if template else []

    draft_result = await generate_resume_draft(agent_state.slots, template)
    draft = draft_result.get("draft", "")
    agent_state.slots["resume_draft"] = draft
    agent_state.stage = "revise"
    debug["slots_after"] = dict(agent_state.slots)
    debug["draft_generated"] = True
    debug["draft_length"] = len(draft)

    style_options = template.get("style_options", []) if template else []
    style_label = next(
        (
            option.get("label", "")
            for option in style_options
            if option.get("value") == agent_state.slots.get("style")
        ),
        "\u9ed8\u8ba4",
    )
    graph_state["reply"] = _build_draft_reply(style_label, draft)
    graph_state["workflow_action"] = "generate_draft"
    return _finalize_debug(graph_state)


async def revise_node(graph_state: GraphState) -> GraphState:
    agent_state = graph_state["state"]
    message = graph_state["message"]
    debug = graph_state["debug"]

    template = load_task_template(agent_state.task_type or "")
    debug["task_template_loaded"] = bool(template)
    debug["required_slots"] = template.get("required_slots", []) if template else []

    if any(keyword in message.strip() for keyword in FINAL_CONFIRM_WORDS):
        missing_export_info = _check_missing_export_info(agent_state.slots)
        debug["final_confirmed"] = True
        debug["export_info_missing"] = list(missing_export_info)

        if missing_export_info:
            agent_state.stage = "collect_export_info"
            graph_state["reply"] = (
                "\u7b80\u5386\u5185\u5bb9\u5df2\u786e\u8ba4\u3002"
                "\u4e3a\u4e86\u751f\u6210\u5b8c\u6574 Word \u7b80\u5386\uff0c"
                "\u8fd8\u9700\u8981\u8865\u5145\u4e0b\u9762\u8fd9\u4e9b\u4e2a\u4eba\u4fe1\u606f\uff1a"
                f"{_missing_export_info_labels(missing_export_info)}\u3002"
                "\u4f60\u53ef\u4ee5\u4e00\u6b21\u6027\u544a\u8bc9\u6211\u3002"
            )
            graph_state["workflow_action"] = "collect_export_info"
            return _finalize_debug(graph_state)

        agent_state.stage = "final"
        debug["export_info_collected"] = True
        graph_state["reply"] = "\u597d\u7684\uff0c\u5f53\u524d\u7b80\u5386\u5df2\u786e\u8ba4\u4f5c\u4e3a\u6700\u7ec8\u7248\uff0c\u73b0\u5728\u53ef\u4ee5\u5bfc\u51fa Word \u7b80\u5386\u3002"
        graph_state["workflow_action"] = "final_confirmed"
        return _finalize_debug(graph_state)

    current_draft = agent_state.slots.get("resume_draft", "")
    revise_result = await revise_resume_draft(current_draft, message, agent_state.slots)
    revised_draft = revise_result.get("revised_draft", current_draft)
    agent_state.slots["resume_draft"] = revised_draft
    debug["slots_after"] = dict(agent_state.slots)
    debug["revise_called"] = True
    debug["draft_length"] = len(revised_draft)
    graph_state["reply"] = (
        "\u6211\u5df2\u7ecf\u6839\u636e\u4f60\u7684\u610f\u89c1\u4fee\u6539\u4e86\u7b80\u5386\uff0c\u4e0b\u9762\u662f\u65b0\u7248\uff1a\n\n"
        f"{revised_draft}\n\n"
        "\u4f60\u8fd8\u53ef\u4ee5\u7ee7\u7eed\u4fee\u6539\uff0c\u6216\u8005\u8f93\u5165\uff1a\u786e\u8ba4\u6700\u7ec8\u7248\u3002"
    )
    graph_state["workflow_action"] = "revise_draft"
    return _finalize_debug(graph_state)


async def collect_export_info_node(graph_state: GraphState) -> GraphState:
    agent_state = graph_state["state"]
    message = graph_state["message"]
    debug = graph_state["debug"]

    agent_state.slots = _extract_export_info(message, agent_state.slots)
    debug["slots_after"] = dict(agent_state.slots)
    missing_export_info = _check_missing_export_info(agent_state.slots)
    debug["export_info_missing"] = list(missing_export_info)

    if missing_export_info:
        graph_state["reply"] = (
            "\u8fd8\u6709\u4ee5\u4e0b\u4e2a\u4eba\u4fe1\u606f\u672a\u8865\u5145\u5b8c\u6574\uff1a"
            f"{_missing_export_info_labels(missing_export_info)}\u3002"
            "\u8bf7\u7ee7\u7eed\u8865\u5145\u3002"
        )
        graph_state["workflow_action"] = "collect_export_info"
    else:
        agent_state.stage = "final"
        debug["export_info_collected"] = True
        graph_state["reply"] = "\u4e2a\u4eba\u4fe1\u606f\u5df2\u8865\u5145\u5b8c\u6574\uff0c\u73b0\u5728\u53ef\u4ee5\u5bfc\u51fa Word \u7b80\u5386\u3002"
        graph_state["workflow_action"] = "final_confirmed"

    return _finalize_debug(graph_state)


async def final_node(graph_state: GraphState) -> GraphState:
    debug = graph_state["debug"]
    debug["export_info_collected"] = True
    debug["export_info_missing"] = []
    graph_state["reply"] = "\u597d\u7684\uff0c\u5f53\u524d\u7b80\u5386\u5df2\u786e\u8ba4\u4f5c\u4e3a\u6700\u7ec8\u7248\uff0c\u73b0\u5728\u53ef\u4ee5\u5bfc\u51fa Word \u7b80\u5386\u3002"
    graph_state["workflow_action"] = "final_idle"
    return _finalize_debug(graph_state)


async def clarify_task_node(graph_state: GraphState) -> GraphState:
    graph_state["reply"] = (
        "\u6211\u8fd8\u4e0d\u786e\u5b9a\u4f60\u60f3\u5b8c\u6210\u54ea\u7c7b\u4efb\u52a1\u3002"
        "\u4f60\u53ef\u4ee5\u7b80\u5355\u8bf4\u4e00\u4e0b\u4f60\u60f3\u8ba9\u6211\u5e2e\u4f60\u4ea7\u51fa\u4ec0\u4e48\uff0c"
        "\u6bd4\u5982\u7b80\u5386\u3001PPT\u3001\u65b9\u6848\u3001\u6587\u6848\u6216\u4ee3\u7801\u9879\u76ee\u3002"
    )
    graph_state["workflow_action"] = "clarify_task"
    return _finalize_debug(graph_state)


def _route_stage(graph_state: GraphState) -> str:
    route_map = {
        "start": "start_node",
        "collect_info": "collect_info_node",
        "confirm_style": "confirm_style_node",
        "draft": "draft_node",
        "revise": "revise_node",
        "final": "final_node",
        "clarify_task": "clarify_task_node",
        "collect_export_info": "collect_export_info_node",
    }
    return route_map.get(graph_state.get("stage", "start"), "start_node")


def _build_real_graph():
    graph = StateGraph(GraphState)
    graph.add_node("route_stage_node", route_stage_node)
    graph.add_node("start_node", start_node)
    graph.add_node("collect_info_node", collect_info_node)
    graph.add_node("confirm_style_node", confirm_style_node)
    graph.add_node("draft_node", draft_node)
    graph.add_node("revise_node", revise_node)
    graph.add_node("final_node", final_node)
    graph.add_node("clarify_task_node", clarify_task_node)
    graph.add_node("collect_export_info_node", collect_export_info_node)

    graph.add_edge(START, "route_stage_node")
    graph.add_conditional_edges(
        "route_stage_node",
        _route_stage,
        {
            "start_node": "start_node",
            "collect_info_node": "collect_info_node",
            "confirm_style_node": "confirm_style_node",
            "draft_node": "draft_node",
            "revise_node": "revise_node",
            "final_node": "final_node",
            "clarify_task_node": "clarify_task_node",
            "collect_export_info_node": "collect_export_info_node",
        },
    )

    graph.add_edge("start_node", END)
    graph.add_edge("collect_info_node", END)
    graph.add_edge("confirm_style_node", END)
    graph.add_edge("draft_node", END)
    graph.add_edge("revise_node", END)
    graph.add_edge("final_node", END)
    graph.add_edge("clarify_task_node", END)
    graph.add_edge("collect_export_info_node", END)
    return graph.compile()


def route_stage_preview_node(preview_state: PreviewState) -> PreviewState:
    return preview_state


def start_preview_node(preview_state: PreviewState) -> PreviewState:
    return {"workflow_action": "detect_task"}


def collect_info_preview_node(preview_state: PreviewState) -> PreviewState:
    return {"workflow_action": "collect_slots"}


def confirm_style_preview_node(preview_state: PreviewState) -> PreviewState:
    return {"workflow_action": "confirm_style"}


def draft_preview_node(preview_state: PreviewState) -> PreviewState:
    return {"workflow_action": "generate_draft"}


def revise_preview_node(preview_state: PreviewState) -> PreviewState:
    return {"workflow_action": "revise_draft"}


def final_preview_node(preview_state: PreviewState) -> PreviewState:
    return {"workflow_action": "final_idle"}


def clarify_task_preview_node(preview_state: PreviewState) -> PreviewState:
    return {"workflow_action": "clarify_task"}


def collect_export_info_preview_node(preview_state: PreviewState) -> PreviewState:
    return {"workflow_action": "collect_export_info"}


def _route_preview_stage(preview_state: PreviewState) -> str:
    route_map = {
        "start": "start_preview_node",
        "collect_info": "collect_info_preview_node",
        "confirm_style": "confirm_style_preview_node",
        "draft": "draft_preview_node",
        "revise": "revise_preview_node",
        "final": "final_preview_node",
        "clarify_task": "clarify_task_preview_node",
        "collect_export_info": "collect_export_info_preview_node",
    }
    return route_map.get(preview_state.get("stage", "start"), "start_preview_node")


def _build_preview_graph():
    graph = StateGraph(PreviewState)
    graph.add_node("route_stage_preview_node", route_stage_preview_node)
    graph.add_node("start_preview_node", start_preview_node)
    graph.add_node("collect_info_preview_node", collect_info_preview_node)
    graph.add_node("confirm_style_preview_node", confirm_style_preview_node)
    graph.add_node("draft_preview_node", draft_preview_node)
    graph.add_node("revise_preview_node", revise_preview_node)
    graph.add_node("final_preview_node", final_preview_node)
    graph.add_node("clarify_task_preview_node", clarify_task_preview_node)
    graph.add_node("collect_export_info_preview_node", collect_export_info_preview_node)

    graph.add_edge(START, "route_stage_preview_node")
    graph.add_conditional_edges(
        "route_stage_preview_node",
        _route_preview_stage,
        {
            "start_preview_node": "start_preview_node",
            "collect_info_preview_node": "collect_info_preview_node",
            "confirm_style_preview_node": "confirm_style_preview_node",
            "draft_preview_node": "draft_preview_node",
            "revise_preview_node": "revise_preview_node",
            "final_preview_node": "final_preview_node",
            "clarify_task_preview_node": "clarify_task_preview_node",
            "collect_export_info_preview_node": "collect_export_info_preview_node",
        },
    )

    graph.add_edge("start_preview_node", END)
    graph.add_edge("collect_info_preview_node", END)
    graph.add_edge("confirm_style_preview_node", END)
    graph.add_edge("draft_preview_node", END)
    graph.add_edge("revise_preview_node", END)
    graph.add_edge("final_preview_node", END)
    graph.add_edge("clarify_task_preview_node", END)
    graph.add_edge("collect_export_info_preview_node", END)
    return graph.compile()


_LANGGRAPH_REAL_WORKFLOW = _build_real_graph()
_LANGGRAPH_PREVIEW = _build_preview_graph()


def run_langgraph_preview(stage: str, task_type: str | None, message: str) -> dict[str, Any]:
    initial_state: PreviewState = {
        "stage": stage,
        "task_type": task_type,
        "message": message,
        "reply": "",
        "workflow_action": "",
    }
    return dict(_LANGGRAPH_PREVIEW.invoke(initial_state))


async def run_langgraph_workflow(state: AgentState, user_message: str) -> dict[str, Any]:
    debug = _build_base_debug(state, user_message)
    debug["langgraph_preview"] = run_langgraph_preview(state.stage, state.task_type, user_message)

    initial_state: GraphState = {
        "stage": state.stage,
        "task_type": state.task_type,
        "state": state,
        "message": user_message,
        "reply": "",
        "options": [],
        "debug": debug,
        "workflow_action": "",
    }

    result = await _LANGGRAPH_REAL_WORKFLOW.ainvoke(initial_state)
    return {
        "reply": result["reply"],
        "options": result["options"],
        "debug": result["debug"],
    }
