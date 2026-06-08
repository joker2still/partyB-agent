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
    "确认最终版",
    "最终版",
    "可以了",
    "没问题",
    "确认",
]

KNOWN_STAGES = {
    "start",
    "collect_info",
    "confirm_style",
    "draft",
    "revise",
    "final",
    "clarify_task",
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
        f"好的，我会按「{style_label}」风格生成。下面是第一版简历初稿：\n\n"
        f"{draft}\n\n"
        "你可以告诉我哪里需要修改，比如：更突出项目、压缩工作经历、增加技能栈、调整语气等。"
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
        graph_state["reply"] = "当前流程状态异常，我会重新回到需求确认阶段。"
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
                "好的，我理解你想写简历。为了先搭好简历骨架，我需要了解下面几项：\n\n"
                + "\n".join(question_lines)
                + "\n\n"
                + "你可以一次性回答，也可以先回答其中一部分。"
            )
        else:
            graph_state["reply"] = (
                "好的，我理解你想写简历。接下来我会先收集你的基础信息，"
                "你可以先告诉我求职方向、教育经历和工作/项目经历。"
            )
        graph_state["workflow_action"] = "detect_task"
    else:
        agent_state.task_type = "unknown"
        agent_state.stage = "clarify_task"
        graph_state["reply"] = (
            "我还不确定你想完成哪类任务。你可以简单说一下你想让我帮你产出什么，"
            "比如简历、PPT、方案、文案或代码项目。"
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
                "我已经记录了你刚才提供的信息。现在还缺少下面几项：\n\n"
                + "\n".join(question_lines)
                + "\n\n"
                + "你可以继续补充。"
            )
        else:
            graph_state["reply"] = "我已经记录了你刚才提供的信息。还有些必填项未补全，你可以继续补充。"
    else:
        agent_state.stage = "confirm_style"
        graph_state["options"] = template.get("style_options", []) if template else []
        graph_state["reply"] = "基础信息已经够了。接下来请确认简历风格，你可以选择下面一种："

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
        graph_state["reply"] = "我需要先确认简历风格，你可以选择下面一种："
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
        "默认",
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
        agent_state.stage = "final"
        debug["final_confirmed"] = True
        graph_state["reply"] = "好的，当前简历已确认作为最终版。下一步可以进入 Word 简历导出功能。"
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
        "我已经根据你的意见修改了简历，下面是新版：\n\n"
        f"{revised_draft}\n\n"
        "你还可以继续修改，或者输入：确认最终版。"
    )
    graph_state["workflow_action"] = "revise_draft"
    return _finalize_debug(graph_state)


async def final_node(graph_state: GraphState) -> GraphState:
    graph_state["reply"] = "好的，当前简历已确认作为最终版。下一步可以进入 Word 简历导出功能。"
    graph_state["workflow_action"] = "final_idle"
    return _finalize_debug(graph_state)


async def clarify_task_node(graph_state: GraphState) -> GraphState:
    graph_state["reply"] = (
        "我还不确定你想完成哪类任务。你可以简单说一下你想让我帮你产出什么，"
        "比如简历、PPT、方案、文案或代码项目。"
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
        },
    )

    graph.add_edge("start_node", END)
    graph.add_edge("collect_info_node", END)
    graph.add_edge("confirm_style_node", END)
    graph.add_edge("draft_node", END)
    graph.add_edge("revise_node", END)
    graph.add_edge("final_node", END)
    graph.add_edge("clarify_task_node", END)
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


def _route_preview_stage(preview_state: PreviewState) -> str:
    route_map = {
        "start": "start_preview_node",
        "collect_info": "collect_info_preview_node",
        "confirm_style": "confirm_style_preview_node",
        "draft": "draft_preview_node",
        "revise": "revise_preview_node",
        "final": "final_preview_node",
        "clarify_task": "clarify_task_preview_node",
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
        },
    )

    graph.add_edge("start_preview_node", END)
    graph.add_edge("collect_info_preview_node", END)
    graph.add_edge("confirm_style_preview_node", END)
    graph.add_edge("draft_preview_node", END)
    graph.add_edge("revise_preview_node", END)
    graph.add_edge("final_preview_node", END)
    graph.add_edge("clarify_task_preview_node", END)
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
