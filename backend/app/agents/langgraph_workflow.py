from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class GraphState(TypedDict):
    stage: str
    task_type: str | None
    message: str
    reply: str
    workflow_action: str


def route_stage_node(state: GraphState) -> GraphState:
    return state


def start_node(state: GraphState) -> GraphState:
    return {"workflow_action": "detect_task"}


def collect_info_node(state: GraphState) -> GraphState:
    return {"workflow_action": "collect_slots"}


def confirm_style_node(state: GraphState) -> GraphState:
    return {"workflow_action": "confirm_style"}


def draft_node(state: GraphState) -> GraphState:
    return {"workflow_action": "generate_draft"}


def revise_node(state: GraphState) -> GraphState:
    return {"workflow_action": "revise_draft"}


def final_node(state: GraphState) -> GraphState:
    return {"workflow_action": "final_idle"}


def clarify_task_node(state: GraphState) -> GraphState:
    return {"workflow_action": "clarify_task"}


def _route_stage(state: GraphState) -> str:
    stage = state.get("stage", "start")
    route_map = {
        "start": "start_node",
        "collect_info": "collect_info_node",
        "confirm_style": "confirm_style_node",
        "draft": "draft_node",
        "revise": "revise_node",
        "final": "final_node",
        "clarify_task": "clarify_task_node",
    }
    return route_map.get(stage, "start_node")


def _build_graph():
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


_LANGGRAPH_PREVIEW = _build_graph()


def run_langgraph_preview(stage: str, task_type: str | None, message: str) -> dict:
    initial_state: GraphState = {
        "stage": stage,
        "task_type": task_type,
        "message": message,
        "reply": "",
        "workflow_action": "",
    }
    result = _LANGGRAPH_PREVIEW.invoke(initial_state)
    return dict(result)
