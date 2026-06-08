from app.agents.langgraph_workflow import run_langgraph_workflow
from app.models.state import AgentState


async def handle_chat_turn(state: AgentState, user_message: str) -> dict:
    result = await run_langgraph_workflow(state, user_message)
    return {
        "reply": result["reply"],
        "options": result["options"],
        "debug": result["debug"],
    }
