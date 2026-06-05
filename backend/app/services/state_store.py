from uuid import uuid4

from app.models.state import AgentMessage, AgentState


_STATE_STORE: dict[str, AgentState] = {}


def get_or_create_state(session_id: str | None) -> AgentState:
    resolved_session_id = session_id or str(uuid4())

    state = _STATE_STORE.get(resolved_session_id)
    if state is not None:
        return state

    state = AgentState(session_id=resolved_session_id)
    _STATE_STORE[resolved_session_id] = state
    return state


def save_state(state: AgentState) -> None:
    _STATE_STORE[state.session_id] = state


def append_message(state: AgentState, role: str, content: str) -> None:
    state.history.append(AgentMessage(role=role, content=content))
