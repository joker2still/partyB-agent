from fastapi.testclient import TestClient

from app.main import app
from app.services import state_store
from app.services.state_store import get_or_create_state, save_state


client = TestClient(app)


def test_chat_routes_resume_request(monkeypatch) -> None:
    state_store._STATE_STORE.clear()

    async def fail_if_called(_: str) -> str:
        raise AssertionError("call_llm should not be used for keyword-based routing")

    monkeypatch.setattr("app.main.call_llm", fail_if_called)

    response = client.post(
        "/chat",
        json={"session_id": None, "message": "\u5e2e\u6211\u5199\u7b80\u5386"},
    )

    body = response.json()

    assert response.status_code == 200
    assert body["debug"]["task_type"] == "resume"
    assert body["debug"]["stage"] == "collect_info"
    assert body["debug"]["task_router_result"]["task_type"] == "resume"
    assert body["debug"]["task_router_result"]["confidence"] == 0.9
    assert body["debug"]["task_template_loaded"] is True
    assert body["debug"]["required_slots"] == [
        "target_position",
        "education",
        "work_experience",
        "project_experience",
    ]
    assert body["debug"]["history_count"] == 2
    assert "\u4f60\u60f3\u6295\u4ec0\u4e48\u5c97\u4f4d" in body["reply"]
    assert "\u8bf7\u544a\u8bc9\u6211\u4f60\u7684\u6559\u80b2\u7ecf\u5386" in body["reply"]
    assert "\u8bf7\u544a\u8bc9\u6211\u4f60\u7684\u5de5\u4f5c\u7ecf\u5386" in body["reply"]


def test_chat_routes_unknown_request(monkeypatch) -> None:
    state_store._STATE_STORE.clear()

    async def fake_detect_task_type(message: str) -> dict:
        assert message == "do something"
        return {
            "task_type": "unknown",
            "confidence": 0.4,
            "reason": "not enough signal",
        }

    async def fail_if_called(_: str) -> str:
        raise AssertionError("call_llm should not be used during start-stage routing")

    monkeypatch.setattr("app.main.detect_task_type", fake_detect_task_type)
    monkeypatch.setattr("app.main.call_llm", fail_if_called)

    response = client.post(
        "/chat",
        json={"session_id": None, "message": "do something"},
    )

    body = response.json()

    assert response.status_code == 200
    assert body["debug"]["task_type"] == "unknown"
    assert body["debug"]["stage"] == "clarify_task"
    assert body["debug"]["task_router_result"]["task_type"] == "unknown"
    assert body["debug"]["task_template_loaded"] is False
    assert body["debug"]["required_slots"] == []
    assert body["debug"]["history_count"] == 2
    assert "\u6211\u8fd8\u4e0d\u786e\u5b9a" in body["reply"]


def test_chat_uses_normal_llm_reply_after_start(monkeypatch) -> None:
    state_store._STATE_STORE.clear()

    state = get_or_create_state("existing-session")
    state.task_type = "resume"
    state.stage = "collect_info"
    save_state(state)

    raw_message = "\u8fd9\u662f\u6211\u7684\u6559\u80b2\u7ecf\u5386"
    llm_reply = "\u6211\u5df2\u7ecf\u4e86\u89e3\u4f60\u7684\u6559\u80b2\u80cc\u666f\u3002"

    async def fake_call_llm(prompt: str) -> str:
        assert raw_message in prompt
        return llm_reply

    async def fail_if_called(_: str) -> dict:
        raise AssertionError("detect_task_type should not be called after start stage")

    monkeypatch.setattr("app.main.call_llm", fake_call_llm)
    monkeypatch.setattr("app.main.detect_task_type", fail_if_called)

    response = client.post(
        "/chat",
        json={"session_id": "existing-session", "message": raw_message},
    )

    body = response.json()

    assert response.status_code == 200
    assert body["session_id"] == "existing-session"
    assert body["reply"] == llm_reply
    assert body["debug"]["task_type"] == "resume"
    assert body["debug"]["stage"] == "collect_info"
    assert body["debug"]["task_router_result"] is None
    assert body["debug"]["task_template_loaded"] is True
    assert body["debug"]["required_slots"] == [
        "target_position",
        "education",
        "work_experience",
        "project_experience",
    ]
    assert body["debug"]["history_count"] == 2
