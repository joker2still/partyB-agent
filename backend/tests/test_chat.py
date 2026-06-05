from fastapi.testclient import TestClient

from app.main import app
from app.services import state_store


client = TestClient(app)


def test_chat_returns_reply_and_debug(monkeypatch) -> None:
    state_store._STATE_STORE.clear()

    raw_message = "\u6d4b\u8bd5\u6d88\u606f"
    llm_reply = "\u5df2\u7406\u89e3\u4f60\u7684\u9700\u6c42\u3002"

    async def fake_call_llm(prompt: str) -> str:
        assert raw_message in prompt
        return llm_reply

    monkeypatch.setattr("app.main.call_llm", fake_call_llm)

    response = client.post(
        "/chat",
        json={"session_id": None, "message": raw_message},
    )

    body = response.json()

    assert response.status_code == 200
    assert isinstance(body["session_id"], str)
    assert body["session_id"]
    assert body["reply"] == llm_reply
    assert body["debug"]["session_id"] == body["session_id"]
    assert body["debug"]["raw_message"] == raw_message
    assert body["debug"]["task_type"] is None
    assert body["debug"]["stage"] == "start"
    assert body["debug"]["slots"] == {}
    assert body["debug"]["missing_slots"] == []
    assert body["debug"]["history_count"] == 2


def test_chat_reuses_session_and_accumulates_history(monkeypatch) -> None:
    state_store._STATE_STORE.clear()

    async def fake_call_llm(_: str) -> str:
        return "ack"

    monkeypatch.setattr("app.main.call_llm", fake_call_llm)

    first_response = client.post(
        "/chat",
        json={"session_id": None, "message": "first"},
    )
    session_id = first_response.json()["session_id"]

    second_response = client.post(
        "/chat",
        json={"session_id": session_id, "message": "second"},
    )

    second_body = second_response.json()

    assert second_response.status_code == 200
    assert second_body["session_id"] == session_id
    assert second_body["debug"]["history_count"] == 4
