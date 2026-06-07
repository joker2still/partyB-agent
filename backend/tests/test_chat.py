from fastapi.testclient import TestClient

from app.agents.slot_extractor import fallback_extract_slots, robust_json_parse
from app.main import app
from app.services import state_store
from app.services.state_store import get_or_create_state, save_state


client = TestClient(app)


def test_robust_json_parse_supports_fenced_json() -> None:
    raw_text = """```json
{
  "updated_slots": {
    "target_position": "AI Agent 开发岗位",
    "education": "",
    "work_experience": "暂无",
    "project_experience": ""
  },
  "reason": "parsed"
}
```"""

    parsed = robust_json_parse(raw_text)

    assert parsed is not None
    assert parsed["updated_slots"]["target_position"] == "AI Agent 开发岗位"
    assert parsed["updated_slots"]["work_experience"] == "暂无"


def test_fallback_extract_slots_handles_resume_message() -> None:
    template = {
        "required_slots": [
            "target_position",
            "education",
            "work_experience",
            "project_experience",
        ],
        "optional_slots": [],
    }
    message = (
        "我想投 AI Agent 开发岗位，本科是浙江工商大学大数据专业，2023年毕业。"
        "我有一个 Agentic RAG 本地知识库项目，用 FastAPI、React、Ollama、Qdrant 做的。"
        "工作经历暂时先不写。"
    )

    slots = fallback_extract_slots(message, template, {})

    assert slots == {
        "target_position": "AI Agent 开发岗位",
        "education": "浙江工商大学大数据专业本科，2023年毕业",
        "work_experience": "暂无",
        "project_experience": "我有一个 Agentic RAG 本地知识库项目，用 FastAPI、React、Ollama、Qdrant 做的，技术栈 FastAPI、React、Ollama、Qdrant",
    }


def test_chat_routes_resume_request(monkeypatch) -> None:
    state_store._STATE_STORE.clear()

    async def fail_if_called(_: str) -> str:
        raise AssertionError("call_llm should not be used for keyword-based routing")

    monkeypatch.setattr("app.main.call_llm", fail_if_called)

    response = client.post(
        "/chat",
        json={"session_id": None, "message": "帮我写简历"},
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
    assert "你想投什么岗位" in body["reply"]
    assert "请告诉我你的教育经历" in body["reply"]
    assert "请告诉我你的工作经历" in body["reply"]


def test_chat_collect_info_uses_extracted_slots_and_returns_style_options(monkeypatch) -> None:
    state_store._STATE_STORE.clear()

    state = get_or_create_state("resume-session")
    state.task_type = "resume"
    state.stage = "collect_info"
    save_state(state)

    raw_output = """
```json
{
  "updated_slots": {
    "target_position": "AI Agent 开发岗位",
    "education": "浙江工商大学大数据专业本科，2023年毕业",
    "work_experience": "暂无",
    "project_experience": "Agentic RAG 本地知识库项目，技术栈 FastAPI、React、Ollama、Qdrant"
  },
  "reason": "用户已提供完整基础信息"
}
```"""

    async def fake_extract_slots(message: str, task_template: dict, current_slots: dict) -> dict:
        assert message.startswith("我想投 AI Agent 开发岗位")
        assert task_template["task_type"] == "resume"
        assert current_slots == {}
        return {
            "updated_slots": {
                "target_position": "AI Agent 开发岗位",
                "education": "浙江工商大学大数据专业本科，2023年毕业",
                "work_experience": "暂无",
                "project_experience": "Agentic RAG 本地知识库项目，技术栈 FastAPI、React、Ollama、Qdrant",
            },
            "reason": "用户已提供完整基础信息",
            "raw_llm_output": raw_output,
            "parse_success": True,
        }

    monkeypatch.setattr("app.main.extract_slots", fake_extract_slots)

    response = client.post(
        "/chat",
        json={
            "session_id": "resume-session",
            "message": (
                "我想投 AI Agent 开发岗位，本科是浙江工商大学大数据专业，2023年毕业。"
                "我有一个 Agentic RAG 本地知识库项目，用 FastAPI、React、Ollama、Qdrant 做的。"
                "工作经历暂时先不写。"
            ),
        },
    )

    body = response.json()

    assert response.status_code == 200
    assert body["debug"]["slot_extractor_called"] is True
    assert body["debug"]["slot_extractor_parse_success"] is True
    assert body["debug"]["slot_extractor_raw_output"] == raw_output
    assert body["debug"]["slots_before"] == {}
    assert body["debug"]["slots_after"] == {
        "target_position": "AI Agent 开发岗位",
        "education": "浙江工商大学大数据专业本科，2023年毕业",
        "work_experience": "暂无",
        "project_experience": "Agentic RAG 本地知识库项目，技术栈 FastAPI、React、Ollama、Qdrant",
    }
    assert body["debug"]["extracted_slots"] == body["debug"]["slots_after"]
    assert body["debug"]["missing_slots"] == []
    assert body["debug"]["stage"] == "confirm_style"
    assert body["debug"]["options_count"] == 4
    assert len(body["options"]) == 4
    assert body["options"][0]["label"] == "简洁专业"
    assert "请确认简历风格" in body["reply"]


def test_chat_confirm_style_returns_options_when_message_not_matched() -> None:
    state_store._STATE_STORE.clear()

    state = get_or_create_state("style-session")
    state.task_type = "resume"
    state.stage = "confirm_style"
    save_state(state)

    response = client.post(
        "/chat",
        json={"session_id": "style-session", "message": "你帮我决定"},
    )

    body = response.json()

    assert response.status_code == 200
    assert body["debug"]["stage"] == "confirm_style"
    assert body["debug"]["options_count"] == 4
    assert body["debug"]["selected_style"] == ""
    assert len(body["options"]) == 4
    assert "我需要先确认简历风格" in body["reply"]


def test_chat_confirm_style_accepts_label_and_enters_draft() -> None:
    state_store._STATE_STORE.clear()

    state = get_or_create_state("style-picked-session")
    state.task_type = "resume"
    state.stage = "confirm_style"
    save_state(state)

    response = client.post(
        "/chat",
        json={"session_id": "style-picked-session", "message": "技术硬核"},
    )

    body = response.json()

    assert response.status_code == 200
    assert body["debug"]["stage"] == "draft"
    assert body["debug"]["selected_style"] == "technical_strong"
    assert body["debug"]["slots"]["style"] == "technical_strong"
    assert body["debug"]["options_count"] == 0
    assert body["options"] == []
    assert "技术硬核" in body["reply"]


def test_chat_uses_normal_llm_reply_after_collect_info(monkeypatch) -> None:
    state_store._STATE_STORE.clear()

    state = get_or_create_state("review-session")
    state.task_type = "resume"
    state.stage = "revise"
    save_state(state)

    raw_message = "请再精简一点"
    llm_reply = "我会进一步精简内容。"

    async def fake_call_llm(prompt: str) -> str:
        assert raw_message in prompt
        return llm_reply

    async def fail_if_called(*_: object, **__: object) -> dict:
        raise AssertionError("routing or extraction should not be called in revise stage")

    monkeypatch.setattr("app.main.call_llm", fake_call_llm)
    monkeypatch.setattr("app.main.detect_task_type", fail_if_called)
    monkeypatch.setattr("app.main.extract_slots", fail_if_called)

    response = client.post(
        "/chat",
        json={"session_id": "review-session", "message": raw_message},
    )

    body = response.json()

    assert response.status_code == 200
    assert body["reply"] == llm_reply
    assert body["debug"]["stage"] == "revise"
