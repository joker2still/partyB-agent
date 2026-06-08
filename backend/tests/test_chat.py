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
        "project_experience": (
            "我有一个 Agentic RAG 本地知识库项目，用 FastAPI、React、Ollama、Qdrant 做的，"
            "技术栈 FastAPI、React、Ollama、Qdrant"
        ),
    }


def test_main_delegates_to_workflow_controller(monkeypatch) -> None:
    state_store._STATE_STORE.clear()

    async def fake_handle_chat_turn(state, user_message: str) -> dict:
        assert state.session_id
        assert user_message == "测试入口"
        return {
            "reply": "已进入控制器",
            "options": [{"label": "A", "value": "a"}],
            "debug": {"workflow_action": "detect_task"},
        }

    monkeypatch.setattr("app.main.handle_chat_turn", fake_handle_chat_turn)

    response = client.post("/chat", json={"session_id": None, "message": "测试入口"})
    body = response.json()

    assert response.status_code == 200
    assert body["reply"] == "已进入控制器"
    assert body["options"] == [{"label": "A", "value": "a"}]
    assert body["debug"]["workflow_action"] == "detect_task"
    assert body["debug"]["history_count"] == 2


def test_chat_routes_resume_request() -> None:
    state_store._STATE_STORE.clear()

    response = client.post("/chat", json={"session_id": None, "message": "帮我写简历"})
    body = response.json()

    assert response.status_code == 200
    assert body["debug"]["task_type"] == "resume"
    assert body["debug"]["stage"] == "collect_info"
    assert body["debug"]["workflow_stage_before"] == "start"
    assert body["debug"]["workflow_stage_after"] == "collect_info"
    assert body["debug"]["workflow_action"] == "detect_task"
    assert body["debug"]["langgraph_enabled"] is True
    assert body["debug"]["langgraph_real_workflow"] is True
    assert "你想投什么岗位" in body["reply"]


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

    monkeypatch.setattr("app.agents.langgraph_workflow.extract_slots", fake_extract_slots)

    response = client.post(
        "/chat",
        json={
            "session_id": "resume-session",
            "message": "我想投 AI Agent 开发岗位，本科是浙江工商大学大数据专业，2023年毕业。",
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["debug"]["slot_extractor_called"] is True
    assert body["debug"]["slot_extractor_parse_success"] is True
    assert body["debug"]["slot_extractor_raw_output"] == raw_output
    assert body["debug"]["missing_slots"] == []
    assert body["debug"]["stage"] == "confirm_style"
    assert body["debug"]["workflow_stage_before"] == "collect_info"
    assert body["debug"]["workflow_stage_after"] == "confirm_style"
    assert body["debug"]["workflow_action"] == "collect_slots"
    assert len(body["options"]) == 4


def test_chat_confirm_style_returns_options_when_message_not_matched() -> None:
    state_store._STATE_STORE.clear()

    state = get_or_create_state("style-session")
    state.task_type = "resume"
    state.stage = "confirm_style"
    save_state(state)

    response = client.post("/chat", json={"session_id": "style-session", "message": "你帮我决定"})
    body = response.json()

    assert response.status_code == 200
    assert body["debug"]["stage"] == "confirm_style"
    assert body["debug"]["workflow_action"] == "confirm_style"
    assert len(body["options"]) == 4
    assert "我需要先确认简历风格" in body["reply"]


def test_chat_confirm_style_accepts_label_generates_draft_and_enters_revise(monkeypatch) -> None:
    state_store._STATE_STORE.clear()

    state = get_or_create_state("style-picked-session")
    state.task_type = "resume"
    state.stage = "confirm_style"
    save_state(state)

    async def fake_generate_resume_draft(slots: dict, task_template: dict) -> dict:
        assert slots["style"] == "technical_strong"
        return {
            "draft": "# 简历初稿\n\n## 项目经历\n- Agentic RAG 项目",
            "reason": "generated",
        }

    monkeypatch.setattr("app.agents.langgraph_workflow.generate_resume_draft", fake_generate_resume_draft)

    response = client.post("/chat", json={"session_id": "style-picked-session", "message": "技术硬核"})
    body = response.json()

    assert response.status_code == 200
    assert body["debug"]["stage"] == "revise"
    assert body["debug"]["selected_style"] == "technical_strong"
    assert body["debug"]["draft_generated"] is True
    assert body["debug"]["workflow_action"] == "generate_draft"
    assert "下面是第一版简历初稿" in body["reply"]


def test_chat_generates_draft_when_stage_is_draft(monkeypatch) -> None:
    state_store._STATE_STORE.clear()

    state = get_or_create_state("draft-session")
    state.task_type = "resume"
    state.stage = "draft"
    state.slots["style"] = "simple_professional"
    save_state(state)

    async def fake_generate_resume_draft(slots: dict, task_template: dict) -> dict:
        return {
            "draft": "# 简历初稿\n\n## 教育经历\n- 待补充",
            "reason": "generated",
        }

    monkeypatch.setattr("app.agents.langgraph_workflow.generate_resume_draft", fake_generate_resume_draft)

    response = client.post("/chat", json={"session_id": "draft-session", "message": "继续生成"})
    body = response.json()

    assert response.status_code == 200
    assert body["debug"]["stage"] == "revise"
    assert body["debug"]["workflow_action"] == "generate_draft"
    assert "下面是第一版简历初稿" in body["reply"]


def test_chat_revises_resume_draft(monkeypatch) -> None:
    state_store._STATE_STORE.clear()

    state = get_or_create_state("revise-session")
    state.task_type = "resume"
    state.stage = "revise"
    state.slots["resume_draft"] = "# 简历初稿\n\n## 项目经历\n- 原始版本"
    save_state(state)

    async def fake_revise_resume_draft(current_draft: str, user_feedback: str, slots: dict) -> dict:
        assert user_feedback == "更突出项目"
        return {
            "revised_draft": "# 简历初稿\n\n## 项目经历\n- 强化后的项目亮点",
            "reason": "revised",
        }

    monkeypatch.setattr("app.agents.langgraph_workflow.revise_resume_draft", fake_revise_resume_draft)

    response = client.post("/chat", json={"session_id": "revise-session", "message": "更突出项目"})
    body = response.json()

    assert response.status_code == 200
    assert body["debug"]["revise_called"] is True
    assert body["debug"]["stage"] == "revise"
    assert body["debug"]["workflow_action"] == "revise_draft"
    assert "我已经根据你的意见修改了简历" in body["reply"]


def test_chat_confirm_final_with_missing_export_info_enters_collect_export_info() -> None:
    state_store._STATE_STORE.clear()

    state = get_or_create_state("export-info-session")
    state.task_type = "resume"
    state.stage = "revise"
    state.slots["resume_draft"] = "# 简历初稿"
    save_state(state)

    response = client.post("/chat", json={"session_id": "export-info-session", "message": "确认最终版"})
    body = response.json()

    assert response.status_code == 200
    assert body["debug"]["stage"] == "collect_export_info"
    assert body["debug"]["export_info_missing"] == ["name", "phone", "email", "age", "location"]
    assert body["debug"]["export_info_collected"] is False
    assert body["debug"]["workflow_action"] == "collect_export_info"
    assert "为了生成完整 Word 简历" in body["reply"]


def test_chat_collect_export_info_completes_and_enters_final() -> None:
    state_store._STATE_STORE.clear()

    state = get_or_create_state("collect-export-session")
    state.task_type = "resume"
    state.stage = "collect_export_info"
    state.slots["resume_draft"] = "# 简历初稿"
    save_state(state)

    response = client.post(
        "/chat",
        json={
            "session_id": "collect-export-session",
            "message": "我叫张三，手机号 13800000000，邮箱 zhangsan@example.com，今年 24 岁，现居杭州。",
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["debug"]["stage"] == "final"
    assert body["debug"]["export_info_missing"] == []
    assert body["debug"]["export_info_collected"] is True
    assert body["debug"]["workflow_action"] == "final_confirmed"
    assert body["debug"]["slots"]["name"] == "张三"
    assert body["debug"]["slots"]["phone"] == "13800000000"
    assert body["debug"]["slots"]["email"] == "zhangsan@example.com"
    assert body["debug"]["slots"]["age"] == "24"
    assert body["debug"]["slots"]["location"] == "杭州"
    assert "个人信息已补充完整" in body["reply"]


def test_chat_collect_export_info_supports_compact_age_and_phone_syntax() -> None:
    state_store._STATE_STORE.clear()

    state = get_or_create_state("compact-export-session")
    state.task_type = "resume"
    state.stage = "collect_export_info"
    state.slots["resume_draft"] = "# 简历初稿"
    state.slots["name"] = "张文宣"
    state.slots["email"] = "z694908768@gmail.com"
    save_state(state)

    response = client.post(
        "/chat",
        json={
            "session_id": "compact-export-session",
            "message": "手机19975267684，年龄26，现居杭州。",
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["debug"]["stage"] == "final"
    assert body["debug"]["export_info_missing"] == []
    assert body["debug"]["export_info_collected"] is True
    assert body["debug"]["slots"]["phone"] == "19975267684"
    assert body["debug"]["slots"]["age"] == "26"
    assert body["debug"]["slots"]["location"] == "杭州"
    assert "个人信息已补充完整" in body["reply"]


def test_chat_confirm_final_with_complete_export_info_enters_final() -> None:
    state_store._STATE_STORE.clear()

    state = get_or_create_state("final-session")
    state.task_type = "resume"
    state.stage = "revise"
    state.slots.update(
        {
            "resume_draft": "# 简历初稿",
            "name": "张三",
            "phone": "13800000000",
            "email": "zhangsan@example.com",
            "age": "24",
            "location": "杭州",
        }
    )
    save_state(state)

    response = client.post("/chat", json={"session_id": "final-session", "message": "确认最终版"})
    body = response.json()

    assert response.status_code == 200
    assert body["debug"]["stage"] == "final"
    assert body["debug"]["export_info_collected"] is True
    assert body["debug"]["workflow_action"] == "final_confirmed"
    assert "现在可以导出 Word 简历" in body["reply"]


def test_chat_resets_unknown_stage_to_start() -> None:
    state_store._STATE_STORE.clear()

    state = get_or_create_state("broken-session")
    state.task_type = "resume"
    state.stage = "broken"
    save_state(state)

    response = client.post("/chat", json={"session_id": "broken-session", "message": "继续"})
    body = response.json()

    assert response.status_code == 200
    assert body["debug"]["stage"] == "start"
    assert body["debug"]["workflow_stage_before"] == "broken"
    assert body["debug"]["workflow_stage_after"] == "start"
    assert body["debug"]["workflow_action"] == "clarify_task"
    assert body["reply"] == "当前流程状态异常，我会重新回到需求确认阶段。"
