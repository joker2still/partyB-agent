from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.services import state_store
from app.services.state_store import get_or_create_state, save_state


client = TestClient(app)


def test_export_resume_returns_404_for_missing_session() -> None:
    state_store._STATE_STORE.clear()

    response = client.post("/export/resume", json={"session_id": "missing-session"})

    assert response.status_code == 404


def test_export_resume_returns_400_without_resume_draft() -> None:
    state_store._STATE_STORE.clear()

    state = get_or_create_state("empty-draft-session")
    save_state(state)

    response = client.post("/export/resume", json={"session_id": "empty-draft-session"})

    assert response.status_code == 400


def test_export_resume_returns_docx_file() -> None:
    state_store._STATE_STORE.clear()

    state = get_or_create_state("final-export-session")
    state.stage = "final"
    state.slots["resume_draft"] = "# 张三\n\n## 项目经历\n- Agentic RAG 项目"
    save_state(state)

    response = client.post("/export/resume", json={"session_id": "final-export-session"})

    assert response.status_code == 200
    assert (
        response.headers["content-type"]
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert "resume_final-export-session.docx" in response.headers["content-disposition"]
    assert response.content[:2] == b"PK"

    exported_path = Path("backend/exports/resume_final-export-session.docx")
    assert exported_path.exists()


def test_export_resume_supports_structured_slots() -> None:
    state_store._STATE_STORE.clear()

    state = get_or_create_state("structured-export-session")
    state.stage = "final"
    state.slots.update(
        {
            "name": "张三",
            "phone": "13800000000",
            "email": "zhangsan@example.com",
            "location": "杭州",
            "target_position": "AI Agent 开发岗位",
            "education": "浙江工商大学，大数据专业，本科，2023年毕业",
            "skills": "- FastAPI\n- React\n- Qdrant",
            "project_experience": "- Agentic RAG 本地知识库项目",
            "work_experience": "暂无",
            "self_evaluation": "具备较强的工程落地能力。",
            "resume_draft": "# 张三\n\n## 项目经历\n- Agentic RAG 本地知识库项目",
        }
    )
    save_state(state)

    response = client.post("/export/resume", json={"session_id": "structured-export-session"})

    assert response.status_code == 200
    assert "resume_structured-export-session.docx" in response.headers["content-disposition"]
    assert response.content[:2] == b"PK"
