from pathlib import Path

from docx import Document


EXPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "exports"


def _sanitize_session_id(session_id: str) -> str:
    cleaned = "".join(char for char in session_id if char.isalnum() or char in {"-", "_"})
    return cleaned or "default"


def export_resume_docx(markdown_text: str, session_id: str) -> str:
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"resume_{_sanitize_session_id(session_id)}.docx"
    output_path = EXPORTS_DIR / filename

    document = Document()

    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("# "):
            document.add_heading(line[2:].strip(), level=1)
            continue

        if line.startswith("## "):
            document.add_heading(line[3:].strip(), level=2)
            continue

        if line.startswith("- "):
            document.add_paragraph(line[2:].strip(), style="List Bullet")
            continue

        if line.startswith("\u00b7 "):
            document.add_paragraph(line[2:].strip(), style="List Bullet")
            continue

        document.add_paragraph(line)

    document.save(output_path)
    return str(output_path)
