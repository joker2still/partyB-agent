import json
from pathlib import Path


_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


def load_task_template(task_type: str) -> dict:
    template_path = _TEMPLATE_DIR / f"{task_type}.json"
    if not template_path.exists():
        return {}

    try:
        with template_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(data, dict):
        return {}

    return data
