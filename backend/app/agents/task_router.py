import json

from app.services.llm_provider import call_llm


_RESUME_KEYWORDS = ["\u7b80\u5386", "resume", "cv", "\u6c42\u804c", "\u5c97\u4f4d"]


def _unknown_result(reason: str) -> dict:
    return {
        "task_type": "unknown",
        "confidence": 0.0,
        "reason": reason,
    }


def _parse_router_result(raw_text: str) -> dict:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return _unknown_result("LLM \u8fd4\u56de\u65e0\u6cd5\u89e3\u6790\u4e3a JSON")

    task_type = data.get("task_type")
    confidence = data.get("confidence")
    reason = data.get("reason")

    if task_type not in {"resume", "unknown"}:
        return _unknown_result("LLM \u8fd4\u56de\u7684 task_type \u975e\u6cd5")

    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        return _unknown_result("LLM \u8fd4\u56de\u7684 confidence \u975e\u6cd5")

    if not isinstance(reason, str) or not reason.strip():
        return _unknown_result("LLM \u8fd4\u56de\u7684 reason \u975e\u6cd5")

    return {
        "task_type": task_type,
        "confidence": confidence_value,
        "reason": reason,
    }


async def detect_task_type(message: str) -> dict:
    lowered_message = message.lower()
    if any(keyword in lowered_message for keyword in _RESUME_KEYWORDS):
        return {
            "task_type": "resume",
            "confidence": 0.9,
            "reason": "\u547d\u4e2d\u7b80\u5386\u76f8\u5173\u5173\u952e\u8bcd",
        }

    prompt = (
        "\u4f60\u662f\u4efb\u52a1\u8bc6\u522b\u5668\u3002\u8bf7\u5224\u65ad\u7528\u6237\u60f3\u5b8c\u6210\u4ec0\u4e48\u4efb\u52a1\u3002\n"
        "\u5f53\u524d\u53ea\u652f\u6301\uff1a\n"
        "- resume\uff1a\u5199\u7b80\u5386\u3001\u4f18\u5316\u7b80\u5386\u3001\u6c42\u804c\u6750\u6599\n"
        "- unknown\uff1a\u5176\u4ed6\u4efb\u52a1\n\n"
        "\u8fd4\u56de JSON\uff1a\n"
        "{\n"
        '  "task_type": "resume \u6216 unknown",\n'
        '  "confidence": 0\u52301\u4e4b\u95f4\u7684\u5c0f\u6570,\n'
        '  "reason": "\u7b80\u77ed\u539f\u56e0"\n'
        "}\n\n"
        "\u4e0d\u8981\u8fd4\u56de\u89e3\u91ca\uff0c\u4e0d\u8981\u8fd4\u56de Markdown\uff0c\u53ea\u80fd\u8fd4\u56de JSON\u3002\n"
        f"\u7528\u6237\u8f93\u5165\uff1a{message}"
    )
    raw_result = await call_llm(prompt)
    return _parse_router_result(raw_result)
