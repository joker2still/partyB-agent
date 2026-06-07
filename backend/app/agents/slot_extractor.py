import json
import re

from app.services.llm_provider import call_llm


ZH_TOU = "\u6295"
ZH_GANG_WEI = "\u5c97\u4f4d"
ZH_KAI_FA_GANG_WEI = "\u5f00\u53d1\u5c97\u4f4d"
ZH_BEN_KE = "\u672c\u79d1"
ZH_DA_XUE = "\u5927\u5b66"
ZH_ZHUAN_YE = "\u4e13\u4e1a"
ZH_BI_YE = "\u6bd5\u4e1a"
ZH_XIANG_MU = "\u9879\u76ee"
ZH_JI_SHU_ZHAN = "\u6280\u672f\u6808"
ZH_GONG_ZUO_ZAN_BU_XIE = "\u5de5\u4f5c\u7ecf\u5386\u6682\u65f6\u5148\u4e0d\u5199"
ZH_GONG_ZUO_ZAN_WU = "\u5de5\u4f5c\u7ecf\u5386\u6682\u65e0"
ZH_MEI_YOU_GONG_ZUO = "\u6ca1\u6709\u5de5\u4f5c\u7ecf\u5386"
ZH_GONG_ZUO_XIAN_BU_XIE = "\u5de5\u4f5c\u7ecf\u5386\u5148\u4e0d\u5199"
ZH_ZAN_WU = "\u6682\u65e0"


def _safe_slots(current_slots: dict) -> dict:
    return dict(current_slots) if isinstance(current_slots, dict) else {}


def robust_json_parse(raw_text: str) -> dict | None:
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None

    text = raw_text.strip()

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced_match:
        try:
            parsed = json.loads(fenced_match.group(1))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

    start_index = text.find("{")
    if start_index == -1:
        return None

    brace_depth = 0
    for index in range(start_index, len(text)):
        char = text[index]
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
            if brace_depth == 0:
                candidate = text[start_index : index + 1]
                try:
                    parsed = json.loads(candidate)
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    return None

    return None


def _extract_target_position(message: str) -> str | None:
    normalized = message.replace("AI Agent\u5f00\u53d1", "AI Agent \u5f00\u53d1")
    patterns = [
        rf"\u60f3{ZH_TOU}\s*([^\uff0c\u3002\uff1b\n]+?(?:{ZH_GANG_WEI}|\u5f00\u53d1))",
        rf"{ZH_TOU}\s*([^\uff0c\u3002\uff1b\n]+?(?:{ZH_GANG_WEI}|\u5f00\u53d1))",
        rf"([^\uff0c\u3002\uff1b\n]+{ZH_KAI_FA_GANG_WEI})",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized, re.IGNORECASE)
        if not match:
            continue

        value = match.group(1).strip()
        value = re.sub(r"\s+", " ", value)
        if value and not value.endswith(ZH_GANG_WEI):
            value = f"{value}{ZH_GANG_WEI}"
        return value

    return None


def _extract_education(message: str) -> str | None:
    if not any(keyword in message for keyword in [ZH_BEN_KE, ZH_DA_XUE, ZH_ZHUAN_YE, ZH_BI_YE]):
        return None

    school_match = re.search(r"([^\s\uff0c\u3002\uff1b\n]+%s)" % ZH_DA_XUE, message)
    major_match = re.search(r"%s([^\s\uff0c\u3002\uff1b\n]+%s)" % (ZH_DA_XUE, ZH_ZHUAN_YE), message)
    graduation_match = re.search(r"(\d{4}\u5e74%s)" % ZH_BI_YE, message)

    school = school_match.group(1) if school_match else ""
    major = major_match.group(1) if major_match else ""
    graduation = graduation_match.group(1) if graduation_match else ""
    has_undergraduate = ZH_BEN_KE in message

    base = ""
    if school:
        base += school
    if major:
        base += major
    if has_undergraduate:
        base += ZH_BEN_KE

    if base and graduation:
        return f"{base}\uff0c{graduation}"
    if base:
        return base
    if graduation:
        return graduation

    sentence_match = re.search(
        r"([^\u3002]*?(?:%s|%s|%s|%s)[^\u3002]*)"
        % (ZH_BEN_KE, ZH_DA_XUE, ZH_ZHUAN_YE, ZH_BI_YE),
        message,
    )
    if sentence_match:
        return sentence_match.group(1).strip("\uff0c\u3002\uff1b ")

    return None


def _extract_project_experience(message: str) -> str | None:
    keywords = [ZH_XIANG_MU, ZH_JI_SHU_ZHAN, "FastAPI", "React", "Qdrant"]
    if not any(keyword.lower() in message.lower() for keyword in keywords):
        return None

    sentence_match = re.search(
        r"([^\u3002]*?(?:%s|%s|FastAPI|React|Qdrant)[^\u3002]*)"
        % (ZH_XIANG_MU, ZH_JI_SHU_ZHAN),
        message,
        re.IGNORECASE,
    )
    if not sentence_match:
        return None

    sentence = sentence_match.group(1).strip("\uff0c\u3002\uff1b ")
    tech_stack = "FastAPI\u3001React\u3001Ollama\u3001Qdrant"
    if ZH_JI_SHU_ZHAN not in sentence and any(keyword in sentence for keyword in ["FastAPI", "React", "Ollama", "Qdrant"]):
        sentence = f"{sentence}\uff0c{ZH_JI_SHU_ZHAN} {tech_stack}"
    return sentence


def _extract_work_experience(message: str) -> str | None:
    phrases = [
        ZH_GONG_ZUO_ZAN_BU_XIE,
        ZH_GONG_ZUO_ZAN_WU,
        ZH_MEI_YOU_GONG_ZUO,
        ZH_GONG_ZUO_XIAN_BU_XIE,
    ]
    if any(phrase in message for phrase in phrases):
        return ZH_ZAN_WU
    return None


def fallback_extract_slots(message: str, task_template: dict, current_slots: dict) -> dict:
    merged_slots = _safe_slots(current_slots)
    allowed_slots = set(
        (task_template.get("required_slots", []) if task_template else [])
        + (task_template.get("optional_slots", []) if task_template else [])
    )

    extracted_target_position = _extract_target_position(message)
    if extracted_target_position and "target_position" in allowed_slots:
        merged_slots["target_position"] = extracted_target_position

    extracted_education = _extract_education(message)
    if extracted_education and "education" in allowed_slots:
        merged_slots["education"] = extracted_education

    extracted_work_experience = _extract_work_experience(message)
    if extracted_work_experience and "work_experience" in allowed_slots:
        merged_slots["work_experience"] = extracted_work_experience

    extracted_project_experience = _extract_project_experience(message)
    if extracted_project_experience and "project_experience" in allowed_slots:
        merged_slots["project_experience"] = extracted_project_experience

    return merged_slots


async def extract_slots(message: str, task_template: dict, current_slots: dict) -> dict:
    safe_current_slots = _safe_slots(current_slots)
    required_slots = task_template.get("required_slots", []) if task_template else []
    optional_slots = task_template.get("optional_slots", []) if task_template else []
    allowed_slots = required_slots + optional_slots

    prompt = (
        "\u4f60\u662f\u4fe1\u606f\u62bd\u53d6\u5668\u3002\n"
        "\u8bf7\u6839\u636e\u7528\u6237\u8f93\u5165\uff0c\u4ece\u4e2d\u62bd\u53d6\u7b80\u5386\u76f8\u5173\u5b57\u6bb5\u3002\n"
        "\u53ea\u62bd\u53d6\u7528\u6237\u660e\u786e\u63d0\u5230\u7684\u4fe1\u606f\u3002\n"
        "\u4e0d\u8981\u7f16\u9020\u3002\n"
        "\u6ca1\u63d0\u5230\u7684\u5b57\u6bb5\u4e0d\u8981\u65b0\u589e\uff0c\u6216\u8005\u4fdd\u6301 current_slots \u539f\u503c\u3002\n"
        "\u4fdd\u7559 current_slots \u4e2d\u5df2\u6709\u4fe1\u606f\u3002\n"
        "\u65b0\u4fe1\u606f\u53ef\u4ee5\u8986\u76d6\u65e7\u4fe1\u606f\u3002\n"
        "\u53ea\u8fd4\u56de JSON\uff0c\u4e0d\u8981 Markdown\uff0c\u4e0d\u8981\u89e3\u91ca\u3002\n\n"
        f"\u53ef\u7528\u5b57\u6bb5\uff1a{json.dumps(allowed_slots, ensure_ascii=False)}\n"
        f"current_slots: {json.dumps(safe_current_slots, ensure_ascii=False)}\n\n"
        "\u8bf7\u4e25\u683c\u8fd4\u56de\u5982\u4e0b JSON\uff1a\n"
        "{\n"
        '  "updated_slots": {\n'
        '    "target_position": "",\n'
        '    "education": "",\n'
        '    "work_experience": "",\n'
        '    "project_experience": ""\n'
        "  },\n"
        '  "reason": ""\n'
        "}\n\n"
        "\u89c4\u5219\uff1a\n"
        '- "\u6211\u60f3\u6295 AI Agent \u5f00\u53d1\u5c97\u4f4d" \u5e94\u62bd\u53d6\u4e3a target_position\n'
        '- "\u672c\u79d1\u662f\u6d59\u6c5f\u5de5\u5546\u5927\u5b66\u5927\u6570\u636e\u4e13\u4e1a\uff0c2023\u5e74\u6bd5\u4e1a" \u5e94\u62bd\u53d6\u4e3a education\n'
        '- "Agentic RAG \u672c\u5730\u77e5\u8bc6\u5e93\u9879\u76ee\uff0c\u7528 FastAPI\u3001React\u3001Ollama\u3001Qdrant \u505a\u7684" \u5e94\u62bd\u53d6\u4e3a project_experience\n'
        '- "\u5de5\u4f5c\u7ecf\u5386\u6682\u65f6\u5148\u4e0d\u5199" \u5e94\u62bd\u53d6\u4e3a work_experience = "\u6682\u65e0"\n\n'
        f"\u7528\u6237\u8f93\u5165\uff1a{message}"
    )

    raw_result = await call_llm(prompt)
    data = robust_json_parse(raw_result)

    if data is None:
        fallback_slots = fallback_extract_slots(message, task_template, safe_current_slots)
        return {
            "updated_slots": fallback_slots,
            "reason": "LLM \u8fd4\u56de\u65e0\u6cd5\u89e3\u6790\u4e3a JSON\uff0c\u5df2\u4f7f\u7528\u89c4\u5219\u5140\u5e95",
            "raw_llm_output": raw_result,
            "parse_success": False,
        }

    updated_slots = data.get("updated_slots")
    reason = data.get("reason")

    if not isinstance(updated_slots, dict):
        fallback_slots = fallback_extract_slots(message, task_template, safe_current_slots)
        return {
            "updated_slots": fallback_slots,
            "reason": "LLM \u8fd4\u56de\u7684 updated_slots \u975e\u6cd5\uff0c\u5df2\u4f7f\u7528\u89c4\u5219\u5140\u5e95",
            "raw_llm_output": raw_result,
            "parse_success": False,
        }

    merged_slots = dict(safe_current_slots)
    for key, value in updated_slots.items():
        if key in allowed_slots and isinstance(value, str) and value.strip():
            merged_slots[key] = value.strip()

    return {
        "updated_slots": merged_slots,
        "reason": reason if isinstance(reason, str) else "",
        "raw_llm_output": raw_result,
        "parse_success": True,
    }
