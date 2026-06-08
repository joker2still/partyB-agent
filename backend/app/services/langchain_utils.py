import json
import re

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate


def build_json_prompt(template: str, variables: dict) -> str:
    prompt_template = PromptTemplate.from_template(template)
    return prompt_template.format(**variables)


def _robust_json_parse(raw: str) -> dict:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("empty json output")

    text = raw.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced_match:
        try:
            parsed = json.loads(fenced_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    start_index = text.find("{")
    if start_index == -1:
        raise ValueError("json object not found")

    brace_depth = 0
    for index in range(start_index, len(text)):
        char = text[index]
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
            if brace_depth == 0:
                candidate = text[start_index : index + 1]
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
                break

    raise ValueError("unable to parse json output")


def parse_json_output(raw: str) -> dict:
    parser = JsonOutputParser()
    try:
        parsed = parser.parse(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return _robust_json_parse(raw)
