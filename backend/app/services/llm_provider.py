import httpx

from app.core.config import get_settings


ERROR_MESSAGE = (
    "LLM \u8c03\u7528\u5931\u8d25\uff0c"
    "\u8bf7\u68c0\u67e5\u6a21\u578b\u670d\u52a1\u6216 API \u914d\u7f6e\u3002"
)


async def _call_ollama(prompt: str) -> str:
    settings = get_settings()
    url = f"{settings.ollama_base_url.rstrip('/')}/api/generate"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            url,
            json={
                "model": settings.ollama_model,
                "prompt": prompt,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()

    content = data.get("response")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Missing Ollama response field.")

    return content


async def _call_api(prompt: str) -> str:
    settings = get_settings()
    if not settings.api_base_url or not settings.api_key or not settings.api_model:
        raise ValueError("API provider config is incomplete.")

    url = f"{settings.api_base_url.rstrip('/')}/chat/completions"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.api_model,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        data = response.json()

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Missing API choices field.")

    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Missing API message content.")

    return content


async def call_llm(prompt: str) -> str:
    settings = get_settings()

    try:
        if settings.llm_provider == "api":
            return await _call_api(prompt)

        return await _call_ollama(prompt)
    except Exception:
        return ERROR_MESSAGE
