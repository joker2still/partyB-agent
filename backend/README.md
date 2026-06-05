# Backend

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Environment

Copy `.env.example` to `.env`.

Use Ollama:

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5
```

Use an OpenAI-compatible API:

```bash
LLM_PROVIDER=api
API_BASE_URL=https://your-api-host/v1
API_KEY=your_api_key
API_MODEL=your_model_name
```

## Run

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Test

```bash
pytest
```
