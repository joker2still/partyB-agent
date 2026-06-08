# PartyB Agent

PartyB Agent is a full-stack prototype for a "vendor-style requirement handling agent".
Instead of jumping straight into task completion, the agent is designed to clarify scope,
ask follow-up questions, confirm choices, and then deliver work step by step.

Version 1 focuses on resume writing, while the backend structure is already split into
`agents`, `services`, and `templates` so it can later expand to PPTs, proposals, and
project planning workflows.

## Structure

```text
partyB-agent/
  backend/
  frontend/
  README.md
```

## Backend

1. Enter the backend directory:

```bash
cd backend
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Copy env settings:

```bash
copy .env.example .env
```

4. Start the API server:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Backend runs at [http://127.0.0.1:8000](http://127.0.0.1:8000).

## LLM Provider Config

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

If the LLM call fails, `/chat` returns:

```text
LLM 调用失败，请检查模型服务或 API 配置。
```

## LangChain Usage

The current backend only uses LangChain for:

- `PromptTemplate`
- `JsonOutputParser`

LangGraph is not in use yet. The workflow is still driven by the project's own
state machine and workflow controller.

## Frontend

1. Enter the frontend directory:

```bash
cd frontend
```

2. Install dependencies:

```bash
npm install
```

3. Start the dev server:

```bash
npm run dev
```

Frontend runs at [http://localhost:5173](http://localhost:5173).

## Test `/health`

Open [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) in a browser, or run:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

Run backend tests:

```bash
cd backend
pytest
```

## Test Chat In Browser

1. Start backend on `127.0.0.1:8000`.
2. Start frontend on `localhost:5173`.
3. Open [http://localhost:5173](http://localhost:5173).
4. Type a message and send it.
5. Frontend sends `POST http://127.0.0.1:8000/chat`.
6. Backend calls the configured LLM provider and returns:
   `session_id`, `reply`, and `debug`.

## API

### `GET /health`

Response:

```json
{"status":"ok"}
```

### `POST /chat`

Request:

```json
{
  "session_id": null,
  "message": "Help me write a product manager resume."
}
```

Response shape:

```json
{
  "session_id": "session-default",
  "reply": "I understand your request.",
  "debug": {
    "provider": "ollama",
    "model": "qwen2.5",
    "raw_message": "Help me write a product manager resume."
  }
}
```
