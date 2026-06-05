type ChatRequest = {
  session_id: string | null;
  message: string;
};

type ChatResponse = {
  session_id: string;
  reply: string;
  debug: Record<string, unknown>;
};

const API_URL = "http://127.0.0.1:8000/chat";

export async function sendChatMessage(payload: ChatRequest): Promise<ChatResponse> {
  const response = await fetch(API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }

  return response.json() as Promise<ChatResponse>;
}
