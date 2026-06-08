export type ChatRequest = {
  session_id: string | null;
  message: string;
};

export type ChatOption = {
  label: string;
  value: string;
  description: string;
};

export type ChatResponse = {
  session_id: string;
  reply: string;
  debug: Record<string, unknown>;
  options: ChatOption[];
};

export type ExportResult = {
  blob: Blob;
  filename: string;
};

const CHAT_API_URL = "http://127.0.0.1:8000/chat";
const EXPORT_API_URL = "http://127.0.0.1:8000/export/resume";

function getFilenameFromDisposition(contentDisposition: string | null): string {
  if (!contentDisposition) {
    return "resume.docx";
  }

  const utf8Match = contentDisposition.match(/filename\*=UTF-8''([^;]+)/i);
  if (utf8Match?.[1]) {
    return decodeURIComponent(utf8Match[1]);
  }

  const filenameMatch = contentDisposition.match(/filename="?([^"]+)"?/i);
  if (filenameMatch?.[1]) {
    return filenameMatch[1];
  }

  return "resume.docx";
}

export async function sendChatMessage(payload: ChatRequest): Promise<ChatResponse> {
  const response = await fetch(CHAT_API_URL, {
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

export async function exportResumeDocx(sessionId: string): Promise<ExportResult> {
  const response = await fetch(EXPORT_API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ session_id: sessionId }),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }

  return {
    blob: await response.blob(),
    filename: getFilenameFromDisposition(response.headers.get("Content-Disposition")),
  };
}
