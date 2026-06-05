import { FormEvent, useState } from "react";

import { sendChatMessage } from "./api";

type Message = {
  role: "user" | "assistant";
  content: string;
};

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const trimmed = input.trim();
    if (!trimmed || isLoading) {
      return;
    }

    setMessages((prev) => [...prev, { role: "user", content: trimmed }]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await sendChatMessage({
        session_id: sessionId,
        message: trimmed,
      });

      setSessionId(response.session_id);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: response.reply },
      ]);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "请求失败，请检查后端是否启动。";

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `请求失败：${message}` },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="chat-card">
        <header className="chat-header">
          <h1>PartyB Agent</h1>
          <p>乙方型需求对接 Agent</p>
        </header>

        <main className="chat-messages">
          {messages.length === 0 ? (
            <div className="message assistant">
              请输入你的需求，我会先进行需求对接。
            </div>
          ) : (
            messages.map((message, index) => (
              <div key={`${message.role}-${index}`} className={`message ${message.role}`}>
                {message.content}
              </div>
            ))
          )}
        </main>

        <form className="chat-form" onSubmit={handleSubmit}>
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="输入你的需求"
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading}>
            {isLoading ? "发送中..." : "发送"}
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;
