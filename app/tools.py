"""Tool-calling agent (chuẩn OpenAI): LLM TỰ quyết định khi nào & tìm gì.

Khác gì chế độ "agentic"?
- agentic: CODE của ta điều phối từng bước (viết lại -> tìm -> đánh giá).
- tools:   LLM được trao một "công cụ" search_course_materials và TỰ quyết định
           gọi bao nhiêu lần, với truy vấn gì, rồi tự dừng để trả lời.

Chạy trên model HỖ TRỢ tool-calling: nhiều model OpenRouter (kể cả bản free như
Llama 3.3, Qwen, DeepSeek) và Ollama chạy local. Đây là giao thức "tools" của
OpenAI; với nhà cung cấp Anthropic (Claude) ta chuyển về chế độ agentic cho đồng nhất.
"""
from __future__ import annotations

import json

from app import agent, config, embeddings, rag, rerank, vector_store

TOOLS_SYSTEM_PROMPT = """Bạn là trợ giảng AI, giúp học sinh hiểu tài liệu học tập.
Bạn có công cụ `search_course_materials` để tra cứu trong tài liệu đã được upload.
Quy tắc:
- Hãy GỌI công cụ (một hoặc nhiều lần, với truy vấn khác nhau) để thu thập thông tin TRƯỚC khi trả lời.
- CHỈ trả lời dựa trên nội dung công cụ trả về. Nếu không tìm thấy, nói rõ: "Tôi không tìm thấy thông tin này trong tài liệu." Không bịa.
- Trả lời bằng đúng ngôn ngữ của câu hỏi, trình bày rõ ràng, và trích nguồn theo dạng [Nguồn N]."""

# Khai báo "công cụ" theo chuẩn OpenAI function-calling.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_course_materials",
            "description": "Tìm các đoạn liên quan trong tài liệu học tập đã được upload.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Truy vấn tìm kiếm (tiếng Việt hoặc tiếng Anh).",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


def _search(query: str) -> list[dict]:
    """Thực thi việc tìm kiếm khi LLM gọi công cụ: truy hồi + xếp hạng lại."""
    candidates = config.RERANK_CANDIDATES if config.RERANK_ENABLED else config.TOP_K
    hits = vector_store.query(embeddings.embed_query(query), candidates)
    return rerank.rerank(query, hits, config.TOP_K)


def _format_hits(hits: list[dict]) -> str:
    if not hits:
        return "Không tìm thấy đoạn nào liên quan trong tài liệu."
    return "\n\n---\n\n".join(
        f"[{h['source']} #{h['chunk_index']}]\n{h['text']}" for h in hits
    )


def _format_sources(sources: dict) -> list[dict]:
    out = []
    for i, hit in enumerate(sources.values(), start=1):
        out.append(
            {
                "label": f"Nguồn {i}",
                "source": hit["source"],
                "chunk_index": hit["chunk_index"],
                "snippet": hit["text"][:200] + ("…" if len(hit["text"]) > 200 else ""),
            }
        )
    return out


def answer_question(question: str, history: list[dict] | None = None) -> dict:
    history = history or []

    # Claude dùng giao thức tool riêng -> dùng chế độ agentic cho đồng nhất.
    if config.LLM_PROVIDER == "anthropic":
        result = agent.answer_question(question, history)
        result.setdefault("steps", []).insert(
            0, "(Chế độ tools cần API chuẩn OpenAI; với Claude đã dùng agentic.)"
        )
        return result

    client = rag._openai_client()

    # Dựng hội thoại: system + lịch sử gần đây + câu hỏi mới.
    messages: list[dict] = [{"role": "system", "content": TOOLS_SYSTEM_PROMPT}]
    for turn in history[-(config.HISTORY_TURNS * 2):]:
        role, content = turn.get("role"), turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})

    steps: list[str] = []
    sources: dict = {}  # khử trùng nguồn theo (file, chunk)

    try:
        for _ in range(config.TOOLS_MAX_CALLS):
            resp = client.chat.completions.create(
                model=config.OPENROUTER_MODEL,
                max_tokens=config.MAX_TOKENS,
                messages=messages,
                tools=TOOLS,
            )
            msg = resp.choices[0].message
            tool_calls = msg.tool_calls or []

            # Không gọi công cụ nữa -> đây là câu trả lời cuối.
            if not tool_calls:
                return {
                    "answer": msg.content or "",
                    "sources": _format_sources(sources),
                    "steps": steps,
                }

            # Ghi lại message assistant (kèm tool_calls) để giữ đúng luồng hội thoại.
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            # Thực thi từng lời gọi công cụ rồi trả kết quả về cho LLM.
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                query = (args.get("query") or "").strip() or question
                steps.append(f'🔎 search_course_materials("{query}")')

                hits = _search(query)
                for h in hits:
                    sources[(h["source"], h["chunk_index"])] = h

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": _format_hits(hits),
                    }
                )

        # Hết lượt gọi công cụ -> ép trả lời bằng văn bản (gọi lần cuối, KHÔNG kèm tools).
        steps.append("Đã đủ lượt tìm kiếm -> tổng hợp câu trả lời.")
        final = client.chat.completions.create(
            model=config.OPENROUTER_MODEL,
            max_tokens=config.MAX_TOKENS,
            messages=messages,
        )
        return {
            "answer": final.choices[0].message.content or "",
            "sources": _format_sources(sources),
            "steps": steps,
        }

    except Exception as exc:  # noqa: BLE001
        # Model được chọn có thể không hỗ trợ tool-calling.
        return {
            "answer": (
                f"Model hiện tại có thể không hỗ trợ tool-calling ({exc}). "
                "Hãy chọn model có 'tools' trên OpenRouter (vd meta-llama/llama-3.3-70b-instruct:free), "
                "hoặc đặt RAG_MODE=agentic."
            ),
            "sources": _format_sources(sources),
            "steps": steps,
        }
