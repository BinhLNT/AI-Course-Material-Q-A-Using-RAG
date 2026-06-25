"""Trái tim của RAG: lấy câu hỏi -> tìm đoạn liên quan -> nhờ LLM trả lời.

Đây là chế độ "basic" (tìm 1 lần), đồng thời cung cấp các HÀM DÙNG CHUNG cho
chế độ agentic (app/agent.py) và tools (app/tools.py):
- _generate()             : gọi LLM (tự theo nhà cung cấp đã chọn).
- contextualize_question(): biến câu hỏi nối tiếp thành câu hỏi độc lập (nhớ hội thoại).
- retrieve()              : truy hồi + xếp hạng lại (re-ranking).
- _build_context()        : ghép các đoạn thành ngữ cảnh có đánh số nguồn.

Phần SINH câu trả lời hỗ trợ 2 nhà cung cấp (chọn trong .env qua LLM_PROVIDER):
- "openrouter": API tương thích OpenAI, có nhiều model MIỄN PHÍ (mặc định).
- "anthropic" : Claude (cần ANTHROPIC_API_KEY).

VÌ SAO "grounding" (chỉ trả lời dựa trên tài liệu) lại quan trọng?
- Tránh "bịa" (hallucination). Ta yêu cầu model CHỈ dùng thông tin trong các đoạn
  được cung cấp và TRÍCH DẪN nguồn -> học sinh tin tưởng và kiểm chứng được.
"""
from __future__ import annotations

from functools import lru_cache

from app import config, embeddings, rerank, vector_store

SYSTEM_PROMPT = """Bạn là trợ giảng AI, giúp học sinh hiểu tài liệu học tập.
Quy tắc bắt buộc:
- CHỈ trả lời dựa trên phần "NGỮ CẢNH" được cung cấp bên dưới.
- Nếu ngữ cảnh KHÔNG chứa thông tin để trả lời, hãy nói rõ: "Tôi không tìm thấy thông tin này trong tài liệu." Tuyệt đối không bịa.
- Trả lời bằng ĐÚNG ngôn ngữ của câu hỏi (hỏi tiếng Việt thì trả lời tiếng Việt).
- Trình bày rõ ràng, dễ hiểu; có thể dùng gạch đầu dòng hoặc ví dụ.
- Khi dùng thông tin từ một nguồn, ghi chú nguồn theo dạng [Nguồn N]."""


# ---------------------------------------------------------------------------
# Lớp SINH câu trả lời (2 nhà cung cấp). Client tạo "lười" qua lru_cache.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _openai_client():
    # OpenRouter tương thích OpenAI: chỉ cần đổi base_url + api key là dùng được.
    from openai import OpenAI

    return OpenAI(api_key=config.OPENROUTER_API_KEY, base_url=config.OPENROUTER_BASE_URL)


@lru_cache(maxsize=1)
def _anthropic_client():
    from anthropic import Anthropic

    return Anthropic()  # tự đọc ANTHROPIC_API_KEY từ biến môi trường


def _generate(system_prompt: str, user_prompt: str) -> str:
    """Gửi prompt tới LLM đang chọn và trả về văn bản câu trả lời."""
    if config.LLM_PROVIDER == "anthropic":
        resp = _anthropic_client().messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=config.MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    resp = _openai_client().chat.completions.create(
        model=config.OPENROUTER_MODEL,
        max_tokens=config.MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Nhớ hội thoại (multi-turn)
# ---------------------------------------------------------------------------
def history_block(history: list[dict] | None) -> str:
    """Định dạng vài lượt hội thoại gần nhất để đưa vào prompt."""
    if not history:
        return ""
    recent = history[-(config.HISTORY_TURNS * 2):]
    lines = []
    for turn in recent:
        who = "Học sinh" if turn.get("role") == "user" else "Trợ giảng"
        lines.append(f"{who}: {turn.get('content', '')}")
    return "LỊCH SỬ HỘI THOẠI GẦN ĐÂY:\n" + "\n".join(lines)


def contextualize_question(history: list[dict] | None, question: str) -> str:
    """Biến câu hỏi nối tiếp ("vậy còn cái đó?") thành câu hỏi ĐỘC LẬP để tìm kiếm.

    VÌ SAO? Embedding của "cái đó là gì?" gần như vô nghĩa khi tìm kiếm. Phải thay
    đại từ bằng danh từ cụ thể dựa trên lịch sử thì mới truy hồi đúng."""
    if not history:
        return question
    prompt = (
        history_block(history) + "\n\n"
        "Dựa trên lịch sử trên, hãy VIẾT LẠI câu hỏi mới nhất thành một câu hỏi ĐỘC LẬP, "
        "đầy đủ ngữ cảnh (thay các từ như 'nó', 'cái đó', 'vậy còn'... bằng danh từ cụ thể). "
        "Chỉ in ra câu hỏi đã viết lại, không thêm lời giải thích.\n\n"
        f"CÂU HỎI MỚI: {question}"
    )
    rewritten = _generate("Bạn viết lại câu hỏi cho độc lập về ngữ cảnh.", prompt).strip()
    return rewritten or question


# ---------------------------------------------------------------------------
# Truy hồi (có re-ranking)
# ---------------------------------------------------------------------------
def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    """Lấy nhiều ứng viên bằng embedding rồi xếp hạng lại, giữ `top_k` đoạn tốt nhất."""
    top_k = top_k or config.TOP_K
    candidates = config.RERANK_CANDIDATES if config.RERANK_ENABLED else top_k
    hits = vector_store.query(embeddings.embed_query(query), candidates)
    return rerank.rerank(query, hits, top_k)


def _build_context(hits: list[dict]) -> str:
    """Ghép các đoạn tìm được thành một khối ngữ cảnh có đánh số nguồn."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        label = f"[Nguồn {i}: {hit['source']}]"
        blocks.append(f"{label}\n{hit['text']}")
    return "\n\n---\n\n".join(blocks)


def _make_sources(hits: list[dict]) -> list[dict]:
    return [
        {
            "label": f"Nguồn {i}",
            "source": hit["source"],
            "chunk_index": hit["chunk_index"],
            "snippet": hit["text"][:200] + ("…" if len(hit["text"]) > 200 else ""),
        }
        for i, hit in enumerate(hits, start=1)
    ]


def answer_question(question: str, history: list[dict] | None = None) -> dict:
    """Chế độ basic: nhớ hội thoại -> tìm 1 lần (có rerank) -> trả lời."""
    history = history or []

    # Nhớ hội thoại: viết lại câu hỏi cho độc lập rồi mới tìm kiếm.
    standalone = contextualize_question(history, question)
    hits = retrieve(standalone, config.TOP_K)

    if not hits:
        return {
            "answer": "Chưa có tài liệu nào được nạp. Hãy tải tài liệu lên trước nhé.",
            "sources": [],
        }

    context = _build_context(hits)
    hist = history_block(history)
    user_prompt = (
        (hist + "\n\n" if hist else "")
        + f"NGỮ CẢNH (các đoạn trích từ tài liệu):\n\n{context}\n\n"
        + f"---\n\nCÂU HỎI CỦA HỌC SINH: {question}\n\n"
        + "Hãy trả lời dựa trên ngữ cảnh ở trên."
    )
    answer = _generate(SYSTEM_PROMPT, user_prompt)
    return {"answer": answer, "sources": _make_sources(hits)}
