"""Agentic RAG: vòng lặp TỰ ĐIỀU HƯỚNG việc tìm kiếm.

Khác gì RAG cơ bản (app/rag.py)?
- Cơ bản:  hỏi -> tìm 1 lần -> trả lời.
- Agentic: hỏi -> (1) VIẾT LẠI / TÁCH câu hỏi thành nhiều truy vấn
                 -> (2) TRUY HỒI rồi TỰ ĐÁNH GIÁ "đã đủ thông tin chưa?"
                 -> (3) NẾU CHƯA, sinh truy vấn BỔ SUNG và tìm tiếp (lặp)
                 -> (4) TỔNG HỢP câu trả lời cuối + trích nguồn.

Có nhớ hội thoại (viết lại câu hỏi nối tiếp) và re-ranking (xếp hạng lại đoạn),
tái sử dụng các hàm dùng chung trong app/rag.py.
"""
from __future__ import annotations

import re

from app import config, embeddings, rag, rerank, vector_store


def _rewrite_queries(question: str) -> list[str]:
    """Bước 1: tách câu hỏi thành 1..N truy vấn tìm kiếm ngắn gọn."""
    prompt = (
        "Bạn là trợ lý tìm kiếm tài liệu. Từ câu hỏi của học sinh, hãy tạo từ 1 đến "
        f"{config.AGENT_MAX_QUERIES} TRUY VẤN tìm kiếm ngắn gọn, mỗi truy vấn tập trung "
        "vào MỘT khía cạnh cần tra trong tài liệu.\n"
        "Chỉ in ra các truy vấn, MỖI DÒNG MỘT truy vấn, không đánh số, không giải thích.\n\n"
        f"Câu hỏi: {question}"
    )
    text = rag._generate("Bạn tạo các truy vấn tìm kiếm tốt và ngắn gọn.", prompt)

    queries: list[str] = []
    for line in text.splitlines():
        # bỏ tiền tố kiểu "1.", "- ", "* ", "• " ở đầu dòng
        cleaned = re.sub(r"^[\s\-\*•\d\.\)]+", "", line).strip()
        if cleaned:
            queries.append(cleaned)

    queries = queries[: config.AGENT_MAX_QUERIES]
    return queries or [question]  # fallback an toàn nếu model trả lời lạ


def _decide(question: str, context: str) -> tuple[bool, str]:
    """Bước tự đánh giá: ngữ cảnh đã đủ chưa? Trả về (đã_đủ, truy_vấn_bổ_sung)."""
    prompt = (
        "Dưới đây là NGỮ CẢNH đã thu thập từ tài liệu để trả lời câu hỏi.\n\n"
        f"NGỮ CẢNH:\n{context}\n\n"
        f"CÂU HỎI: {question}\n\n"
        "Ngữ cảnh trên đã ĐỦ và LIÊN QUAN để trả lời đầy đủ câu hỏi chưa?\n"
        "- Nếu ĐỦ: chỉ in đúng một từ: ENOUGH\n"
        "- Nếu CHƯA đủ: in: NEED: <một truy vấn tìm kiếm bổ sung để lấp phần còn thiếu>"
    )
    text = rag._generate(
        "Bạn đánh giá ngữ cảnh có đủ để trả lời câu hỏi hay không.", prompt
    ).strip()

    upper = text.upper()
    if "NEED:" in upper:
        idx = upper.index("NEED:")
        follow = text[idx + len("NEED:"):].strip().splitlines()[0].strip()
        if follow:
            return False, follow
    # Mặc định coi như ĐỦ -> tránh lặp vô hạn khi model trả lời mơ hồ.
    return True, ""


def _top_hits(gathered: dict, query: str) -> list[dict]:
    """Chọn các đoạn liên quan nhất: ưu tiên re-ranking theo `query`,
    nếu tắt rerank thì sắp theo khoảng cách embedding."""
    hits = list(gathered.values())
    if config.RERANK_ENABLED:
        return rerank.rerank(query, hits, config.AGENT_CONTEXT_LIMIT)
    hits.sort(key=lambda h: h["distance"])
    return hits[: config.AGENT_CONTEXT_LIMIT]


def answer_question(question: str, history: list[dict] | None = None) -> dict:
    """Chế độ agentic. Trả về {answer, sources, steps}.

    `steps` là nhật ký từng bước để hiển thị 'quá trình suy nghĩ' trên giao diện."""
    history = history or []
    steps: list[str] = []

    # Nhớ hội thoại: viết lại câu hỏi cho độc lập trước khi tìm.
    standalone = rag.contextualize_question(history, question)
    if standalone != question:
        steps.append(f'Hiểu câu hỏi (theo ngữ cảnh): "{standalone}"')

    # 1) Viết lại / tách câu hỏi
    queries = _rewrite_queries(standalone)
    steps.append("Tách thành truy vấn: " + " | ".join(queries))

    # Bộ nhớ các đoạn đã thu thập, khử trùng theo (nguồn, chỉ số chunk).
    gathered: dict[tuple, dict] = {}
    per_query_k = config.RERANK_CANDIDATES if config.RERANK_ENABLED else config.TOP_K

    def _retrieve(qs: list[str]) -> int:
        added = 0
        for q in qs:
            vec = embeddings.embed_query(q)
            for hit in vector_store.query(vec, per_query_k):
                key = (hit["source"], hit["chunk_index"])
                if key not in gathered:
                    gathered[key] = hit
                    added += 1
                elif hit["distance"] < gathered[key]["distance"]:
                    gathered[key] = hit  # giữ bản liên quan hơn
        return added

    # 2) Vòng lặp: truy hồi -> tự đánh giá -> (có thể) tìm thêm
    for it in range(config.AGENT_MAX_ITERS):
        added = _retrieve(queries)
        steps.append(f"Vòng {it + 1}: thu thập thêm {added} đoạn (tổng {len(gathered)}).")

        if not gathered:
            return {
                "answer": "Chưa có tài liệu nào được nạp. Hãy tải tài liệu lên trước nhé.",
                "sources": [],
                "steps": steps,
            }

        # Vòng cuối thì không đánh giá nữa, đi thẳng tới bước tổng hợp.
        if it == config.AGENT_MAX_ITERS - 1:
            break

        enough, follow = _decide(standalone, rag._build_context(_top_hits(gathered, standalone)))
        if enough:
            steps.append("Tự đánh giá: ĐỦ thông tin -> tổng hợp câu trả lời.")
            break
        steps.append(f'Tự đánh giá: CHƯA đủ -> tìm thêm: "{follow}"')
        queries = [follow]

    # 3) Tổng hợp câu trả lời cuối từ các đoạn liên quan nhất (đã xếp hạng lại).
    hits = _top_hits(gathered, standalone)
    context = rag._build_context(hits)
    hist = rag.history_block(history)
    user_prompt = (
        (hist + "\n\n" if hist else "")
        + f"NGỮ CẢNH (các đoạn trích từ tài liệu):\n\n{context}\n\n"
        + f"---\n\nCÂU HỎI CỦA HỌC SINH: {question}\n\n"
        + "Hãy trả lời dựa trên ngữ cảnh ở trên."
    )
    answer = rag._generate(rag.SYSTEM_PROMPT, user_prompt)

    return {"answer": answer, "sources": rag._make_sources(hits), "steps": steps}
