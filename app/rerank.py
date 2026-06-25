"""Re-ranking: xếp hạng lại các đoạn truy hồi để giữ phần liên quan nhất.

VÌ SAO cần bước này?
- Tìm bằng embedding rất NHANH nhưng đôi khi "xêm xêm" về nghĩa, không thật sự
  trúng. Ta lấy NHIỀU ứng viên (vd 20 đoạn) bằng embedding, rồi dùng một mô hình
  "cross-encoder" đọc TRỰC TIẾP cặp (câu hỏi, đoạn) để chấm điểm liên quan kỹ hơn,
  cuối cùng chỉ giữ lại vài đoạn tốt nhất.
- Cross-encoder chính xác hơn embedding nhưng CHẬM hơn, nên chỉ dùng để xếp hạng
  lại số ít ứng viên (không dùng để quét cả kho).
"""
from __future__ import annotations

from functools import lru_cache

from app import config


@lru_cache(maxsize=1)
def _get_model():
    # Tải mô hình cross-encoder 1 lần rồi giữ trong bộ nhớ. Lần đầu sẽ tự tải về.
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANK_MODEL)


def rerank(query: str, hits: list[dict], top_n: int) -> list[dict]:
    """Trả về `top_n` đoạn liên quan nhất với `query`.

    Nếu tắt rerank hoặc gặp lỗi (vd chưa tải được model), rơi về cách an toàn:
    giữ nguyên thứ tự embedding và cắt lấy top_n."""
    if not hits:
        return []
    if not config.RERANK_ENABLED:
        return hits[:top_n]
    try:
        model = _get_model()
        pairs = [(query, h["text"]) for h in hits]
        scores = model.predict(pairs)  # điểm càng cao = càng liên quan
        ranked = sorted(zip(hits, scores), key=lambda x: float(x[1]), reverse=True)
        return [hit for hit, _ in ranked[:top_n]]
    except Exception:
        # Không để rerank làm sập cả hệ thống — luôn có đường lui.
        return hits[:top_n]
