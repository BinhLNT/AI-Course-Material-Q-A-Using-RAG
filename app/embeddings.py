"""Biến văn bản thành vector số (embedding) bằng model chạy ngay trên máy.

VÌ SAO dùng embedding?
- Máy không "hiểu" chữ. Embedding biến mỗi đoạn văn thành một dãy số sao cho
  các đoạn có nghĩa gần nhau thì vector cũng gần nhau. Nhờ đó ta tìm kiếm theo
  *ngữ nghĩa* (ý nghĩa) chứ không chỉ khớp từ khoá.

VÌ SAO có tiền tố "query:" và "passage:"?
- Họ model e5 được huấn luyện theo quy ước: câu hỏi thêm "query: ", còn đoạn
  tài liệu thêm "passage: ". Thêm đúng tiền tố giúp truy hồi chính xác hơn rõ rệt.
  (Nếu bạn đổi sang model embedding KHÁC họ e5, hãy bỏ 2 tiền tố này đi.)
"""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app import config


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Tải model 1 lần rồi giữ trong bộ nhớ. Lần đầu sẽ tự tải model về máy
    (cần mạng), các lần sau dùng lại bản đã tải."""
    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embedding cho các đoạn TÀI LIỆU (dùng tiền tố 'passage:')."""
    model = _get_model()
    inputs = [f"passage: {t}" for t in texts]
    # normalize_embeddings=True -> chuẩn hoá độ dài vector, hợp với đo cosine.
    vectors = model.encode(inputs, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embedding cho CÂU HỎI (dùng tiền tố 'query:')."""
    model = _get_model()
    vector = model.encode(
        f"query: {text}", normalize_embeddings=True, show_progress_bar=False
    )
    return vector.tolist()
