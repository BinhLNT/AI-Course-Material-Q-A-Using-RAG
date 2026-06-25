"""Lưu và truy hồi vector bằng ChromaDB (cơ sở dữ liệu vector chạy local).

VÌ SAO cần "vector database"?
- Khi có hàng nghìn chunk, ta cần TÌM NHANH vài chunk gần nghĩa nhất với câu hỏi.
  Chroma làm đúng việc đó (tìm "hàng xóm gần nhất" trong không gian vector) và
  LƯU BỀN VỮNG trên đĩa, nên tắt máy mở lại không phải nạp tài liệu từ đầu.
"""
from __future__ import annotations

import chromadb

from app import config

# PersistentClient -> dữ liệu được ghi xuống đĩa tại CHROMA_DIR.
_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def _make_collection():
    # "hnsw:space": "cosine" -> đo độ tương đồng bằng cosine, hợp với vector
    # đã được chuẩn hoá ở bước embedding.
    return _client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


_collection = _make_collection()


def add_chunks(source: str, chunks: list[str], embeddings: list[list[float]]) -> None:
    """Thêm các chunk của một file vào kho vector.

    Mỗi chunk lưu kèm metadata (tên file nguồn + thứ tự chunk) để sau này
    truy ngược ra "câu trả lời lấy từ đâu"."""
    # Nếu file này đã từng được nạp, xoá HẾT chunk cũ của nó trước khi nạp lại.
    # Vì sao? Nếu file mới ngắn hơn (ít chunk hơn) mà chỉ ghi đè theo id, các
    # chunk cũ thừa sẽ "mồ côi" và vẫn bị tìm thấy -> trả lời theo nội dung đã xoá.
    _collection.delete(where={"source": source})

    ids = [f"{source}::{i}" for i in range(len(chunks))]
    metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
    # upsert: an toàn kể cả khi nạp lại cùng file (không lỗi trùng id).
    _collection.upsert(
        ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas
    )


def query(query_embedding: list[float], top_k: int) -> list[dict]:
    """Trả về danh sách chunk gần nghĩa nhất với câu hỏi."""
    result = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    # Chroma trả về list lồng (một phần tử cho mỗi câu hỏi). Ta chỉ hỏi 1 câu
    # nên lấy phần tử [0]. Nếu kho rỗng, các list này sẽ rỗng.
    docs = result["documents"][0] if result["documents"] else []
    metas = result["metadatas"][0] if result["metadatas"] else []
    dists = result["distances"][0] if result["distances"] else []

    hits: list[dict] = []
    for doc, meta, dist in zip(docs, metas, dists):
        hits.append(
            {
                "text": doc,
                "source": meta.get("source"),
                "chunk_index": meta.get("chunk_index"),
                "distance": dist,
            }
        )
    return hits


def count() -> int:
    """Tổng số chunk hiện có trong kho."""
    return _collection.count()


def reset() -> None:
    """Xoá toàn bộ dữ liệu đã nạp để làm lại từ đầu."""
    global _collection
    _client.delete_collection(config.COLLECTION_NAME)
    _collection = _make_collection()
