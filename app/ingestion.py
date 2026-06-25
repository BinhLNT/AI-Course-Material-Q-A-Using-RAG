"""Ghép các bước "nạp tài liệu" lại thành một quy trình duy nhất:

    đọc file  ->  cắt chunk  ->  tạo embedding  ->  lưu vào vector DB

VÌ SAO tách riêng file này?
- Mỗi module trên (loaders/chunking/embeddings/vector_store) chỉ lo MỘT việc.
  File này đóng vai "nhạc trưởng" ghép chúng lại -> code dễ đọc, dễ test.
"""
from __future__ import annotations

from pathlib import Path

from app import chunking, config, embeddings, loaders, vector_store


def ingest_file(path: Path) -> int:
    """Nạp 1 file vào hệ thống. Trả về số chunk đã tạo (0 nếu file rỗng)."""
    text = loaders.load_text(path)                                   # 1) đọc chữ
    chunks = chunking.split_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)  # 2) cắt
    if not chunks:
        return 0
    vectors = embeddings.embed_documents(chunks)                     # 3) embedding
    vector_store.add_chunks(path.name, chunks, vectors)              # 4) lưu kho
    return len(chunks)
