"""Cắt văn bản dài thành các "chunk" (đoạn nhỏ) có gối nhau.

VÌ SAO phải chunk thay vì nhồi cả tài liệu?
1. Truy hồi chính xác hơn: mỗi vector đại diện cho 1 đoạn ngắn tập trung 1 ý,
   nên khi tìm kiếm theo ngữ nghĩa sẽ "trúng" đúng phần liên quan.
2. Tiết kiệm token & tiền: chỉ đưa vài đoạn liên quan vào prompt cho Claude,
   không đưa cả cuốn sách (vừa tốn tiền, vừa loãng, vừa dễ vượt giới hạn).

VÌ SAO cần "overlap" (gối nhau)?
- Để một câu/ý nằm vắt ngang ranh giới 2 chunk không bị cắt mất ngữ cảnh.
"""
from __future__ import annotations


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Cắt `text` thành danh sách đoạn dài tối đa `chunk_size` ký tự,
    mỗi đoạn gối lên đoạn trước `overlap` ký tự."""
    text = text.strip()
    if not text:
        return []

    # Bước nhảy giữa các điểm bắt đầu. max(1, ...) để không bao giờ lặp vô hạn
    # (kể cả khi lỡ đặt overlap >= chunk_size).
    step = max(1, chunk_size - overlap)

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:  # đã tới cuối văn bản -> dừng
            break
        start += step
    return chunks
