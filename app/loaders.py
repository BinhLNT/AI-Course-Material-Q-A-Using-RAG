"""Đọc nội dung văn bản từ nhiều định dạng file: PDF, DOCX, TXT, MD.

VÌ SAO cần bước này?
- Model AI chỉ làm việc với *văn bản thuần*. PDF/DOCX là định dạng phức tạp,
  phải "bóc" chữ ra trước khi chunk + embedding.
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from pypdf import PdfReader

# Các định dạng ta hỗ trợ đọc.
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def load_text(path: Path) -> str:
    """Trả về toàn bộ văn bản trong file. Ném lỗi nếu định dạng không hỗ trợ."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix == ".docx":
        return _load_docx(path)
    if suffix in {".txt", ".md"}:
        # errors="ignore" để không vỡ nếu file có vài ký tự lạ.
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Định dạng không hỗ trợ: {suffix}")


def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    # Mỗi trang trích ra văn bản; nối lại bằng dòng trống để giữ ranh giới trang.
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _load_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)
