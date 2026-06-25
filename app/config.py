"""Cấu hình tập trung cho toàn bộ ứng dụng.

VÌ SAO tách riêng một file config?
- Mọi "con số ma thuật" (tên model, kích thước chunk, top_k...) nằm gọn 1 chỗ,
  dễ tìm và dễ chỉnh mà không phải lục khắp code.
- KHÔNG hard-code API key trong code. Ta đọc từ biến môi trường (.env) để tránh
  vô tình commit bí mật lên Git.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Đọc file .env (nếu có) và nạp vào biến môi trường của tiến trình.
load_dotenv()

# Thư mục gốc của project (…/AI-Course-Material-Q-A-Using-RAG)
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Đường dẫn lưu trữ ---
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"   # nơi lưu file học sinh tải lên
CHROMA_DIR = DATA_DIR / "chroma"    # nơi ChromaDB lưu vector (bền vững trên đĩa)
WEB_DIR = BASE_DIR / "web"          # chứa giao diện web (index.html)

# Tự tạo thư mục nếu chưa tồn tại (an toàn khi chạy lần đầu).
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# --- Nhà cung cấp LLM (phần SINH câu trả lời) ---
# "openrouter" -> dùng model MIỄN PHÍ qua OpenRouter (mặc định, hợp để test)
# "anthropic"  -> dùng Claude (cần ANTHROPIC_API_KEY, trả phí)
# Lưu ý: phần EMBEDDING luôn chạy local & miễn phí, KHÔNG phụ thuộc lựa chọn này.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower()
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))

# OpenRouter — API tương thích OpenAI, có nhiều model miễn phí (slug có đuôi ":free").
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")

# Claude (Anthropic) — chỉ dùng khi LLM_PROVIDER="anthropic".
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-8")

# --- Model embedding (biến văn bản -> vector số, chạy ngay trên máy) ---
# Mặc định dùng e5 đa ngôn ngữ: nhỏ, nhanh, hỗ trợ tiếng Việt tốt.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

# --- Tham số RAG ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))       # số ký tự mỗi đoạn
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))  # số ký tự gối nhau giữa 2 đoạn
TOP_K = int(os.getenv("TOP_K", "5"))                    # số đoạn lấy ra khi truy hồi

# Tên "bộ sưu tập" vector trong ChromaDB.
COLLECTION_NAME = "course_materials"

# --- Chế độ RAG ---
# "agentic" -> vòng lặp tự điều hướng (mặc định)
# "basic"   -> tìm 1 lần (đơn giản nhất)
# "tools"   -> tool-calling agent: LLM tự quyết định gọi công cụ tìm kiếm
RAG_MODE = os.getenv("RAG_MODE", "agentic").lower()
AGENT_MAX_ITERS = int(os.getenv("AGENT_MAX_ITERS", "2"))        # số vòng truy hồi tối đa
AGENT_MAX_QUERIES = int(os.getenv("AGENT_MAX_QUERIES", "3"))    # số truy vấn con tối đa
AGENT_CONTEXT_LIMIT = int(os.getenv("AGENT_CONTEXT_LIMIT", "8"))  # số đoạn tối đa khi tổng hợp
TOOLS_MAX_CALLS = int(os.getenv("TOOLS_MAX_CALLS", "5"))        # số lượt gọi công cụ tối đa (chế độ tools)

# --- Re-ranking (xếp hạng lại kết quả truy hồi cho chính xác hơn) ---
# Lấy nhiều ứng viên bằng embedding (nhanh) rồi dùng mô hình cross-encoder chấm
# điểm liên quan kỹ hơn để giữ lại các đoạn tốt nhất.
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
# Mô hình cross-encoder đa ngôn ngữ, nhỏ gọn (hỗ trợ tiếng Việt).
# Chất lượng cao hơn (nặng hơn): BAAI/bge-reranker-v2-m3
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))  # số ứng viên lấy ra trước khi xếp hạng lại

# Số lượt hội thoại gần nhất được nhớ và đưa vào ngữ cảnh.
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "4"))
