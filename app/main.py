"""Ứng dụng web FastAPI.

Cung cấp:
- Giao diện web (GET /)            -> trang index.html cho học sinh dùng.
- API tải tài liệu (POST /api/upload)
- API hỏi đáp (POST /api/chat)
- API xem trạng thái (GET /api/status) và xoá dữ liệu (POST /api/reset)

VÌ SAO phục vụ luôn web từ FastAPI (cùng một cổng)?
- Frontend và backend cùng "origin" nên KHÔNG dính lỗi CORS, đỡ phức tạp cho
  người mới. Chỉ cần chạy 1 server là xong.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import agent, config, ingestion, loaders, rag, tools, vector_store

app = FastAPI(title="Trợ giảng AI - Hỏi đáp tài liệu (RAG)")


class ChatRequest(BaseModel):
    """Khuôn dữ liệu cho body của /api/chat. Pydantic tự kiểm tra kiểu giúp ta."""

    question: str
    # Lịch sử hội thoại: danh sách {"role": "user"|"assistant", "content": "..."}
    history: list[dict] = []


@app.get("/")
def index() -> FileResponse:
    """Trả về trang web chính."""
    return FileResponse(config.WEB_DIR / "index.html")


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)) -> dict:
    """Nhận một hoặc nhiều file, lưu lại rồi nạp vào hệ thống RAG."""
    results = []
    for f in files:
        # An toàn: chỉ giữ TÊN file (bỏ mọi đường dẫn và '..') để chống lỗ hổng
        # "path traversal" — kẻ xấu gửi filename kiểu "../../README.md" có thể
        # ghi đè file ngoài thư mục uploads. Path(...).name loại bỏ điều đó.
        filename = Path(f.filename or "khong_ten").name
        suffix = Path(filename).suffix.lower()
        if suffix not in loaders.SUPPORTED_EXTENSIONS:
            results.append(
                {"filename": filename, "status": "bỏ qua (định dạng không hỗ trợ)", "chunks": 0}
            )
            continue

        dest = config.UPLOAD_DIR / filename
        dest.write_bytes(await f.read())
        try:
            n = ingestion.ingest_file(dest)
            results.append({"filename": filename, "status": "đã nạp", "chunks": n})
        except Exception as exc:  # noqa: BLE001 - báo lỗi thân thiện thay vì sập
            results.append({"filename": filename, "status": f"lỗi: {exc}", "chunks": 0})

    return {"results": results, "total_chunks": vector_store.count()}


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    """Trả lời câu hỏi của học sinh dựa trên tài liệu đã nạp.

    Chọn chế độ theo config.RAG_MODE: "agentic" (vòng lặp) hay "basic" (tìm 1 lần)."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi đang để trống.")
    if config.RAG_MODE == "tools":
        handler = tools
    elif config.RAG_MODE == "agentic":
        handler = agent
    else:
        handler = rag
    return handler.answer_question(req.question, req.history)


@app.get("/api/status")
def status() -> dict:
    """Cho biết số chunk trong kho và chế độ RAG đang dùng."""
    return {"chunks": vector_store.count(), "mode": config.RAG_MODE}


@app.post("/api/reset")
def reset() -> dict:
    """Xoá sạch dữ liệu đã nạp."""
    vector_store.reset()
    return {"status": "đã xoá toàn bộ dữ liệu"}
