# 📚 Trợ giảng AI — Hỏi đáp tài liệu học tập (RAG)

Một ứng dụng web nhỏ: học sinh **tải tài liệu lên** (PDF/DOCX/TXT/MD), rồi **đặt câu hỏi**.
AI trả lời **chỉ dựa trên tài liệu** đó và **dẫn nguồn** để học sinh kiểm chứng.

Stack: Python · tự code từ đầu · embedding **local miễn phí** (e5 đa ngôn ngữ) ·
**ChromaDB** (vector DB) · **FastAPI** + web tĩnh · sinh câu trả lời bằng **OpenRouter (model miễn phí)** hoặc **Claude**.

Tính năng: **3 chế độ truy hồi** (basic · agentic · tools) · **nhớ hội thoại** · **re-ranking** · trích nguồn.

---

## 1. RAG là gì và vì sao dùng nó?

**RAG = Retrieval-Augmented Generation** = "sinh câu trả lời có bổ trợ bằng truy hồi".

Thay vì hỏi thẳng AI (dễ bị **bịa** — *hallucination* — vì model không biết tài liệu
riêng của bạn), RAG làm 2 bước:

1. **Truy hồi (Retrieval):** tìm trong tài liệu của bạn vài đoạn *liên quan nhất* tới câu hỏi.
2. **Sinh (Generation):** đưa các đoạn đó cho LLM (Claude hoặc model OpenRouter) và yêu cầu trả lời *dựa trên chúng*.

**Vì sao không chọn cách khác?**

| Cách làm | Vấn đề |
|---|---|
| Hỏi thẳng AI, không có tài liệu | AI không biết nội dung khoá học → bịa hoặc trả lời chung chung |
| Nhồi **cả** tài liệu vào prompt mỗi lần hỏi | Tốn token/tiền, chậm, và dễ vượt giới hạn khi tài liệu lớn |
| **Fine-tune** (huấn luyện lại model) | Tốn kém, phức tạp, cập nhật tài liệu mới là phải train lại |
| **RAG** ✅ | Rẻ, nhanh, cập nhật tài liệu = nạp thêm file; trả lời bám sát tài liệu |

---

## 2. Kiến trúc & luồng hoạt động

Có **2 luồng** chính:

```
LÚC NẠP TÀI LIỆU (chạy 1 lần cho mỗi file):
  File (PDF/DOCX/TXT)
      │  loaders.py     → bóc ra văn bản thuần
      ▼
  Văn bản dài
      │  chunking.py    → cắt thành đoạn nhỏ có gối nhau
      ▼
  Nhiều chunk
      │  embeddings.py  → biến mỗi chunk thành vector số
      ▼
  Vector + nội dung + nguồn
      │  vector_store.py→ lưu vào ChromaDB (trên đĩa)
      ▼
  📦 Kho vector

LÚC HỎI (chạy mỗi câu hỏi):
  Câu hỏi (+ lịch sử hội thoại)
      │  rag.py         → viết lại thành câu hỏi độc lập (nhớ hội thoại)
      ▼
  Câu hỏi độc lập
      │  embeddings.py + vector_store.py → lấy nhiều đoạn ứng viên (📦)
      ▼
  Ứng viên
      │  rerank.py      → xếp hạng lại, giữ đoạn tốt nhất
      ▼
  Đoạn liên quan
      │  rag.py / agent.py / tools.py  (tuỳ RAG_MODE) → gọi LLM
      ▼
  ✅ Câu trả lời + nguồn (+ "🧠 các bước" ở chế độ agentic/tools)
```

---

## 3. Cấu trúc thư mục (và vì sao tách như vậy)

```
AI-Course-Material-Q-A-Using-RAG/
├─ app/                      # toàn bộ code backend
│  ├─ config.py              # CẤU HÌNH tập trung (model, đường dẫn, tham số RAG)
│  ├─ loaders.py             # ĐỌC file → văn bản (PDF/DOCX/TXT/MD)
│  ├─ chunking.py            # CẮT văn bản thành chunk
│  ├─ embeddings.py          # văn bản → VECTOR (model local e5)
│  ├─ vector_store.py        # LƯU & TÌM vector (ChromaDB)
│  ├─ rerank.py              # XẾP HẠNG LẠI ứng viên (cross-encoder)
│  ├─ ingestion.py           # "nhạc trưởng": ghép các bước nạp tài liệu
│  ├─ rag.py                 # chế độ BASIC + hàm dùng chung (gọi LLM, nhớ hội thoại, truy hồi)
│  ├─ agent.py               # chế độ AGENTIC (vòng lặp tự điều hướng)
│  ├─ tools.py               # chế độ TOOLS (tool-calling agent)
│  └─ main.py                # FastAPI: các API + phục vụ web + định tuyến 3 chế độ
├─ web/
│  └─ index.html             # giao diện (HTML + JS thuần, không cần build)
├─ data/                     # tự sinh khi chạy (đã .gitignore)
│  ├─ uploads/               # file học sinh tải lên
│  └─ chroma/                # database vector
├─ requirements.txt          # danh sách thư viện cần cài
├─ .env.example              # mẫu cấu hình — copy thành .env
└─ README.md                 # file bạn đang đọc
```

> **Vì sao mỗi việc một file (separation of concerns)?**
> Mỗi module chỉ lo **một** nhiệm vụ. Khi muốn đổi vector DB (vd. sang FAISS) bạn
> chỉ sửa `vector_store.py`; đổi model embedding chỉ sửa `embeddings.py`. Code dễ
> đọc, dễ sửa, dễ test — đây là cách nghĩ quan trọng nhất khi học lập trình.

---

## 4. Cài đặt & chạy (Windows / PowerShell)

> Yêu cầu: **Python 3.10+** và **một API key cho phần sinh câu trả lời**.
> Mặc định dùng **OpenRouter** với model **miễn phí** — tạo key free tại
> <https://openrouter.ai/keys>. (Phần embedding chạy local, **không cần key**.)
> Muốn dùng Claude: đặt `LLM_PROVIDER=anthropic` và key của Anthropic.

```powershell
# 1) Vào thư mục project
cd f:\Documents\GitHub\AI-Course-Material-Q-A-Using-RAG

# 2) Tạo & kích hoạt môi trường ảo (bạn đã có sẵn .venv thì bỏ qua bước tạo)
python -m venv .venv
.venv\Scripts\Activate.ps1
#   Nếu PowerShell báo chặn script, chạy 1 lần:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3) Cài thư viện (lần đầu hơi lâu vì có PyTorch)
pip install -r requirements.txt

# 4) Tạo file .env từ mẫu rồi điền key
Copy-Item .env.example .env
notepad .env        # mặc định LLM_PROVIDER=openrouter -> điền OPENROUTER_API_KEY

# 5) Chạy server
uvicorn app.main:app --reload
```

Mở trình duyệt tại **<http://127.0.0.1:8000>** → tải tài liệu lên → đặt câu hỏi.

> ⏳ **Lần chạy đầu** sẽ tự **tải model embedding** về máy (vài trăm MB, cần mạng).
> Các lần sau dùng lại bản đã tải, không cần mạng cho phần embedding.

---

### Đổi nhà cung cấp LLM (miễn phí ↔ Claude ↔ offline)

Mọi thứ điều khiển trong `.env`, **không phải sửa code** (nhờ lớp sinh câu trả lời đã tách riêng):

- **Miễn phí (mặc định):** `LLM_PROVIDER=openrouter` + `OPENROUTER_API_KEY` + `OPENROUTER_MODEL`
  (slug có đuôi `:free`, chọn ở <https://openrouter.ai/models>).
- **Claude:** `LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY` + `CLAUDE_MODEL`.
- **Hoàn toàn offline (không internet, không key):** cài [Ollama](https://ollama.com), rồi đặt
  `OPENROUTER_BASE_URL=http://localhost:11434/v1`, `OPENROUTER_API_KEY=ollama`,
  `OPENROUTER_MODEL=llama3.1`. Ollama cũng theo chuẩn OpenAI nên **dùng chung code**.

> Model miễn phí có **giới hạn lượt/ngày** và chất lượng thường thấp hơn Claude — rất hợp để học & test.

## 5. Giải thích từng quyết định kỹ thuật ("vì sao làm vậy mới tốt")

### a) Chunking — kích thước & overlap
- `CHUNK_SIZE=1000` ký tự, `CHUNK_OVERLAP=150`.
- **Chunk quá lớn** → 1 vector gói nhiều ý → tìm kiếm kém chính xác, tốn token.
- **Chunk quá nhỏ** → mất ngữ cảnh, câu trả lời rời rạc.
- **Overlap** giúp câu nằm vắt ngang ranh giới 2 chunk không bị cắt mất nghĩa.
- 👉 Chỉnh trong `.env`. Tài liệu kỹ thuật nhiều thuật ngữ có thể giảm CHUNK_SIZE.

### b) Embedding local, đa ngôn ngữ — `intfloat/multilingual-e5-small`
- Chạy **miễn phí ngay trên máy**, **không cần API key thứ 2**, hỗ trợ **tiếng Việt**.
- e5 yêu cầu thêm tiền tố: tài liệu → `passage:`, câu hỏi → `query:`
  (xem `embeddings.py`). Thêm đúng tiền tố giúp truy hồi **chính xác hơn rõ rệt**.
- Vector được **chuẩn hoá** (`normalize_embeddings=True`) để đo độ tương đồng bằng **cosine**.
- Muốn chất lượng cao hơn: đổi `EMBEDDING_MODEL=intfloat/multilingual-e5-base` (chậm hơn).

### c) ChromaDB — đo bằng cosine, dùng `upsert`
- `metadata={"hnsw:space":"cosine"}`: chọn thước đo phù hợp với vector đã chuẩn hoá.
- `upsert` thay vì `add`: tải lại **cùng một file** sẽ **ghi đè**, không bị trùng dữ liệu.
- **Lưu bền vững trên đĩa** (`PersistentClient`): tắt máy mở lại không phải nạp lại.

### d) Prompt "grounding" — chống bịa
Trong `rag.py`, system prompt buộc model (Claude hoặc model OpenRouter):
- **Chỉ** dùng phần NGỮ CẢNH; nếu không có thì nói *"không tìm thấy trong tài liệu"*.
- Trả lời **đúng ngôn ngữ** câu hỏi.
- **Dẫn nguồn** `[Nguồn N]`.
- 👉 Đây là điểm cốt lõi làm cho một trợ giảng AI **đáng tin** với học sinh.

### e) Trả về danh sách nguồn
Mỗi câu trả lời kèm các đoạn đã dùng (tên file + trích đoạn) → học sinh **tự kiểm chứng**.

---

## 5b. Ba chế độ RAG: basic · agentic · tools

Cả 3 chế độ **dùng chung** pipeline (embedding e5 → ChromaDB → re-ranking → nhớ hội thoại → trả lời có trích nguồn). Khác nhau ở **ai điều khiển việc tìm kiếm** và **tìm bao nhiêu lần**. Đổi bằng `RAG_MODE` trong `.env`:

| | **basic** | **agentic** (mặc định) | **tools** |
|---|---|---|---|
| Ai điều khiển tìm kiếm | Cố định (code) | Code điều phối từng bước | **LLM tự quyết định** |
| Số lần tìm | 1 lần | Nhiều vòng (tách + tự đánh giá) | LLM tự gọi tuỳ ý |
| Số lần gọi LLM / câu hỏi | ~1–2 | ~3–4+ | nhiều nhất (tới `TOOLS_MAX_CALLS`+1) |
| Tốc độ / chi phí | Nhanh, rẻ nhất | Trung bình | Chậm, tốn nhất |
| Cần model hỗ trợ tool-calling? | Không | Không | **Có** |
| Hợp khi | Tài liệu đơn giản, cần nhanh/rẻ | Câu hỏi phức tạp, model free | Có model mạnh, muốn linh hoạt nhất |

**Ưu / nhược nhanh:**
- **basic** — ✅ nhanh, rẻ, ổn định trên mọi model · ❌ câu hỏi phức tạp dễ bỏ sót, không tự sửa.
- **agentic** — ✅ tách câu hỏi + tìm nhiều vòng + tự đánh giá → ít bỏ sót; chạy ổn trên model free · ❌ chậm/tốn hơn, luồng cố định.
- **tools** — ✅ linh hoạt & "agentic" thật sự, mạnh nhất với model giỏi · ❌ cần model hỗ trợ tool-calling, khó đoán/debug, tốn nhất.

> 💡 Thử cùng một câu hỏi phức tạp ở cả 3 chế độ rồi mở **"🧠 Xem quá trình suy nghĩ"** để thấy rõ khác biệt.

### Chi tiết chế độ agentic (mặc định)

Thay vì tìm 1 lần, hệ thống **tự điều hướng việc tìm kiếm**:

1. **Viết lại & tách** câu hỏi thành 1–3 truy vấn tìm kiếm.
2. **Truy hồi nhiều vòng**: tìm → gộp → LLM **tự đánh giá** đã đủ thông tin chưa.
3. Nếu **chưa đủ** → LLM sinh **truy vấn bổ sung** và tìm tiếp (tối đa `AGENT_MAX_ITERS` vòng).
4. **Tổng hợp** câu trả lời cuối từ các đoạn liên quan nhất + trích nguồn.

Code điều phối ở [app/agent.py](app/agent.py); từng bước hiện trong mục **"🧠 Xem quá trình suy nghĩ"** trên giao diện.

**Vì sao tốt hơn?** Câu hỏi phức tạp/nhiều ý được tách nhỏ và tìm nhiều vòng → ít bỏ sót, ít bịa. Vì **code điều khiển từng bước** (không dựa vào tool-calling) nên **chạy ổn cả trên model miễn phí**.

Tinh chỉnh trong `.env`: `RAG_MODE` (đổi `basic` để so sánh), `AGENT_MAX_ITERS`, `AGENT_MAX_QUERIES`, `AGENT_CONTEXT_LIMIT`.

> ⚠️ Chi phí/giới hạn: mỗi câu hỏi agentic gọi LLM **nhiều lần hơn** (viết lại + đánh giá + tổng hợp). Với model free có giới hạn lượt/ngày, nếu bị chặn hãy đặt `AGENT_MAX_ITERS=1` hoặc `RAG_MODE=basic`.

---

## 5c. Nhớ hội thoại · Re-ranking · Tool-calling agent

**a) Nhớ hội thoại (hỏi nối tiếp).** Giao diện gửi kèm lịch sử chat. Trước khi tìm,
hệ thống **viết lại** câu hỏi nối tiếp thành câu hỏi độc lập (vd: *"vậy còn cái thứ hai?"*
→ *"Đặc điểm của phương pháp thứ hai là gì?"*) — xem `rag.contextualize_question()`.
**Vì sao?** Embedding của "cái đó" gần như vô nghĩa; phải thay bằng danh từ cụ thể mới tìm đúng.

**b) Re-ranking (xếp hạng lại).** Tìm bằng embedding rất nhanh nhưng đôi khi không thật
trúng. Ta lấy nhiều ứng viên (`RERANK_CANDIDATES`=20) rồi dùng mô hình **cross-encoder**
đọc trực tiếp cặp *(câu hỏi, đoạn)* để chấm điểm kỹ hơn, giữ lại top tốt nhất — xem
[app/rerank.py](app/rerank.py). Cross-encoder chính xác hơn nhưng chậm hơn, nên chỉ dùng
để xếp hạng lại số ít ứng viên. Tắt bằng `RERANK_ENABLED=false`.

**c) Tool-calling agent (`RAG_MODE=tools`).** Thay vì code điều phối, **LLM tự quyết định**
khi nào & tìm gì qua công cụ `search_course_materials`, gọi nhiều lần rồi tự dừng để trả lời
— xem [app/tools.py](app/tools.py).

> 💡 **Tool-calling MIỄN PHÍ được không?** Được. Tool-calling là tính năng của *model*, không
> phải của nhà cung cấp. Nhiều model **free trên OpenRouter** hỗ trợ (lọc cột *Tools* ở
> <https://openrouter.ai/models>): vd `meta-llama/llama-3.3-70b-instruct:free`,
> `qwen/qwen-2.5-72b-instruct:free`, `deepseek/deepseek-chat-v3-0324:free`. **Ollama** local
> cũng hỗ trợ. Nếu model không hỗ trợ, hệ thống báo lỗi gợi ý đổi model hoặc dùng `agentic`.
> Với nhà cung cấp Claude, chế độ tools tự chuyển sang `agentic` (vì dùng giao thức tool của OpenAI).

---

## 6. Mẹo & hướng nâng cấp

- **Tiết kiệm chi phí:** đổi `CLAUDE_MODEL` trong `.env`:
  `claude-sonnet-4-6` (cân bằng) hoặc `claude-haiku-4-5` (rẻ/nhanh nhất).
- **Trả lời theo từng chữ (streaming):** dùng `client.messages.stream(...)` để chữ
  hiện dần như ChatGPT (cần thêm endpoint dạng SSE).
- **Re-ranking** đã bật sẵn (`RERANK_ENABLED`). Muốn chính xác hơn nữa: đổi
  `RERANK_MODEL=BAAI/bge-reranker-v2-m3` (nặng hơn) hoặc thử *hybrid search* (kết hợp tìm từ khoá + ngữ nghĩa).
- **Đánh giá chất lượng:** tạo bộ câu hỏi mẫu + đáp án đúng, đo xem RAG trả lời trúng không.
- **Tách theo từng học sinh/lớp:** dùng nhiều collection trong Chroma, hoặc thêm
  `metadata` lớp học và lọc khi truy vấn.

---

## 7. Xử lý sự cố thường gặp

| Triệu chứng | Cách xử lý |
|---|---|
| Lỗi 401 / `api_key` | Chưa điền đúng key trong `.env` (OpenRouter: `OPENROUTER_API_KEY`; Claude: `ANTHROPIC_API_KEY`) |
| OpenRouter báo `model not found` / 404 | Slug sai hoặc model không còn miễn phí. Mở <https://openrouter.ai/models>, copy slug `:free` còn hiệu lực vào `OPENROUTER_MODEL` |
| OpenRouter báo 402 / hết hạn mức | Model miễn phí có giới hạn lượt/ngày. Đợi reset, đổi model `:free` khác, hoặc nạp ít credit |
| Chế độ `tools` báo lỗi / không gọi công cụ | Model đang dùng không hỗ trợ tool-calling. Chọn model có cột *Tools* trên OpenRouter, hoặc đặt `RAG_MODE=agentic` |
| Lần đầu chậm hơn trước | Đang tải thêm **model re-ranking**. Tải 1 lần rồi cache; muốn bỏ thì đặt `RERANK_ENABLED=false` |
| `Activate.ps1 ... cannot be loaded` | Chạy `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` rồi kích hoạt lại |
| Cài `pip install` rất lâu | Bình thường — `sentence-transformers` kéo theo PyTorch khá nặng |
| `pip install` báo không tìm thấy wheel cho `torch`/`sentence-transformers` | Python quá mới so với thư viện. Tạo venv bằng Python 3.11 hoặc 3.12 rồi cài lại |
| `.venv` báo thiếu `pyvenv.cfg` | Môi trường ảo bị hỏng. Xoá thư mục đó và tạo lại bằng `python -m venv .venv` |
| Hỏi mà luôn báo "không tìm thấy" | Chưa nạp tài liệu, hoặc tài liệu là PDF scan (ảnh) nên không bóc được chữ |
| Tiếng Việt trả lời chưa hay | Thử `EMBEDDING_MODEL=intfloat/multilingual-e5-base` và tăng `TOP_K` |

---

## 8. Một hướng thay thế đáng biết

Nếu tài liệu **nhỏ** (vài chục trang), bạn có thể bỏ qua vector DB và **nhồi cả tài liệu**
vào Claude (cửa sổ ngữ cảnh tới 1 triệu token), kết hợp **prompt caching** (rẻ hơn khi
hỏi nhiều lần) và tính năng **citations** sẵn có của Claude. RAG (dự án này) là lựa chọn
đúng khi tài liệu **lớn dần** hoặc có **nhiều khoá học**.
