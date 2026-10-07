# Vietnamese Legal RAG Chatbot

Chatbot hỏi đáp pháp luật lao động Việt Nam sử dụng FastAPI, LangChain,
Qdrant, BM25, BGE embedding/reranker và OpenAI hoặc Gemini để sinh câu trả lời.

## Yêu cầu

- Python 3.12
- Một Qdrant collection có thể truy cập từ máy chạy ứng dụng
- OpenAI API key cho pre-retrieval
- OpenAI hoặc Gemini API key cho bước sinh câu trả lời
- Docker Desktop nếu chạy theo cách khuyến nghị

## Cấu hình

Tạo file môi trường từ file mẫu:

```bash
cp chatbot/.env.example chatbot/.env
```

Các biến quan trọng:

```env
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8001
SECRET_KEY=thay-bang-mot-chuoi-bi-mat-it-nhat-32-ky-tu

LLM_PROVIDER=gemini
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
PRE_RETRIEVAL_MODEL=gpt-4o-mini
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash

QDRANT_URL=
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=HTTM

BM25_STORE_PATH=./bm25/bm25_index
RERANKER_PROVIDER=bge
```

Nếu `LLM_PROVIDER=openai`, cần điền `OPENAI_API_KEY`. Nếu dùng Gemini,
điền thêm `GEMINI_API_KEY`. Không commit file `.env` lên Git.

## Chạy toàn bộ bằng Docker Compose (khuyến nghị)

Đảm bảo Docker Desktop đang chạy và file `chatbot/.env` đã được cấu hình. Từ
thư mục gốc của repository, khởi động cả chatbot và frontend bằng một lệnh:

```bash
docker compose up
```

Ở lần đầu, Docker sẽ build image và tải model cần thiết. Khi cả hai service đã
khởi động:

- Frontend: http://localhost:5173
- Swagger API: http://localhost:8001/docs
- Health check: http://localhost:8001/api/health

Nhấn `Ctrl+C` để dừng, sau đó xóa các container và network bằng:

```bash
docker compose down
```

Muốn chạy dưới nền:

```bash
docker compose up -d
docker compose logs -f
```

### Hot reload khi phát triển

- Khi sửa file Python trong `chatbot/app/`, Uvicorn tự phát hiện thay đổi và
  khởi động lại backend. Pipeline/model sẽ được nạp lại nên có thể mất một lúc.
- Khi sửa `frontend/index.html`, `frontend/styles.css` hoặc `frontend/app.js`,
  dev server tự refresh trang đang mở trong trình duyệt.
- Thay đổi source code thông thường không cần chạy lại `docker compose up`.
- Khi sửa `requirements.txt`, `Dockerfile`, `compose.yaml` hoặc dependency, cần
  build lại bằng:

```bash
docker compose up --build
```

- Khi sửa `chatbot/.env`, recreate service để biến môi trường mới được áp dụng:

```bash
docker compose up -d --force-recreate chatbot
```

Volume `hf-cache` giữ model Hugging Face giữa các lần chạy nên các lần khởi động
sau nhanh hơn. Nếu volume này đã được tạo bởi lệnh `docker run` trước đây,
Compose có thể cảnh báo rằng volume không do Compose tạo; cảnh báo này không ảnh
hưởng tới việc chạy và cache cũ vẫn được tái sử dụng.

## Chạy trực tiếp không dùng Docker

Tạo virtual environment và cài dependency:

```bash
cd chatbot
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Index dữ liệu vào Qdrant và BM25 trước lần chạy đầu tiên:

```bash
python -c "import asyncio; from app.indexing.indexer import get_indexer; asyncio.run(get_indexer().index_directory('data'))"
```

Khởi động FastAPI:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

API chạy tại `http://localhost:8001`.

## Chạy riêng backend bằng Docker

Docker image sử dụng Qdrant Cloud đã được index từ trước và BM25 index có sẵn
trong `chatbot/bm25/bm25_index`. Không cần index lại khi build hoặc khởi động
container.

Đảm bảo file `.env` chứa thông tin kết nối Qdrant Cloud, sau đó build image từ
thư mục gốc của repository:

```bash
docker build -t legal-chatbot ./chatbot
```

Khởi động API:

```bash
docker run --rm \
  --name legal-chatbot \
  -p 8001:8001 \
  --env-file chatbot/.env \
  -v hf-cache:/app/.cache/huggingface \
  legal-chatbot
```

Volume `hf-cache` giữ lại model Hugging Face giữa các lần chạy. Image sử dụng
một Uvicorn worker để tránh nạp nhiều bản embedding và reranker model vào RAM.

Sau khi API được triển khai, người sử dụng chỉ cần địa chỉ API do người vận
hành cung cấp; họ không cần cài Python, Docker, Qdrant hoặc index dữ liệu.

## Kiểm tra API

Kiểm tra trạng thái:

```bash
curl http://localhost:8001/api/health
```

Gửi câu hỏi:

```bash
curl -X POST http://localhost:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Người lao động được nghỉ phép năm bao nhiêu ngày?"}'
```

Swagger UI có tại `http://localhost:8001/docs`.

## Lưu ý

- Lần khởi động đầu có thể lâu vì ứng dụng cần tải BGE embedding và reranker.
- Chỉ cần index lại khi tài liệu nguồn thay đổi.
- Nếu thiếu bộ nhớ GPU, giảm `EMBEDDING_BATCH_SIZE` hoặc chạy model trên CPU.
- BM25 index là dữ liệu cục bộ; Qdrant chứa dense vector và phải luôn truy cập
  được trong lúc ứng dụng hoạt động.
