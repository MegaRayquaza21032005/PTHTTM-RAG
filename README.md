# Vietnamese Legal RAG Chatbot

Chatbot hỏi đáp pháp luật lao động Việt Nam sử dụng FastAPI, LangChain,
Qdrant, BM25, BGE embedding/reranker và OpenAI hoặc Gemini để sinh câu trả lời.

## Yêu cầu

- Python 3.12
- Một Qdrant collection có thể truy cập từ máy chạy ứng dụng
- OpenAI API key cho pre-retrieval
- OpenAI hoặc Gemini API key cho bước sinh câu trả lời
- Docker nếu chạy theo cách 2

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

## Cách 1: Chạy trực tiếp

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

## Cách 2: Chạy bằng Docker

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
