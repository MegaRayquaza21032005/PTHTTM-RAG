# Legal Chatbot UI

Giao diện HTML/CSS/JavaScript độc lập để thử nghiệm API chatbot tại
`http://localhost:8001/api`.

## Chạy bằng Docker

Từ thư mục gốc của repository:

```bash
docker build -t legal-chatbot-ui frontend
docker run --rm --name legal-chatbot-ui -p 5173:5173 legal-chatbot-ui
```

Mở `http://localhost:5173` trên trình duyệt. Backend `legal-chatbot` cần đang
chạy tại cổng `8001`.

Để chạy frontend dưới nền:

```bash
docker run --rm -d --name legal-chatbot-ui -p 5173:5173 legal-chatbot-ui
```

## Kiểm thử cấu trúc

```powershell
./frontend/tests/validate.ps1
```
