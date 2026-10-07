FROM python:3.12-slim

WORKDIR /app

COPY index.html styles.css app.js dev_server.py ./

EXPOSE 5173

HEALTHCHECK --interval=30s --timeout=3s --start-period=3s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5173', timeout=2)"]

CMD ["python", "-m", "http.server", "5173", "--bind", "0.0.0.0", "--directory", "/app"]
