"""Dependency-free development server with browser live reload."""

from __future__ import annotations

import os
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
PORT = int(os.getenv("FRONTEND_PORT", "5173"))
WATCHED_SUFFIXES = {".html", ".css", ".js"}

LIVE_RELOAD_CLIENT = """
<script>
(() => {
  const events = new EventSource('/__reload');
  events.addEventListener('change', () => window.location.reload());
})();
</script>
""".strip()


def source_snapshot() -> tuple[tuple[str, int, int], ...]:
    """Return a stable signature for browser assets in the frontend root."""
    files = (
        path
        for path in ROOT.iterdir()
        if path.is_file() and path.suffix.lower() in WATCHED_SUFFIXES
    )
    return tuple(
        sorted(
            (path.name, path.stat().st_mtime_ns, path.stat().st_size)
            for path in files
        )
    )


class DevelopmentHandler(SimpleHTTPRequestHandler):
    """Serve static assets and notify browsers when a source file changes."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:  # noqa: N802 - inherited HTTP handler API
        path = urlparse(self.path).path
        if path == "/__reload":
            self._serve_reload_stream()
            return
        if path in {"/", "/index.html"}:
            self._serve_development_index()
            return
        super().do_GET()

    def _serve_development_index(self) -> None:
        index_path = ROOT / "index.html"
        html = index_path.read_text(encoding="utf-8")
        html = html.replace("</body>", f"{LIVE_RELOAD_CLIENT}\n</body>")
        payload = html.encode("utf-8")

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _serve_reload_stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        initial = source_snapshot()
        heartbeat_at = time.monotonic()

        try:
            while True:
                time.sleep(0.4)
                if source_snapshot() != initial:
                    self.wfile.write(b"event: change\ndata: reload\n\n")
                    self.wfile.flush()
                    return

                if time.monotonic() - heartbeat_at >= 15:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    heartbeat_at = time.monotonic()
        except (BrokenPipeError, ConnectionResetError):
            return


class DevelopmentServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    server = DevelopmentServer(("0.0.0.0", PORT), DevelopmentHandler)
    print(f"Frontend dev server: http://0.0.0.0:{PORT} (live reload enabled)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
