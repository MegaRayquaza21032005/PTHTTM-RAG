(() => {
  "use strict";

  const API_BASE = "http://localhost:8001/api";

  const form = document.querySelector("#question-form");
  const input = document.querySelector("#question-input");
  const sendButton = document.querySelector("#send-button");
  const messages = document.querySelector("#messages");
  const clearButton = document.querySelector("#clear-chat");
  const connectionStatus = document.querySelector("#connection-status");
  const characterCount = document.querySelector("#character-count");
  const loadingTemplate = document.querySelector("#loading-template");
  const welcomeMarkup = messages.innerHTML;

  let isSending = false;

  function normalizeText(value) {
    if (value === undefined || value === null) return "";
    return String(value).normalize("NFC").replace(/\u00a0/g, " ");
  }

  function createElement(tagName, className, text) {
    const element = document.createElement(tagName);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = normalizeText(text);
    return element;
  }

  function setConnectionState(state, label) {
    connectionStatus.className = `connection-status is-${state}`;
    connectionStatus.querySelector(".status-label").textContent = label;
  }

  async function checkHealth() {
    setConnectionState("checking", "Đang kiểm tra");

    try {
      const response = await fetch(`${API_BASE}/health`, {
        headers: { Accept: "application/json" },
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setConnectionState("online", "API sẵn sàng");
    } catch {
      setConnectionState("offline", "Mất kết nối API");
    }
  }

  function scrollToLatest() {
    messages.scrollTo({ top: messages.scrollHeight, behavior: "smooth" });
  }

  function resizeInput() {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
    characterCount.textContent = `${input.value.length} / 4096`;
  }

  function setSending(nextState) {
    isSending = nextState;
    input.disabled = nextState;
    sendButton.disabled = nextState;
  }

  function renderUser(question) {
    const article = createElement("article", "message user-message");
    const badge = createElement("div", "message-badge", "BẠN");
    badge.setAttribute("aria-hidden", "true");
    const content = createElement("div", "message-content");
    content.append(
      createElement("p", "message-author", "CÂU HỎI CỦA BẠN"),
      createElement("p", "", question),
    );
    article.append(content, badge);
    messages.append(article);
  }

  function sourceLabel(source, index) {
    const metadata = source?.metadata ?? {};
    return metadata.dieu || metadata.title || `Nguồn tham khảo ${index + 1}`;
  }

  function sourceDocument(source) {
    const metadata = source?.metadata ?? {};
    if (metadata.title && metadata.title !== metadata.dieu) return metadata.title;
    return metadata.source || `Đoạn tài liệu ${source?.chunk_index ?? 0}`;
  }

  function sourceMetadata(source) {
    const metadata = source?.metadata ?? {};
    const items = [metadata.chuong, metadata.muc].filter(Boolean);
    if (Number.isFinite(source?.score)) {
      const scoreLabel = source.score_type === "rerank"
        ? "Điểm Rerank:"
        : source.score_type === "rrf"
          ? "Điểm RRF:"
          : "Điểm thô:";
      const scoreMeaning = source.score_type === "rrf"
        ? "điểm xếp hạng"
        : "điểm mô hình thô";
      items.push(
        `${scoreLabel} ${source.score.toFixed(4)} (${scoreMeaning}, không phải %)`,
      );
    }
    return items;
  }

  function renderSources(sources) {
    const wrapper = createElement("div", "sources");
    wrapper.append(createElement("p", "sources-heading", `NGUỒN THAM KHẢO · ${sources.length}`));
    wrapper.append(createElement(
      "p",
      "source-score-note",
      "Điểm nguồn là điểm xếp hạng nội bộ, không phải % độ chính xác.",
    ));

    sources.forEach((source, index) => {
      const details = createElement("details", "source-item");
      const summary = document.createElement("summary");
      const identity = createElement("span", "source-identity");
      identity.append(
        createElement("span", "source-title", sourceLabel(source, index)),
        createElement("span", "source-document", sourceDocument(source)),
      );
      summary.append(
        createElement("span", "source-index", String(index + 1).padStart(2, "0")),
        identity,
        createElement("span", "source-toggle", "Xem trích dẫn"),
      );

      const body = createElement("div", "source-body");
      const metadata = createElement("div", "source-meta");
      sourceMetadata(source).forEach((item) => {
        metadata.append(createElement("span", "source-meta-item", item));
      });
      const content = createElement(
        "p",
        "source-content",
        source?.content || "Không có nội dung trích dẫn.",
      );
      if (metadata.childElementCount > 0) body.append(metadata);
      body.append(content);
      details.append(summary, body);
      details.addEventListener("toggle", () => {
        summary.querySelector(".source-toggle").textContent = details.open
          ? "Thu gọn"
          : "Xem trích dẫn";
      });
      wrapper.append(details);
    });

    return wrapper;
  }

  function renderAssistant(answer, sources = []) {
    const article = createElement("article", "message assistant-message");
    const badge = createElement("div", "message-badge", "PĐ");
    badge.setAttribute("aria-hidden", "true");
    const content = createElement("div", "message-content");
    content.append(
      createElement("p", "message-author", "PHÁP ĐIỂN TRẢ LỜI"),
      createElement("p", "", answer),
    );
    if (Array.isArray(sources) && sources.length > 0) {
      content.append(renderSources(sources));
    }
    article.append(badge, content);
    messages.append(article);
  }

  function errorMessage(detail) {
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail.map((item) => item?.msg).filter(Boolean).join("; ");
    }
    return "Không thể xử lý câu hỏi. Hãy kiểm tra backend và thử lại.";
  }

  function renderError(detail) {
    const article = createElement("article", "message assistant-message error-message");
    const badge = createElement("div", "message-badge", "!");
    badge.setAttribute("aria-hidden", "true");
    const content = createElement("div", "message-content");
    content.append(
      createElement("p", "message-author", "KHÔNG THỂ HOÀN TẤT"),
      createElement("p", "", errorMessage(detail)),
    );
    article.append(badge, content);
    messages.append(article);
  }

  function addLoadingMessage() {
    const loading = loadingTemplate.content.firstElementChild.cloneNode(true);
    messages.append(loading);
    return loading;
  }

  async function sendQuestion(question) {
    renderUser(question);
    const loading = addLoadingMessage();
    setSending(true);
    scrollToLatest();

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json; charset=utf-8",
        },
        body: JSON.stringify({ question }),
      });

      const payload = await response.json().catch(() => ({}));
      loading.remove();

      if (!response.ok) {
        renderError(payload.detail || `API trả về lỗi HTTP ${response.status}.`);
        if (response.status >= 500) setConnectionState("offline", "API gặp lỗi");
        return;
      }

      renderAssistant(payload.answer, payload.sources);
      setConnectionState("online", "API sẵn sàng");
    } catch {
      loading.remove();
      renderError("Không kết nối được tới API tại localhost:8001. Hãy kiểm tra container backend.");
      setConnectionState("offline", "Mất kết nối API");
    } finally {
      setSending(false);
      input.focus();
      scrollToLatest();
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question || isSending) return;

    input.value = "";
    resizeInput();
    sendQuestion(question);
  });

  input.addEventListener("input", resizeInput);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  clearButton.addEventListener("click", () => {
    if (isSending) return;
    messages.innerHTML = welcomeMarkup;
    bindSuggestions();
    input.focus();
  });

  function bindSuggestions() {
    messages.querySelectorAll(".suggestion").forEach((button) => {
      button.addEventListener("click", () => {
        input.value = button.textContent.trim();
        resizeInput();
        form.requestSubmit();
      });
    });
  }

  bindSuggestions();
  resizeInput();
  checkHealth();
})();
