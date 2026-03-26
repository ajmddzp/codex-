from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from model_client import (
    ModelClientError,
    chat_completion,
    compose_messages_for_request,
    list_candidate_models as mc_list_candidate_models,
    load_model_config,
)


BASE_DIR = Path(__file__).resolve().parent
WEBUI_PATH = BASE_DIR / "webui.html"
CONFIG_PATH = BASE_DIR / "config.json"
DEFAULT_MODELS = ["gpt-5.4", "gpt-5.3-codex", "gpt-5.2"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def normalize_model_name(model: object) -> str:
    return str(model or "").strip().lower()


def load_candidate_models() -> list[str]:
    try:
        models = mc_list_candidate_models(CONFIG_PATH)
        normalized = [normalize_model_name(model) for model in models]
        return [m for m in normalized if m] or DEFAULT_MODELS[:]
    except Exception:
        return DEFAULT_MODELS[:]


def inject_runtime_config(html: str) -> str:
    if "<script>window.CLOUD_CHAT_ENDPOINTS =" in html:
        return html

    endpoints = {
        "listConversations": "/api/history",
        "createConversation": "/api/conversations",
        "sendMessage": "/api/messages",
        "uploadFile": "/api/files",
        "switchModel": "/api/model/switch",
        "listModels": "/api/models",
    }
    runtime_script = (
        "<script>window.CLOUD_CHAT_ENDPOINTS = "
        + json.dumps(endpoints, ensure_ascii=False)
        + ";</script>"
    )

    if "<script>" in html:
        return html.replace("<script>", runtime_script + "\n<script>", 1)
    return html + "\n" + runtime_script


def parse_json_body(handler: BaseHTTPRequestHandler) -> dict:
    raw = read_body(handler)
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}


def parse_filename_from_multipart(content_type: str, raw_body: bytes) -> str:
    if "multipart/form-data" not in content_type:
        return "unnamed_file"

    text = raw_body.decode("latin1", errors="ignore")
    match = re.search(r'filename="([^"]*)"', text)
    if not match:
        return "unnamed_file"
    filename = match.group(1).strip()
    return filename or "unnamed_file"


def read_body(handler: BaseHTTPRequestHandler) -> bytes:
    length_str = handler.headers.get("Content-Length", "0")
    try:
        length = int(length_str)
    except ValueError:
        length = 0
    if length <= 0:
        return b""
    return handler.rfile.read(length)


def build_model_try_order(requested_model: str, default_model: str, candidates: list[str]) -> list[str]:
    ordered = [requested_model, default_model, *candidates]
    out: list[str] = []
    seen: set[str] = set()
    for item in ordered:
        model = normalize_model_name(item)
        if not model:
            continue
        if model in seen:
            continue
        seen.add(model)
        out.append(model)
    return out


def is_non_retryable_error(error_text: str) -> bool:
    lower = error_text.lower()
    hard_fail_tokens = [
        "401",
        "403",
        "invalid_api_key",
        "unauthorized",
        "forbidden",
        "insufficient_quota",
        "billing",
    ]
    return any(token in lower for token in hard_fail_tokens)


class AppHandler(BaseHTTPRequestHandler):
    server_version = "CloudChatServer/0.4"
    protocol_version = "HTTP/1.1"

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._set_cors_headers()
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path

        if path in ("/", "/webui.html"):
            self._serve_webui()
            return

        if path == "/api/history":
            self._send_json(200, [])
            return

        if re.fullmatch(r"/api/conversations/[^/]+/messages", path):
            self._send_json(200, [])
            return

        if path == "/api/models":
            self._send_json(200, load_candidate_models())
            return

        if path == "/api/health":
            self._send_json(200, {"status": "ok", "mode": "api-ready"})
            return

        self._send_json(404, {"error": "Not Found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path

        if path == "/api/conversations":
            payload = parse_json_body(self)
            models = load_candidate_models()
            self._send_json(
                200,
                {
                    "id": f"todo_conv_{now_ms()}",
                    "title": str(payload.get("title") or "未命名对话"),
                    "model": normalize_model_name(payload.get("model") or models[0]),
                    "updated_at": utc_now_iso(),
                    "_todo": "Replace with create_conversation_api implementation.",
                },
            )
            return

        if path == "/api/messages":
            self._handle_send_message()
            return

        if path == "/api/files":
            content_type = self.headers.get("Content-Type", "")
            raw_body = read_body(self)
            filename = parse_filename_from_multipart(content_type, raw_body)
            self._send_json(
                200,
                {
                    "id": f"todo_file_{now_ms()}",
                    "name": filename,
                    "size": None,
                    "_todo": "Replace with upload_file_api implementation.",
                },
            )
            return

        if path == "/api/model/switch":
            payload = parse_json_body(self)
            self._send_json(
                200,
                {
                    "model": normalize_model_name(payload.get("model") or ""),
                    "_todo": "Replace with switch_model_api implementation.",
                },
            )
            return

        self._send_json(404, {"error": "Not Found"})

    def _handle_send_message(self) -> None:
        payload = parse_json_body(self)
        user_text = str(payload.get("message", "")).strip()
        if not user_text:
            self._send_json(400, {"error": "message is required"})
            return

        try:
            cfg = load_model_config(CONFIG_PATH)
        except ModelClientError as exc:
            self._send_json(500, {"error": "model_config_error", "detail": str(exc)})
            return

        requested_model = normalize_model_name(payload.get("model") or cfg.default_model)
        conversation_id = str(payload.get("conversationId", "")).strip()

        history_payload = payload.get("history")
        history_messages: list[dict[str, str]] = []
        if isinstance(history_payload, list):
            for item in history_payload:
                if not isinstance(item, dict):
                    continue
                role = item.get("role")
                content = str(item.get("content", "")).strip()
                if role in ("system", "user", "assistant") and content:
                    history_messages.append({"role": role, "content": content})

        model_messages = compose_messages_for_request(
            system_prompt=None,
            history_messages=history_messages,
            user_message=user_text,
        )

        try_models = build_model_try_order(
            requested_model,
            normalize_model_name(cfg.default_model),
            [normalize_model_name(model) for model in cfg.candidate_models],
        )
        attempts: list[dict[str, str]] = []
        used_model = ""
        reply_text = ""

        for candidate_model in try_models:
            try:
                reply_text = chat_completion(
                    config=cfg,
                    model=candidate_model,
                    messages=model_messages,
                    metadata={"conversation_id": conversation_id} if conversation_id else None,
                )
                used_model = candidate_model
                break
            except ModelClientError as exc:
                error_text = str(exc)
                attempts.append({"model": candidate_model, "error": error_text})
                if is_non_retryable_error(error_text):
                    break

        if not used_model:
            last_detail = attempts[-1]["error"] if attempts else "unknown model error"
            self._send_json(
                502,
                {
                    "error": "model_request_failed",
                    "detail": last_detail,
                    "attempts": attempts,
                },
            )
            return

        self._send_json(
            200,
            {
                "message": {
                    "id": f"todo_msg_{now_ms()}",
                    "role": "assistant",
                    "content": reply_text,
                    "created_at": utc_now_iso(),
                },
                "model": used_model,
                "fallback_used": used_model != requested_model,
                "attempts": attempts,
            },
        )

    def _serve_webui(self) -> None:
        if not WEBUI_PATH.exists():
            self._send_text(404, "webui.html not found")
            return

        html = WEBUI_PATH.read_text(encoding="utf-8")
        final_html = inject_runtime_config(html)
        self._send_bytes(
            200,
            final_html.encode("utf-8"),
            content_type="text/html; charset=utf-8",
        )

    def _send_json(self, status: int, payload: object) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, body, content_type="application/json; charset=utf-8")

    def _send_text(self, status: int, text: str) -> None:
        self._send_bytes(status, text.encode("utf-8"), content_type="text/plain; charset=utf-8")

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self._set_cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _set_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {format % args}")


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), AppHandler)
    print(f"Server started: http://{host}:{port}")
    print("Mode: api-ready (messages call cloud model with fallback retry)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloud Chat WebUI server (standard library).")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host, default 127.0.0.1")
    parser.add_argument("--port", default=8000, type=int, help="Bind port, default 8000")
    args = parser.parse_args()
    run(host=args.host, port=args.port)
