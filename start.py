from __future__ import annotations

import argparse
import json
import re
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from create_conversation_api import create_conversation
from history_api import list_conversations
from list_messages_api import list_messages
from model_list_api import list_models
from model_client import ModelBusyError
from render_message_api import render_message
from send_message_api import send_message
from switch_model_api import switch_model
from upload_file_api import upload_file


class ChatHandler(BaseHTTPRequestHandler):
    ui_file: Path
    config_path: Path

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            if path in ("/", "/webui.html"):
                self._serve_ui()
                return
            if path == "/api/history":
                self._ok({"conversations": list_conversations()})
                return
            if path == "/api/models":
                self._ok({"models": list_models(config_path=self.config_path)})
                return

            match = re.fullmatch(r"/api/conversations/([^/]+)/messages", path)
            if match:
                conversation_id = match.group(1)
                messages = [
                    render_message(message)
                    for message in list_messages(conversation_id=conversation_id)
                ]
                self._ok({"messages": messages})
                return

            self._error(HTTPStatus.NOT_FOUND, "route_not_found")
        except (ValueError, KeyError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, "bad_request", detail=str(exc))
        except ModelBusyError as exc:
            self._error(HTTPStatus.TOO_MANY_REQUESTS, "server_busy", detail=str(exc))
        except Exception as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error", detail=str(exc))

    def do_POST(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/api/conversations":
                payload = self._read_json()
                conversation = create_conversation(
                    title=payload.get("title"),
                    model=payload.get("model"),
                )
                self._ok(conversation, status=HTTPStatus.CREATED)
                return

            if path == "/api/messages":
                payload = self._read_json()
                reply = send_message(
                    conversation_id=str(payload.get("conversationId", "")).strip(),
                    message=str(payload.get("message", "")),
                    file_ids=payload.get("fileIds") or [],
                    model=payload.get("model"),
                )
                self._ok({"message": render_message(reply)})
                return

            if path == "/api/model/switch":
                payload = self._read_json()
                model_name = switch_model(
                    conversation_id=(str(payload.get("conversationId", "")).strip() or None),
                    model=str(payload.get("model", "")),
                )
                self._ok({"model": model_name})
                return

            if path == "/api/files":
                filename, content, content_type = self._read_single_file(field_name="file")
                file_info = upload_file(
                    filename=filename,
                    content=content,
                    content_type=content_type,
                )
                self._ok(file_info, status=HTTPStatus.CREATED)
                return

            self._error(HTTPStatus.NOT_FOUND, "route_not_found")
        except (ValueError, KeyError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, "bad_request", detail=str(exc))
        except ModelBusyError as exc:
            self._error(HTTPStatus.TOO_MANY_REQUESTS, "server_busy", detail=str(exc))
        except Exception as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error", detail=str(exc))

    def _serve_ui(self) -> None:
        if not self.ui_file.exists():
            self._error(HTTPStatus.NOT_FOUND, "webui_not_found")
            return
        data = self.ui_file.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _ok(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: HTTPStatus, code: str, detail: str | None = None) -> None:
        payload = {"error": code}
        if detail:
            payload["detail"] = detail
        self._ok(payload, status=status)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError as exc:
            raise ValueError("invalid_json") from exc

    def _read_single_file(self, *, field_name: str) -> tuple[str, bytes, str | None]:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise ValueError("content_type_must_be_multipart_form_data")

        boundary_match = re.search(r'boundary="?([^";]+)"?', content_type)
        if not boundary_match:
            raise ValueError("missing_multipart_boundary")
        boundary = boundary_match.group(1).encode("utf-8")

        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length > 0 else b""
        if not raw:
            raise ValueError("empty_multipart_body")

        marker = b"--" + boundary
        parts = raw.split(marker)
        for part in parts:
            if not part:
                continue
            if part.startswith(b"\r\n"):
                part = part[2:]
            if part.endswith(b"--\r\n"):
                part = part[:-4]
            elif part.endswith(b"\r\n"):
                part = part[:-2]
            elif part == b"--":
                continue

            header_blob, sep, body = part.partition(b"\r\n\r\n")
            if sep == b"":
                continue
            headers_text = header_blob.decode("utf-8", errors="ignore")
            disposition_match = re.search(
                r'Content-Disposition:\s*form-data;\s*name="([^"]+)"(?:;\s*filename="([^"]*)")?',
                headers_text,
                flags=re.IGNORECASE,
            )
            if not disposition_match:
                continue
            name, filename = disposition_match.group(1), disposition_match.group(2)
            if name != field_name:
                continue
            file_content = body
            file_type_match = re.search(
                r"Content-Type:\s*([^\r\n]+)", headers_text, flags=re.IGNORECASE
            )
            file_type = file_type_match.group(1).strip() if file_type_match else None
            if file_content.endswith(b"\r\n"):
                file_content = file_content[:-2]
            return filename or "upload.bin", file_content, file_type

        raise ValueError("file_field_not_found")

    def log_message(self, format_: str, *args: Any) -> None:
        return


def make_handler(ui_file: Path, config_path: Path) -> type[ChatHandler]:
    class BoundHandler(ChatHandler):
        pass

    BoundHandler.ui_file = ui_file
    BoundHandler.config_path = config_path
    return BoundHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cloud LLM Chat startup entry")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host, default 127.0.0.1")
    parser.add_argument("--port", type=int, default=8000, help="Bind port, default 8000")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open browser automatically after startup",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ui_file = Path(__file__).with_name("webui.html")
    config_path = Path(__file__).with_name("config.json")
    handler_cls = make_handler(ui_file=ui_file, config_path=config_path)

    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    url = f"http://{args.host}:{args.port}/"
    print(f"CloudChat started at: {url}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("CloudChat stopped")


if __name__ == "__main__":
    main()
