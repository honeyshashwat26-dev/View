#!/usr/bin/env python3
"""
Live viewer for HoneyXCamera HTML.

  POST /frame   <- JPEG bytes from the browser (pushFrameToViewer)
  GET  /stream  -> MJPEG stream (open in a browser tab)
  GET  /        -> simple page embedding /stream
  GET  /health  -> "ok"

Also opens an OpenCV window (cv2.imshow) with the live feed.
Run:  python viewer.py
"""
import io
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np

HOST, PORT = "127.0.0.1", 8000

# Latest JPEG frame, shared between threads
_frame_lock = threading.Lock()
_latest_jpeg: Optional[bytes] = None
_last_frame_ts = 0.0


def set_frame(jpeg: bytes) -> None:
    global _latest_jpeg, _last_frame_ts
    with _frame_lock:
        _latest_jpeg = jpeg
        _last_frame_ts = time.time()


def get_frame() -> Optional[bytes]:
    with _frame_lock:
        return _latest_jpeg


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        if self.path != "/frame":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            self.send_error(400, "empty body")
            return
        data = self.rfile.read(length)
        set_frame(data)
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path == "/":
            body = (
                b"<!doctype html><title>HoneyXCamera Live</title>"
                b"<body style='margin:0;background:#111;display:flex;"
                b"align-items:center;justify-content:center;height:100vh'>"
                b"<img src='/stream' style='max-width:100%;max-height:100%'></body>"
            )
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/stream":
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    jpeg = get_frame()
                    if jpeg:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif self.path == "/health":
            body = b"ok"
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)


def serve():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[viewer] HTTP server on http://{HOST}:{PORT}")
    print(f"[viewer]   browser stream: http://{HOST}:{PORT}/stream")
    print(f"[viewer]   HTML should POST to http://{HOST}:{PORT}/frame")
    server.serve_forever()


def show_with_opencv():
    print("[viewer] OpenCV window — press 'q' or ESC to close (server keeps running).")
    cv2.namedWindow("HoneyXCamera Live", cv2.WINDOW_NORMAL)
    last_bytes = None
    while True:
        jpeg = get_frame()
        if jpeg is None or jpeg is last_bytes:
            time.sleep(0.02)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            continue
        last_bytes = jpeg
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            age = time.time() - _last_frame_ts
            color = (0, 200, 0) if age < 2.0 else (0, 0, 255)
            cv2.circle(img, (15, 15), 8, color, -1)
            cv2.putText(img, time.strftime("%H:%M:%S"), (28, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.imshow("HoneyXCamera Live", img)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    t = threading.Thread(target=serve, daemon=True)
    t.start()
    try:
        show_with_opencv()
    except KeyboardInterrupt:
        pass
    print("[viewer] shutting down")