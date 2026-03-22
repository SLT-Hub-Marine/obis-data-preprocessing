#!/usr/bin/env python3
"""
Simple HTTP server with Range request support for PMTiles.
Python's built-in http.server doesn't support byte-range requests,
which PMTiles requires to read individual tiles from the archive.
"""

import os
import sys
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler


class RangeHTTPRequestHandler(SimpleHTTPRequestHandler):
    """HTTP handler that supports Range requests and CORS."""

    def send_head(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().send_head()

        # Check for Range header
        range_header = self.headers.get("Range")
        if range_header is None:
            return super().send_head()

        # Parse Range: bytes=start-end
        try:
            range_spec = range_header.replace("bytes=", "")
            parts = range_spec.split("-")
            file_size = os.path.getsize(path)
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else file_size - 1
            end = min(end, file_size - 1)
            length = end - start + 1
        except (ValueError, IndexError):
            self.send_error(416, "Invalid Range")
            return None

        ctype = self.guess_type(path)
        f = open(path, "rb")
        f.seek(start)

        self.send_response(206)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        # Return a wrapper that only reads `length` bytes
        return _LimitedReader(f, length)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Content-Length, Accept-Ranges")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


class _LimitedReader:
    """File-like wrapper that limits reads to N bytes."""

    def __init__(self, fh, limit):
        self._fh = fh
        self._remaining = limit

    def read(self, size=-1):
        if self._remaining <= 0:
            return b""
        if size < 0 or size > self._remaining:
            size = self._remaining
        data = self._fh.read(size)
        self._remaining -= len(data)
        return data

    def close(self):
        self._fh.close()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    directory = sys.argv[2] if len(sys.argv) > 2 else "."

    handler = partial(RangeHTTPRequestHandler, directory=directory)
    server = HTTPServer(("0.0.0.0", port), handler)
    print(f"🌐 Serving {directory} at http://localhost:{port}")
    print(f"   Range requests: ✅ supported")
    print(f"   Open: http://localhost:{port}/interactive_map.html")
    print(f"   Press Ctrl+C to stop")
    server.serve_forever()


if __name__ == "__main__":
    main()
