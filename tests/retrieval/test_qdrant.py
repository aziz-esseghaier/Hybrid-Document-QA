"""Tests for Qdrant helpers; snapshot transfer runs against a small local HTTP server."""

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

from doc_qa.retrieval.qdrant import file_sha256, fits_in_point, restore_snapshot, save_snapshot, visual_collection

DATA = b"snapshot bytes" * 1000
SHA256 = hashlib.sha256(DATA).hexdigest()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, body):
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.server.paths.append(self.path)
        self.reply(200, DATA)

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        url = urlparse(self.path)
        self.server.uploads.append((url.path, parse_qs(url.query), body))
        self.reply(self.server.upload_status, b'{"status": "see server"}')


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.paths, httpd.uploads, httpd.upload_status = [], [], 200
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    httpd.url = f"http://127.0.0.1:{httpd.server_port}"
    yield httpd
    httpd.shutdown()
    httpd.server_close()


class Client:
    def __init__(self, checksum):
        self.snapshot = SimpleNamespace(name="c-1.snapshot", checksum=checksum)
        self.deleted = []

    def create_snapshot(self, name, wait):
        return self.snapshot

    def delete_snapshot(self, name, snapshot_name, wait):
        self.deleted.append((name, snapshot_name))


def test_visual_collection_name():
    assert visual_collection("visual", "v2_esg_reports") == "visual_v2_esg_reports"
    assert visual_collection("visual_tomoro", "v2_esg_reports") == "visual_tomoro_v2_esg_reports"


def test_fits_in_point_matches_the_server_limit():
    assert fits_in_point(409, 2560)
    assert not fits_in_point(410, 2560)


def test_file_sha256(tmp_path):
    path = tmp_path / "f"
    path.write_bytes(DATA)
    assert file_sha256(path) == SHA256


def test_save_snapshot_downloads_and_cleans_up(tmp_path, server):
    pytest.importorskip("httpx")
    client, path = Client(SHA256), tmp_path / "c.snapshot"
    assert save_snapshot(client, server.url, "text_a-b", path) == SHA256
    assert path.read_bytes() == DATA
    assert server.paths == ["/collections/text_a-b/snapshots/c-1.snapshot"]
    assert client.deleted == [("text_a-b", "c-1.snapshot")]


def test_save_snapshot_rejects_a_corrupt_download(tmp_path, server):
    pytest.importorskip("httpx")
    client = Client("0" * 64)
    with pytest.raises(RuntimeError, match="checksum"):
        save_snapshot(client, server.url, "c", tmp_path / "c.snapshot")
    assert client.deleted == [("c", "c-1.snapshot")]


def test_restore_snapshot_uploads_the_file(tmp_path, server):
    pytest.importorskip("httpx")
    path = tmp_path / "c.snapshot"
    path.write_bytes(DATA)
    restore_snapshot(server.url, "text_a", path, SHA256)
    url_path, query, body = server.uploads[0]
    assert url_path == "/collections/text_a/snapshots/upload"
    assert query == {"wait": ["true"], "priority": ["snapshot"], "checksum": [SHA256]}
    assert b'name="snapshot"' in body and DATA in body


def test_restore_snapshot_reports_server_errors(tmp_path, server):
    pytest.importorskip("httpx")
    server.upload_status = 400
    path = tmp_path / "c.snapshot"
    path.write_bytes(DATA)
    with pytest.raises(RuntimeError, match="400"):
        restore_snapshot(server.url, "c", path, SHA256)
