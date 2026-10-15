import hashlib
from io import BytesIO

import pytest

from scripts import download_gpu_assets as downloader


class Response(BytesIO):
    status = 206

    def __init__(self, data, start, end, total):
        super().__init__(data)
        self.headers = {"Content-Range": f"bytes {start}-{end}/{total}"}


def install_fake_server(monkeypatch, data):
    requests = []
    def open_request(request, timeout):
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        requests.append((start, end))
        return Response(data[start:end+1], start, end, len(data))
    monkeypatch.setattr(downloader.urllib.request, "urlopen", open_request)
    monkeypatch.setattr(downloader, "CHUNK", 4)
    return requests


def test_resumes_known_parts_and_installs_only_verified_asset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = b"0123456789abcdef"
    digest = hashlib.sha256(data).hexdigest()
    requests = install_fake_server(monkeypatch, data)
    parts = tmp_path / "data/download-parts" / digest
    parts.mkdir(parents=True)
    (parts / "00000.part").write_bytes(data[:4])
    destination = tmp_path / "asset.bin"
    downloader.download("https://example.invalid/asset", destination, digest)
    assert destination.read_bytes() == data
    assert (0, 3) not in requests
    assert not list(parts.glob("*.part"))


def test_checksum_failure_never_installs_asset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    install_fake_server(monkeypatch, b"corrupt data")
    destination = tmp_path / "asset.bin"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        downloader.download("https://example.invalid/asset", destination, "0" * 64)
    assert not destination.exists()
