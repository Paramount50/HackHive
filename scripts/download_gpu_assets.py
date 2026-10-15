"""Resumable, hash-verified range downloads for the two large setup assets.

Useful when a proxy buffers large full-file responses. Only fixed official
PyTorch/Hugging Face URLs are allowed; inference itself remains offline.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import html
from pathlib import Path
import re
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.inference.smolvlm import MODEL_ID, MODEL_REVISION

CHUNK = 8 * 1024 * 1024


def download(url: str, destination: Path, expected_sha256: str):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        with destination.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() == expected_sha256:
                print(f"Already verified: {destination}", flush=True)
                return
        raise ValueError(f"Existing destination failed checksum: {destination}")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=45) as response:
        if response.status != 206:
            raise ValueError("server did not honor byte range")
        total = int(response.headers["Content-Range"].split("/")[-1])
        response.read()
    parts = Path("data/download-parts") / expected_sha256
    parts.mkdir(parents=True, exist_ok=True)
    count = (total + CHUNK - 1) // CHUNK

    def fetch(index):
        start, end = index * CHUNK, min(total, (index + 1) * CHUNK) - 1
        path = parts / f"{index:05d}.part"
        if path.exists() and path.stat().st_size == end - start + 1:
            return
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=45) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
                        raise ValueError("server returned an unexpected range")
                    data = response.read(end - start + 2)
                if len(data) != end - start + 1:
                    raise ValueError("incomplete range")
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(data)
                temporary.replace(path)
                return
            except Exception:
                if attempt == 2:
                    raise

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, index) for index in range(count)]
        try:
            for completed, future in enumerate(as_completed(futures), 1):
                future.result()
                if completed % 8 == 0 or completed == count:
                    print(f"{destination.name}: {completed}/{count} chunks", flush=True)
        except Exception:
            for future in futures:
                future.cancel()
            raise
    temporary = destination.with_suffix(destination.suffix + ".partial")
    digest = hashlib.sha256()
    with temporary.open("wb") as stream:
        for index in range(count):
            data = (parts / f"{index:05d}.part").read_bytes()
            stream.write(data)
            digest.update(data)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("download SHA256 mismatch; final asset was not installed")
    temporary.replace(destination)
    # Remove only the known part files after the verified destination exists.
    for index in range(count):
        (parts / f"{index:05d}.part").unlink()
    print(f"Verified {destination} ({total} bytes)", flush=True)


def main():
    resource = sys.argv[1] if len(sys.argv) == 2 else ""
    if resource == "torch":
        filename = "torch-2.9.1+cu126-cp312-cp312-win_amd64.whl"
        index = "https://download.pytorch.org/whl/cu126/torch/"
        with urllib.request.urlopen(index, timeout=45) as response:
            page = response.read().decode()
        links = [html.unescape(link) for link in re.findall(r'href="([^"]+)"', page)]
        link = next(link for link in links if urllib.parse.unquote(link.split("#")[0]).endswith("/" + filename))
        url = urllib.parse.urljoin(index, link.split("#")[0])
        digest = link.split("#sha256=")[1]
        download(url, Path("data/wheels") / filename, digest)
    elif resource == "model":
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/model.safetensors"
        download(url, Path("models/smolvlm2-500m/model.safetensors"),
                 "b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3")
        # Finish tokenizer/config downloads and write provenance through the normal setup command.
    else:
        raise SystemExit("Usage: download_gpu_assets.py torch|model")


if __name__ == "__main__":
    main()
