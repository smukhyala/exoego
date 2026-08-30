"""Parallel range downloader for large HuggingFace files.

`hf download` routes these through the xet backend, which on this network
buffered gigabytes in RAM and wrote nothing to disk for ten minutes. This does
plain HTTP range requests straight to a file, with resume and visible progress.

Usage:
    python -m src.fetch TSM_features/HMC_84346135_mono10bit.zip [more...]
"""

from __future__ import annotations

import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

from huggingface_hub import get_token

REPO = "cvml-nus/assembly101"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
OUT = Path("data/assembly101")
CONNECTIONS = 8


def _size(url: str, tok: str) -> int:
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"Authorization": f"Bearer {tok}"})
    with urllib.request.urlopen(req) as r:
        return int(r.headers["Content-Length"])


def _part(url: str, tok: str, fh, lock, start: int, end: int, done: list, idx: int):
    pos = start
    while pos <= end:
        try:
            req = urllib.request.Request(
                url, headers={"Authorization": f"Bearer {tok}",
                              "Range": f"bytes={pos}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    with lock:
                        fh.seek(pos)
                        fh.write(b)
                    pos += len(b)
                    done[idx] += len(b)
        except Exception:
            time.sleep(2)  # transient; resume from pos
    return


def fetch(rel: str) -> Path:
    tok = get_token()
    url = f"{BASE}/{rel}"
    dest = OUT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = _size(url, tok)

    if dest.exists() and dest.stat().st_size == total:
        print(f"  {rel}  already complete ({total/1e9:.2f} GB)")
        return dest

    print(f"  {rel}  {total/1e9:.2f} GB  via {CONNECTIONS} connections")
    with open(dest, "wb") as fh:
        fh.truncate(total)
        lock = threading.Lock()
        done = [0] * CONNECTIONS
        span = total // CONNECTIONS
        threads = []
        for i in range(CONNECTIONS):
            s = i * span
            e = total - 1 if i == CONNECTIONS - 1 else (i + 1) * span - 1
            t = threading.Thread(target=_part, args=(url, tok, fh, lock, s, e, done, i),
                                 daemon=True)
            t.start()
            threads.append(t)

        t0 = time.time()
        while any(t.is_alive() for t in threads):
            time.sleep(5)
            got = sum(done)
            el = time.time() - t0
            sp = got / 1e6 / max(el, 1e-9)
            eta = (total - got) / 1e6 / sp / 60 if sp > 0 else float("inf")
            pct = 100 * got / total
            print(f"    {pct:5.1f}%  {got/1e9:6.2f}/{total/1e9:.2f} GB  "
                  f"{sp:5.1f} MB/s  eta {eta:4.1f} min", flush=True)
        for t in threads:
            t.join()

    got = dest.stat().st_size
    print(f"  {rel}  done ({got/1e9:.2f} GB)")
    return dest


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for rel in sys.argv[1:]:
        fetch(rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
