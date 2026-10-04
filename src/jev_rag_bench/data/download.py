from __future__ import annotations

from pathlib import Path

import httpx

_BASE_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset"
_FILES = ["response.jsonl", "source_info.jsonl"]


def download_ragtruth(raw_dir: Path, *, force: bool = False) -> None:
    """RAGTruth의 response.jsonl / source_info.jsonl을 raw_dir에 받는다.

    이미 받아져 있으면 재다운로드하지 않는다 (force=True면 덮어씀).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for filename in _FILES:
            dest = raw_dir / filename
            if dest.exists() and not force:
                continue
            resp = client.get(f"{_BASE_URL}/{filename}")
            resp.raise_for_status()
            dest.write_bytes(resp.content)
