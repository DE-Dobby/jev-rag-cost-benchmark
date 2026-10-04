from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LocalCache:
    """(provider, model, prompt_version, sample_id) 키로 호출 결과를 JSON 파일에 캐시.

    캐시 적중 시 재호출하지 않는다. 키의 각 요소는 파일 경로 세그먼트로 쓰이므로
    디렉터리 트래버설을 막기 위해 '/'를 포함할 수 없다.
    """

    def __init__(self, root_dir: str | Path):
        self._root = Path(root_dir)

    def _path(self, provider: str, model: str, prompt_version: str, sample_id: str) -> Path:
        for part in (provider, model, prompt_version, sample_id):
            if "/" in part or ".." in part:
                raise ValueError(f"캐시 키에 올바르지 않은 문자가 있습니다: {part!r}")
        return self._root / provider / model / prompt_version / f"{sample_id}.json"

    def get(self, provider: str, model: str, prompt_version: str, sample_id: str) -> dict[str, Any] | None:
        path = self._path(provider, model, prompt_version, sample_id)
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    def put(self, provider: str, model: str, prompt_version: str, sample_id: str, value: dict[str, Any]) -> None:
        path = self._path(provider, model, prompt_version, sample_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)
        tmp_path.replace(path)
