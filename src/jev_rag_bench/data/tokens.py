from __future__ import annotations

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    # Phase 1 통계용 근사 토크나이저. 실제 과금 토큰 수는 Phase 2~3에서
    # 각 API 응답 메타데이터로부터 읽어 pricing.yaml 단가와 곱해 계산한다 (여기서는 근사치일 뿐).
    return tiktoken.get_encoding("cl100k_base")


def approx_token_count(text: str) -> int:
    return len(_encoder().encode(text))
