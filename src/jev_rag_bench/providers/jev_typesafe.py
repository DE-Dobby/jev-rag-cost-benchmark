from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


@dataclass
class JevResult:
    noul: float  # 환각(yes) 확률, 0~1
    input_tokens: int
    output_tokens: int
    latency_ms_client: float


class JevTypesafeError(RuntimeError):
    pass


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


class JevTypesafeClient:
    """docs.typesafe.ai/api 기준 Jev System One 어댑터 (확인일 2026-10-04).

    POST https://api.typesafe.ai/v1/systemone
    요청: {state, model, questions: {<name>: {type:"noul", instructions, criteria?}}}
    응답: {model, answers: {<name>: {type:"noul", noul: 0~1}}, usage:{input_tokens, output_tokens}}
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "jev-1.13.0",
        endpoint: str = "https://api.typesafe.ai/v1/systemone",
        timeout_sec: float = 30.0,
        max_attempts: int = 5,
        backoff_base_sec: float = 2.0,
    ):
        self._model = model
        self._endpoint = endpoint
        self._client = httpx.Client(
            timeout=timeout_sec,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        self._max_attempts = max_attempts
        self._backoff_base_sec = backoff_base_sec

    def close(self) -> None:
        self._client.close()

    def judge_noul(
        self,
        state: dict[str, Any],
        *,
        question_name: str,
        instructions: str,
        criteria: dict[str, str] | None = None,
    ) -> JevResult:
        question: dict[str, Any] = {"type": "noul", "instructions": instructions}
        if criteria is not None:
            question["criteria"] = criteria

        payload = {
            "state": state,
            "model": self._model,
            "questions": {question_name: question},
        }

        @retry(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=self._backoff_base_sec, max=60),
            reraise=True,
        )
        def _call() -> httpx.Response:
            resp = self._client.post(self._endpoint, json=payload)
            resp.raise_for_status()
            return resp

        start = time.perf_counter()
        resp = _call()
        latency_ms = (time.perf_counter() - start) * 1000

        body = resp.json()
        answer = body["answers"][question_name]
        if answer["type"] != "noul":
            raise JevTypesafeError(f"예상치 못한 answer type: {answer['type']}")

        usage = body.get("usage", {})
        return JevResult(
            noul=float(answer["noul"]),
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            latency_ms_client=latency_ms,
        )
