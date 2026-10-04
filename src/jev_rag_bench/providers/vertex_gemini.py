from __future__ import annotations

import json
import time
from dataclasses import dataclass

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"hallucinated": {"type": "boolean"}},
    "required": ["hallucinated"],
}


@dataclass
class GeminiResult:
    hallucinated: bool
    input_tokens: int
    output_tokens: int  # candidates_token_count (thinking 제외)
    thinking_tokens: int
    latency_ms_client: float


def _is_retryable(exc: BaseException) -> bool:
    # 429(RESOURCE_EXHAUSTED) 및 5xx만 재시도. google-genai는 이를
    # ClientError(429)/ServerError(5xx)로 노출한다.
    if isinstance(exc, ClientError):
        return exc.code == 429
    return isinstance(exc, ServerError)


class VertexGeminiClient:
    """google-genai SDK 기준 Vertex AI Gemini 어댑터 (확인일 2026-10-04).

    client = genai.Client(vertexai=True, project=..., location=...) — ADC 인증 자동 사용.
    thinking_budget=0으로 고정, response_mime_type=application/json + response_schema로
    {"hallucinated": bool}만 강제.
    """

    def __init__(
        self,
        *,
        project: str,
        location: str,
        thinking_budget: int = 0,
        timeout_sec: float = 60.0,
        max_attempts: int = 5,
        backoff_base_sec: float = 2.0,
    ):
        self._client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=types.HttpOptions(timeout=int(timeout_sec * 1000)),
        )
        self._thinking_budget = thinking_budget
        self._max_attempts = max_attempts
        self._backoff_base_sec = backoff_base_sec

    def judge_hallucination(self, prompt: str, *, model_id: str) -> GeminiResult:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            thinking_config=types.ThinkingConfig(thinking_budget=self._thinking_budget),
        )

        @retry(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=self._backoff_base_sec, max=60),
            reraise=True,
        )
        def _call():
            return self._client.models.generate_content(model=model_id, contents=prompt, config=config)

        start = time.perf_counter()
        response = _call()
        latency_ms = (time.perf_counter() - start) * 1000

        parsed = json.loads(response.text)
        usage = response.usage_metadata

        return GeminiResult(
            hallucinated=bool(parsed["hallucinated"]),
            input_tokens=int(usage.prompt_token_count or 0),
            output_tokens=int(usage.candidates_token_count or 0),
            thinking_tokens=int(usage.thoughts_token_count or 0),
            latency_ms_client=latency_ms,
        )
