from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from ..budget import BudgetGuard
from ..cache.local_cache import LocalCache
from ..cost import ModelPrice, gemini_cost_usd, jev_cost_usd
from ..data.loader import grounding_text
from ..data.schema import MergedSample
from ..data.tokens import approx_token_count
from ..providers.jev_typesafe import JevTypesafeClient
from ..providers.vertex_gemini import VertexGeminiClient
from .prompts import JEV_NOUL_CRITERIA, JEV_NOUL_QUESTIONS, build_gemini_prompt

GEMINI_PROMPT_VERSION = "v1"
JEV_QUESTION_NAME = "hallucination"


@dataclass
class JudgeRecord:
    provider: str
    model: str
    prompt_version: str
    sample_id: str
    cost_usd: float
    latency_ms: float
    cache_hit: bool
    hallucinated: bool  # 0.5 임계값 기준 이진 판정 (noul의 경우 noul>=0.5)
    noul: float | None  # Jev만 채움
    input_tokens: int
    output_tokens: int
    overflow_truncated: bool = False


def _truncate_state_for_jev(state: dict[str, Any], instructions: str, max_input_tokens: int) -> tuple[dict, bool]:
    def token_len(s: dict) -> int:
        return approx_token_count(json.dumps(s, ensure_ascii=False) + instructions)

    if token_len(state) <= max_input_tokens:
        return state, False

    context = state["context"]
    truncated = dict(state)
    while token_len(truncated) > max_input_tokens and len(context) > 0:
        context = context[: int(len(context) * 0.9)]
        truncated = {**state, "context": context}
    return truncated, True


def run_jev(
    sample: MergedSample,
    *,
    client: JevTypesafeClient,
    cache: LocalCache,
    price: ModelPrice,
    budget: BudgetGuard,
    model_id: str,
    question_variant: str = "v1",
    max_input_tokens: int,
    overflow_policy: Literal["truncate", "exclude"] = "truncate",
) -> JudgeRecord | None:
    """반환값이 None이면 overflow_policy=exclude로 제외된 샘플."""
    prompt_version = f"jev_{question_variant}"
    cached = cache.get("jev_typesafe", model_id, prompt_version, sample.response_id)
    if cached is not None:
        return JudgeRecord(**{**cached, "cache_hit": True})

    instructions = JEV_NOUL_QUESTIONS[question_variant]
    state = {
        "task_type": sample.task_type,
        "context": grounding_text(sample),
        "response": sample.response,
    }

    overflow_truncated = False
    full_len = approx_token_count(json.dumps(state, ensure_ascii=False) + instructions)
    if full_len > max_input_tokens:
        if overflow_policy == "exclude":
            return None
        state, overflow_truncated = _truncate_state_for_jev(state, instructions, max_input_tokens)

    result = client.judge_noul(
        state,
        question_name=JEV_QUESTION_NAME,
        instructions=instructions,
        criteria=JEV_NOUL_CRITERIA,
    )
    cost = jev_cost_usd(price, input_tokens=result.input_tokens, output_tokens=result.output_tokens)
    budget.add(cost)

    record = JudgeRecord(
        provider="jev_typesafe",
        model=model_id,
        prompt_version=prompt_version,
        sample_id=sample.response_id,
        cost_usd=cost,
        latency_ms=result.latency_ms_client,
        cache_hit=False,
        hallucinated=result.noul >= 0.5,
        noul=result.noul,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        overflow_truncated=overflow_truncated,
    )
    cache.put("jev_typesafe", model_id, prompt_version, sample.response_id, record.__dict__)
    return record


def run_gemini(
    sample: MergedSample,
    *,
    client: VertexGeminiClient,
    cache: LocalCache,
    price: ModelPrice,
    budget: BudgetGuard,
    model_id: str,
) -> JudgeRecord:
    cached = cache.get("vertex", model_id, GEMINI_PROMPT_VERSION, sample.response_id)
    if cached is not None:
        return JudgeRecord(**{**cached, "cache_hit": True})

    prompt = build_gemini_prompt(task_type=sample.task_type, context=grounding_text(sample), response=sample.response)
    result = client.judge_hallucination(prompt, model_id=model_id)
    cost = gemini_cost_usd(
        price,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        thinking_tokens=result.thinking_tokens,
    )
    budget.add(cost)

    record = JudgeRecord(
        provider="vertex",
        model=model_id,
        prompt_version=GEMINI_PROMPT_VERSION,
        sample_id=sample.response_id,
        cost_usd=cost,
        latency_ms=result.latency_ms_client,
        cache_hit=False,
        hallucinated=result.hallucinated,
        noul=None,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )
    cache.put("vertex", model_id, GEMINI_PROMPT_VERSION, sample.response_id, record.__dict__)
    return record
