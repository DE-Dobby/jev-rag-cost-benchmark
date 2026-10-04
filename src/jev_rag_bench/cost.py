from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ModelPrice:
    input_usd_per_1m_tokens: float
    output_usd_per_1m_tokens: float


def load_pricing(path: str | Path = "configs/pricing.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _price_from_dict(d: dict) -> ModelPrice:
    return ModelPrice(
        input_usd_per_1m_tokens=float(d["input_usd_per_1m_tokens"]),
        output_usd_per_1m_tokens=float(d["output_usd_per_1m_tokens"]),
    )


def jev_cost_usd(price: ModelPrice, *, input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * price.input_usd_per_1m_tokens + output_tokens * price.output_usd_per_1m_tokens) / 1_000_000


def gemini_cost_usd(price: ModelPrice, *, input_tokens: int, output_tokens: int, thinking_tokens: int) -> float:
    # thinking은 Vertex 가격표에서 "Text output (response and reasoning)"로 출력과
    # 같은 단가로 묶여 과금된다 (pricing.yaml 주석 참고). 그래서 output_tokens에 더해서 계산.
    billed_output = output_tokens + thinking_tokens
    return (
        input_tokens * price.input_usd_per_1m_tokens + billed_output * price.output_usd_per_1m_tokens
    ) / 1_000_000


def model_price(pricing: dict, provider: str, key: str) -> ModelPrice:
    return _price_from_dict(pricing[provider][key])
