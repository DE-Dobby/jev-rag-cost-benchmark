from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class LabelSpan(BaseModel):
    start: int
    end: int
    text: str
    meta: str | None = None
    label_type: str
    implicit_true: bool
    due_to_null: bool


class ResponseRecord(BaseModel):
    id: str
    source_id: str
    model: str
    temperature: float
    labels: list[LabelSpan]
    split: Literal["train", "test"]
    quality: str
    response: str


class SourceInfoRecord(BaseModel):
    source_id: str
    task_type: Literal["QA", "Data2txt", "Summary"]
    source: str
    source_info: str | dict[str, Any]
    prompt: str


class MergedSample(BaseModel):
    """response.jsonl과 source_info.jsonl을 source_id로 조인한 레코드."""

    response_id: str
    source_id: str
    task_type: Literal["QA", "Data2txt", "Summary"]
    source: str
    model: str
    temperature: float
    split: Literal["train", "test"]
    quality: str
    prompt: str
    source_info: str | dict[str, Any]
    response: str
    labels: list[LabelSpan]
    hallucinated: bool  # labels.py의 매핑 규칙으로 계산된 응답 단위 라벨
