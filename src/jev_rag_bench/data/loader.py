from __future__ import annotations

import json
from pathlib import Path

from .labels import is_hallucinated_response
from .schema import MergedSample, ResponseRecord, SourceInfoRecord


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_merged_samples(
    raw_dir: Path,
    *,
    implicit_true_as_hallucination: bool,
    due_to_null_as_hallucination: bool,
) -> list[MergedSample]:
    """response.jsonl + source_info.jsonl을 source_id로 조인해 MergedSample 리스트를 만든다."""
    responses = [ResponseRecord.model_validate(d) for d in _read_jsonl(raw_dir / "response.jsonl")]
    source_infos = {
        d["source_id"]: SourceInfoRecord.model_validate(d) for d in _read_jsonl(raw_dir / "source_info.jsonl")
    }

    samples: list[MergedSample] = []
    for r in responses:
        src = source_infos.get(r.source_id)
        if src is None:
            # source_info.jsonl에 대응하는 source_id가 없는 경우 — 발생하면 세어서 보고
            continue
        hallucinated = is_hallucinated_response(
            r.labels,
            implicit_true_as_hallucination=implicit_true_as_hallucination,
            due_to_null_as_hallucination=due_to_null_as_hallucination,
        )
        samples.append(
            MergedSample(
                response_id=r.id,
                source_id=r.source_id,
                task_type=src.task_type,
                source=src.source,
                model=r.model,
                temperature=r.temperature,
                split=r.split,
                quality=r.quality,
                prompt=src.prompt,
                source_info=src.source_info,
                response=r.response,
                labels=r.labels,
                hallucinated=hallucinated,
            )
        )
    return samples


def load_processed_samples(path: Path) -> list[MergedSample]:
    """prepare-data가 저장한 data/processed/ragtruth_merged.jsonl을 다시 불러온다."""
    return [MergedSample.model_validate(d) for d in _read_jsonl(path)]


def grounding_text(sample: MergedSample) -> str:
    """판정 모델에 넘길 "근거 문서" 텍스트. Summary는 원문 그대로, QA/Data2txt는
    source_info 딕셔너리를 JSON으로 직렬화 (지시문 prompt 전체가 아니라 근거 자료만)."""
    if isinstance(sample.source_info, str):
        return sample.source_info
    return json.dumps(sample.source_info, ensure_ascii=False)
