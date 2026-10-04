from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ..budget import BudgetExceeded, BudgetGuard
from ..cache.local_cache import LocalCache
from ..cost import ModelPrice, model_price
from ..data.schema import MergedSample
from ..providers.jev_typesafe import JevTypesafeClient
from ..providers.vertex_gemini import VertexGeminiClient
from .runner import JudgeRecord, run_gemini, run_jev


class MissingCredentialError(RuntimeError):
    pass


def build_clients(config: dict) -> tuple[JevTypesafeClient, VertexGeminiClient]:
    jev_api_key = os.environ.get("JEV_API_KEY")
    if not jev_api_key:
        raise MissingCredentialError("JEV_API_KEY가 설정되어 있지 않습니다 (.env 확인).")
    gcp_project = os.environ.get("GCP_PROJECT_ID") or config["vertex"]["project"]
    gcp_region = os.environ.get("GCP_REGION") or config["vertex"]["region"]
    if not gcp_project:
        raise MissingCredentialError("GCP_PROJECT_ID가 설정되어 있지 않습니다 (.env 확인).")

    jev_cfg = config["jev"]
    jev_client = JevTypesafeClient(
        jev_api_key,
        model=jev_cfg["model"],
        endpoint=jev_cfg["endpoint"],
        max_attempts=config["concurrency"]["retry"]["max_attempts"],
        backoff_base_sec=config["concurrency"]["retry"]["backoff_base_sec"],
    )
    gemini_client = VertexGeminiClient(
        project=gcp_project,
        location=gcp_region,
        thinking_budget=config["vertex"]["thinking_budget"],
        max_attempts=config["concurrency"]["retry"]["max_attempts"],
        backoff_base_sec=config["concurrency"]["retry"]["backoff_base_sec"],
    )
    return jev_client, gemini_client


def build_prices(pricing: dict, config: dict) -> tuple[ModelPrice, ModelPrice, ModelPrice]:
    jev_price = model_price(pricing, "jev", config["jev"]["provider"])
    pro_price = model_price(pricing, "vertex", "gemini_pro")
    flash_price = model_price(pricing, "vertex", "gemini_flash")
    return jev_price, pro_price, flash_price


def run_batch(
    samples: list[MergedSample],
    *,
    config: dict,
    jev_client: JevTypesafeClient,
    gemini_client: VertexGeminiClient,
    cache: LocalCache,
    budget: BudgetGuard,
    jev_price: ModelPrice,
    pro_price: ModelPrice,
    flash_price: ModelPrice,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[tuple[str, MergedSample, JudgeRecord]], list[str]]:
    """samples 각각에 Jev + Gemini Pro + Gemini Flash를 호출 (캐시 적중 시 재호출 없음).

    개별 호출 실패는 errors에 모아두고 계속 진행한다. 예산 초과 시 남은 작업을 중단한다.
    """
    jev_cfg = config["jev"]
    pro_model_id = config["vertex"]["models"]["primary"]
    flash_model_id = config["vertex"]["models"]["cheap"]

    jobs: list[tuple[str, MergedSample]] = []
    for s in samples:
        jobs.append(("jev", s))
        jobs.append(("gemini_pro", s))
        jobs.append(("gemini_flash", s))

    records: list[tuple[str, MergedSample, JudgeRecord]] = []
    errors: list[str] = []
    stop_flag = {"stop": False}
    done_count = {"n": 0}

    def _do(job: tuple[str, MergedSample]):
        kind, sample = job
        if stop_flag["stop"]:
            return None
        if kind == "jev":
            rec = run_jev(
                sample,
                client=jev_client,
                cache=cache,
                price=jev_price,
                budget=budget,
                model_id=jev_cfg["model"],
                question_variant="v1",
                max_input_tokens=jev_cfg["max_input_tokens"],
                overflow_policy=jev_cfg["overflow_policy"],
            )
        elif kind == "gemini_pro":
            rec = run_gemini(sample, client=gemini_client, cache=cache, price=pro_price, budget=budget, model_id=pro_model_id)
        else:
            rec = run_gemini(sample, client=gemini_client, cache=cache, price=flash_price, budget=budget, model_id=flash_model_id)
        return kind, sample, rec

    with ThreadPoolExecutor(max_workers=config["concurrency"]["max_parallel"]) as pool:
        futures = [pool.submit(_do, job) for job in jobs]
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except BudgetExceeded as e:
                stop_flag["stop"] = True
                errors.append(str(e))
                continue
            except Exception as e:  # noqa: BLE001 — 개별 실패는 기록하고 계속
                errors.append(repr(e))
                continue
            done_count["n"] += 1
            if on_progress is not None:
                on_progress(done_count["n"], len(jobs))
            if result is not None:
                records.append(result)

    return records, errors


def to_combined_rows(records: list[tuple[str, MergedSample, JudgeRecord]]) -> list[dict[str, Any]]:
    """(kind, sample, record) 리스트를 sample별 한 행으로 합친다 (Phase 4 입력용)."""
    by_sample: dict[str, dict[str, Any]] = {}
    for kind, sample, rec in records:
        row = by_sample.setdefault(
            sample.response_id,
            {
                "response_id": sample.response_id,
                "task_type": sample.task_type,
                "split": sample.split,
                "hallucinated_gt": sample.hallucinated,
            },
        )
        prefix = {"jev": "jev", "gemini_pro": "pro", "gemini_flash": "flash"}[kind]
        row[f"{prefix}_hallucinated"] = rec.hallucinated
        row[f"{prefix}_cost_usd"] = rec.cost_usd
        row[f"{prefix}_latency_ms"] = rec.latency_ms
        if rec.noul is not None:
            row[f"{prefix}_noul"] = rec.noul
    return list(by_sample.values())
