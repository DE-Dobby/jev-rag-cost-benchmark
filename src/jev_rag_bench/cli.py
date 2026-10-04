import json
import statistics
import time
from pathlib import Path

import typer
from dotenv import load_dotenv

from .budget import BudgetGuard
from .cache.local_cache import LocalCache
from .config import load_config
from .cost import load_pricing
from .data.download import download_ragtruth
from .data.loader import load_merged_samples, load_processed_samples
from .data.sampling import simple_random_sample, stratified_sample_by_task
from .data.stats import build_stats, format_stats
from .data.tokens import approx_token_count
from .judges.orchestrate import (
    MissingCredentialError,
    build_clients,
    build_prices,
    run_batch,
    to_combined_rows,
)
from .metrics.calibration import expected_calibration_error, reliability_bins
from .metrics.report import (
    breakeven_table,
    cost_by_length_bins,
    evaluate_cascade_at_threshold,
    evaluate_simple_config,
    per_task_breakdown,
    per_task_cascade_breakdown,
    pick_best_threshold,
    sweep_thresholds_on_dev,
)
from .plots.calibration_plot import plot_reliability_diagram
from .plots.cost_bins import plot_cost_by_length_bins
from .plots.pareto import plot_cost_vs_f1

app = typer.Typer(help="Jev 캐스케이드 RAG 환각 검증 비용 실험 CLI")


@app.command("prepare-data")
def prepare_data(
    config_path: str = typer.Option("configs/config.yaml", "--config"),
) -> None:
    """RAGTruth 다운로드 + source_id 조인 + 라벨 매핑 + 통계 출력."""
    config = load_config(config_path)
    raw_dir = Path(config["dataset"]["raw_dir"])
    label_mapping = config["dataset"]["label_mapping"]

    typer.echo(f"RAGTruth 원본을 {raw_dir}에 다운로드 (이미 있으면 건너뜀)...")
    download_ragtruth(raw_dir)

    samples = load_merged_samples(
        raw_dir,
        implicit_true_as_hallucination=label_mapping["implicit_true_as_hallucination"],
        due_to_null_as_hallucination=label_mapping["due_to_null_as_hallucination"],
    )

    with open(raw_dir / "response.jsonl", encoding="utf-8") as f:
        n_responses = sum(1 for _ in f)
    if len(samples) != n_responses:
        typer.echo(
            f"[경고] response.jsonl {n_responses}건 중 {n_responses - len(samples)}건이 "
            "source_info.jsonl과 조인되지 않아 제외되었습니다."
        )

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "ragtruth_merged.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.model_dump(mode="json"), ensure_ascii=False) + "\n")
    typer.echo(f"전처리된 {len(samples)}건을 {out_path}에 저장했습니다.\n")

    stats = build_stats(samples)
    typer.echo(format_stats(stats))


def _load_processed_or_exit() -> list:
    processed_path = Path("data/processed/ragtruth_merged.jsonl")
    if not processed_path.exists():
        typer.echo("data/processed/ragtruth_merged.jsonl이 없습니다. 먼저 `prepare-data`를 실행하세요.")
        raise typer.Exit(code=1)
    return load_processed_samples(processed_path)


def _print_cost_latency_summary(records, label: str) -> float:
    by_kind: dict[str, list] = {}
    for kind, _sample, rec in records:
        by_kind.setdefault(kind, []).append(rec)

    total_cost = sum(rec.cost_usd for _, _, rec in records)
    typer.echo(f"\n[{label}] 총 ${total_cost:.6f} (호출 {len(records)}건)")
    for kind, recs in by_kind.items():
        costs = [r.cost_usd for r in recs]
        latencies = sorted(r.latency_ms for r in recs)
        p95_idx = max(0, int(len(latencies) * 0.95) - 1)
        typer.echo(
            f"  {kind}: n={len(recs)}, 비용합=${sum(costs):.6f}, "
            f"평균지연={statistics.mean(latencies):.0f}ms, p95지연={latencies[p95_idx]:.0f}ms"
        )
    return total_cost


@app.command("pilot")
def pilot(
    config_path: str = typer.Option("configs/config.yaml", "--config"),
    n_per_task: int = typer.Option(10, "--n-per-task", help="task_type별 샘플 수 (기본 10 x 3 = 30건)"),
    max_usd: float | None = typer.Option(None, "--max-usd", help="예산 상한 USD (미지정 시 config.budget.max_usd)"),
    dev_test_total: int = typer.Option(1000, "--dev-test-total", help="본 실행 예상 건수 (기본 dev+test 500+500)"),
) -> None:
    """층화 샘플 파일럿 실행: Jev + Gemini Pro/Flash 실제 호출, 비용/지연시간 측정 + 본 실행 비용 추정."""
    load_dotenv()
    config = load_config(config_path)
    pricing = load_pricing()
    samples = _load_processed_or_exit()

    pilot_samples = stratified_sample_by_task(samples, n_per_task=n_per_task, seed=config["seed"])
    typer.echo(f"파일럿 샘플 {len(pilot_samples)}건 (task당 {n_per_task}건)")

    try:
        jev_client, gemini_client = build_clients(config)
    except MissingCredentialError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1) from e

    budget_limit = max_usd if max_usd is not None else float(config["budget"]["max_usd"])
    budget = BudgetGuard(budget_limit)
    cache = LocalCache(config["cache"]["local_dir"])
    jev_price, pro_price, flash_price = build_prices(pricing, config)

    records, errors = run_batch(
        pilot_samples,
        config=config,
        jev_client=jev_client,
        gemini_client=gemini_client,
        cache=cache,
        budget=budget,
        jev_price=jev_price,
        pro_price=pro_price,
        flash_price=flash_price,
    )
    jev_client.close()

    typer.echo(f"\n완료: {len(records)}/{len(pilot_samples) * 3} 호출 성공, 오류 {len(errors)}건")
    for e in errors[:10]:
        typer.echo(f"  - {e}")

    total_cost = _print_cost_latency_summary(records, "파일럿 실측 비용")

    if pilot_samples:
        cost_per_sample = total_cost / len(pilot_samples)
        extrapolated = cost_per_sample * dev_test_total
        typer.echo(
            f"\n[본 실행 비용 추정] 샘플당 평균 ${cost_per_sample:.6f} x {dev_test_total}건 "
            f"≈ ${extrapolated:.4f} (Jev+Gemini Pro+Gemini Flash 전량 호출 기준, 캐스케이드 B는 오프라인 재사용이라 추가 비용 없음)"
        )


@app.command("run")
def run(
    config_path: str = typer.Option("configs/config.yaml", "--config"),
    max_usd: float | None = typer.Option(None, "--max-usd", help="예산 상한 USD (미지정 시 config.budget.max_usd)"),
    dev_size: int | None = typer.Option(None, "--dev-size", help="미지정 시 config.dataset.split.dev_size"),
    test_size: int | None = typer.Option(None, "--test-size", help="미지정 시 config.dataset.split.test_size"),
) -> None:
    """본 실행: config의 dev_size/test_size만큼 RAGTruth train/test에서 뽑아 Jev+Gemini Pro/Flash 호출.

    캐시 적중 시 재호출하지 않으므로 중단 후 같은 명령으로 재실행하면 이어서 진행된다.
    """
    load_dotenv()
    config = load_config(config_path)
    pricing = load_pricing()
    samples = _load_processed_or_exit()

    split_cfg = config["dataset"]["split"]
    seed = config["seed"]
    train_pool = [s for s in samples if s.split == "train"]
    test_pool = [s for s in samples if s.split == "test"]
    dev_samples = simple_random_sample(train_pool, n=dev_size or split_cfg["dev_size"], seed=seed)
    test_samples = simple_random_sample(test_pool, n=test_size or split_cfg["test_size"], seed=seed)
    typer.echo(f"dev(train split에서) {len(dev_samples)}건, test(test split에서) {len(test_samples)}건")

    try:
        jev_client, gemini_client = build_clients(config)
    except MissingCredentialError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1) from e

    budget_limit = max_usd if max_usd is not None else float(config["budget"]["max_usd"])
    budget = BudgetGuard(budget_limit)
    cache = LocalCache(config["cache"]["local_dir"])
    jev_price, pro_price, flash_price = build_prices(pricing, config)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    for name, split_samples in [("dev", dev_samples), ("test", test_samples)]:
        typer.echo(f"\n=== {name} ({len(split_samples)}건) 실행 ===")
        start = time.monotonic()
        last_report = {"n": 0}

        def _progress(done: int, total: int) -> None:
            if done - last_report["n"] >= 100 or done == total:
                last_report["n"] = done
                elapsed = time.monotonic() - start
                typer.echo(f"  {done}/{total} 완료 ({elapsed:.0f}s 경과, 누적비용 ${budget.spent_usd:.4f})")

        records, errors = run_batch(
            split_samples,
            config=config,
            jev_client=jev_client,
            gemini_client=gemini_client,
            cache=cache,
            budget=budget,
            jev_price=jev_price,
            pro_price=pro_price,
            flash_price=flash_price,
            on_progress=_progress,
        )

        typer.echo(f"{name}: 완료 {len(records)}/{len(split_samples) * 3} 호출, 오류 {len(errors)}건")
        for e in errors[:10]:
            typer.echo(f"  - {e}")
        _print_cost_latency_summary(records, f"{name} 실측 비용")

        rows = to_combined_rows(records)
        out_path = results_dir / f"{name}_results.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        typer.echo(f"{name}: {len(rows)}건을 {out_path}에 저장")

    jev_client.close()
    typer.echo(f"\n총 누적 비용: ${budget.spent_usd:.4f}")


def _load_results_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _fmt_ci(value: float, ci: tuple[float, float]) -> str:
    return f"{value:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"


@app.command("sweep")
def sweep(
    config_path: str = typer.Option("configs/config.yaml", "--config"),
    n_bootstrap: int = typer.Option(2000, "--n-bootstrap"),
) -> None:
    """dev에서 캐스케이드 임계값 선택 → test에 고정 적용해 한 번만 평가. 지표/그래프/요약 저장."""
    config = load_config(config_path)
    dev_path = Path("results/dev_results.jsonl")
    test_path = Path("results/test_results.jsonl")
    if not dev_path.exists() or not test_path.exists():
        typer.echo("results/dev_results.jsonl / test_results.jsonl이 없습니다. 먼저 `run`을 실행하세요.")
        raise typer.Exit(code=1)
    dev_rows = _load_results_jsonl(dev_path)
    test_rows = _load_results_jsonl(test_path)

    # 1. dev에서 임계값 스윕 (test는 아직 건드리지 않음)
    thresholds = config["cascade"]["thresholds"]
    dev_sweep = sweep_thresholds_on_dev(dev_rows, thresholds, fallback="pro")
    chosen_threshold = pick_best_threshold(dev_sweep)
    typer.echo(f"dev 스윕(Jev+Pro)으로 고른 임계값: {chosen_threshold:.2f}")
    for r in dev_sweep:
        typer.echo(
            f"  t={r['threshold']:.2f}  f1={r['f1']:.4f}  acc={r['accuracy']:.4f}  "
            f"cost/sample=${r['cost_per_sample']:.6f}  coverage={r['coverage']:.3f}"
        )

    # Jev+Flash 캐스케이드도 같은 방식으로 별도 임계값 선택 (Jev+Pro와 비교용)
    dev_sweep_flash = sweep_thresholds_on_dev(dev_rows, thresholds, fallback="flash")
    chosen_threshold_flash = pick_best_threshold(dev_sweep_flash)
    typer.echo(f"\ndev 스윕(Jev+Flash)으로 고른 임계값: {chosen_threshold_flash:.2f}")

    # 2. test에 고정 적용해서 A/J/C/B 한 번만 평가 (부트스트랩 95% CI)
    result_a = evaluate_simple_config(
        test_rows, name="A_gemini_pro", pred_key="pro_hallucinated", cost_key="pro_cost_usd",
        latency_key="pro_latency_ms", n_bootstrap=n_bootstrap, seed=config["seed"],
    )
    result_j = evaluate_simple_config(
        test_rows, name="J_jev", pred_key="jev_hallucinated", cost_key="jev_cost_usd",
        latency_key="jev_latency_ms", n_bootstrap=n_bootstrap, seed=config["seed"],
    )
    result_c = evaluate_simple_config(
        test_rows, name="C_gemini_flash", pred_key="flash_hallucinated", cost_key="flash_cost_usd",
        latency_key="flash_latency_ms", n_bootstrap=n_bootstrap, seed=config["seed"],
    )
    result_b = evaluate_cascade_at_threshold(
        test_rows, threshold=chosen_threshold, fallback="pro", n_bootstrap=n_bootstrap, seed=config["seed"],
        name="B_cascade_jev_pro",
    )
    result_b2 = evaluate_cascade_at_threshold(
        test_rows, threshold=chosen_threshold_flash, fallback="flash", n_bootstrap=n_bootstrap, seed=config["seed"],
        name="B2_cascade_jev_flash",
    )

    typer.echo("\n[test 최종 평가 — 각 1회만 평가, 부트스트랩 95% CI]")
    for r in [result_a, result_j, result_c, result_b, result_b2]:
        extra = f", coverage={r.coverage:.3f}" if r.coverage is not None else ""
        typer.echo(
            f"  {r.name}: acc={_fmt_ci(r.accuracy, r.accuracy_ci)}  f1={_fmt_ci(r.f1, r.f1_ci)}  "
            f"precision={r.precision:.4f}  recall={r.recall:.4f}  "
            f"cost/sample=${r.cost_per_sample:.6f}  지연(평균/p95)={r.latency_mean_ms:.0f}/{r.latency_p95_ms:.0f}ms{extra}"
        )

    # 3. 태스크별 분해 (test)
    task_breakdown = {
        "A_gemini_pro": per_task_breakdown(test_rows, pred_key="pro_hallucinated"),
        "J_jev": per_task_breakdown(test_rows, pred_key="jev_hallucinated"),
        "C_gemini_flash": per_task_breakdown(test_rows, pred_key="flash_hallucinated"),
        "B_cascade_jev_pro": per_task_cascade_breakdown(test_rows, threshold=chosen_threshold, fallback="pro"),
        "B2_cascade_jev_flash": per_task_cascade_breakdown(
            test_rows, threshold=chosen_threshold_flash, fallback="flash"
        ),
    }
    typer.echo("\n[태스크별 분해 (test)]")
    for name, breakdown in task_breakdown.items():
        typer.echo(f"  {name}:")
        for task, m in breakdown.items():
            typer.echo(f"    {task}: n={m['n']}, acc={m['accuracy']:.4f}, f1={m['f1']:.4f}")

    # 4. Jev 캘리브레이션 (dev+test 합쳐서, 서술 통계라 평가 무결성과 무관)
    combined = dev_rows + test_rows
    probs = [r["jev_noul"] for r in combined]
    labels = [r["hallucinated_gt"] for r in combined]
    ece = expected_calibration_error(probs, labels, n_bins=10)
    bins = reliability_bins(probs, labels, n_bins=10)
    typer.echo(f"\n[Jev 캘리브레이션] ECE={ece:.4f} (dev+test 합쳐 n={len(combined)})")

    # 5. 근거 문서 토큰 길이 구간별 비용
    samples = load_processed_samples(Path("data/processed/ragtruth_merged.jsonl"))
    from .data.loader import grounding_text

    token_lengths = {s.response_id: approx_token_count(grounding_text(s)) for s in samples}
    length_bins = cost_by_length_bins(combined, token_lengths=token_lengths, threshold=chosen_threshold, n_bins=4)
    typer.echo("\n[근거 문서 토큰 길이 구간별 비용 (A vs B)]")
    for b in length_bins:
        typer.echo(
            f"  토큰 {b['token_range'][0]}~{b['token_range'][1]} (n={b['n']}): "
            f"A=${b['a_cost_per_sample']:.6f}  B=${b['b_cost_per_sample']:.6f}"
        )

    # 6. 손익분기 (c_jev < coverage * c_llm)
    jev_cost_mean = statistics.mean(r["jev_cost_usd"] for r in combined)
    pro_cost_mean = statistics.mean(r["pro_cost_usd"] for r in combined)
    breakeven = breakeven_table(dev_sweep, jev_cost_per_sample=jev_cost_mean, pro_cost_per_sample=pro_cost_mean)
    typer.echo("\n[손익분기: c_jev < coverage * c_llm]")
    for b in breakeven:
        typer.echo(f"  t={b['threshold']:.2f}  coverage={b['coverage']:.3f}  성립={b['breakeven_holds']}")

    # 7. 저장: results/summary/
    summary_dir = Path("results/summary")
    summary_dir.mkdir(parents=True, exist_ok=True)

    metrics_json = {
        "chosen_threshold_jev_pro": chosen_threshold,
        "chosen_threshold_jev_flash": chosen_threshold_flash,
        "dev_sweep_jev_pro": dev_sweep,
        "dev_sweep_jev_flash": dev_sweep_flash,
        "test_results": {
            r.name: {
                "accuracy": r.accuracy, "accuracy_ci": r.accuracy_ci,
                "f1": r.f1, "f1_ci": r.f1_ci,
                "precision": r.precision, "recall": r.recall,
                "cost_per_sample": r.cost_per_sample,
                "latency_mean_ms": r.latency_mean_ms, "latency_p95_ms": r.latency_p95_ms,
                "n": r.n, "coverage": r.coverage,
            }
            for r in [result_a, result_j, result_c, result_b, result_b2]
        },
        "task_breakdown": task_breakdown,
        "calibration": {"ece": ece, "n": len(combined)},
        "cost_by_length_bins": length_bins,
        "breakeven": breakeven,
    }
    (summary_dir / "metrics.json").write_text(json.dumps(metrics_json, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 결과 요약\n",
        f"선택된 캐스케이드 임계값 (dev 기준) — Jev+Pro: **{chosen_threshold:.2f}**, "
        f"Jev+Flash: **{chosen_threshold_flash:.2f}**\n",
        "| 구성 | Accuracy | F1 | Precision | Recall | 비용/건 | 지연(평균) | 지연(p95) | Coverage |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in [result_a, result_j, result_c, result_b, result_b2]:
        cov = f"{r.coverage:.3f}" if r.coverage is not None else "—"
        lines.append(
            f"| {r.name} | {_fmt_ci(r.accuracy, r.accuracy_ci)} | {_fmt_ci(r.f1, r.f1_ci)} | "
            f"{r.precision:.4f} | {r.recall:.4f} | ${r.cost_per_sample:.6f} | "
            f"{r.latency_mean_ms:.0f}ms | {r.latency_p95_ms:.0f}ms | {cov} |"
        )
    (summary_dir / "metrics_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 8. 그래프
    point_a = {"cost_per_sample": result_a.cost_per_sample, "f1": result_a.f1}
    point_j = {"cost_per_sample": result_j.cost_per_sample, "f1": result_j.f1}
    point_c = {"cost_per_sample": result_c.cost_per_sample, "f1": result_c.f1}
    point_b_test = {"cost_per_sample": result_b.cost_per_sample, "f1": result_b.f1, "threshold": chosen_threshold}
    point_b2_test = {
        "cost_per_sample": result_b2.cost_per_sample, "f1": result_b2.f1, "threshold": chosen_threshold_flash,
    }
    plot_cost_vs_f1(
        dev_sweep=dev_sweep, point_a=point_a, point_j=point_j, point_c=point_c,
        point_b_test=point_b_test, point_b2_test=point_b2_test, out_path=summary_dir / "pareto.png",
    )
    plot_reliability_diagram(bins, ece=ece, out_path=summary_dir / "calibration.png")
    plot_cost_by_length_bins(length_bins, out_path=summary_dir / "cost_by_length.png")

    typer.echo(
        f"\n요약 저장: {summary_dir}/metrics.json, metrics_table.md, pareto.png, calibration.png, cost_by_length.png"
    )


@app.command("report")
def report() -> None:
    """README/블로그용 결과 요약 생성 (Phase 5)."""
    raise NotImplementedError("Phase 5에서 구현")


if __name__ == "__main__":
    app()
