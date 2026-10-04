import json
from pathlib import Path

import typer

from .config import load_config
from .data.download import download_ragtruth
from .data.loader import load_merged_samples
from .data.stats import build_stats, format_stats

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

    n_responses = sum(1 for _ in open(raw_dir / "response.jsonl", encoding="utf-8"))
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


@app.command("pilot")
def pilot() -> None:
    """층화 샘플 파일럿 실행 (Phase 2)."""
    raise NotImplementedError("Phase 2에서 구현")


@app.command("run")
def run() -> None:
    """본 실행 — 캐시 적중 시 재호출 없음, 중단 후 재개 가능 (Phase 3)."""
    raise NotImplementedError("Phase 3에서 구현")


@app.command("sweep")
def sweep() -> None:
    """임계값 스윕 + 지표 계산 + 그래프 (Phase 4)."""
    raise NotImplementedError("Phase 4에서 구현")


@app.command("report")
def report() -> None:
    """README/블로그용 결과 요약 생성 (Phase 5)."""
    raise NotImplementedError("Phase 5에서 구현")


if __name__ == "__main__":
    app()
