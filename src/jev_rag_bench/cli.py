import typer

app = typer.Typer(help="Jev 캐스케이드 RAG 환각 검증 비용 실험 CLI")


@app.command("prepare-data")
def prepare_data() -> None:
    """RAGTruth 로딩/전처리 (Phase 1)."""
    raise NotImplementedError("Phase 1에서 구현")


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
