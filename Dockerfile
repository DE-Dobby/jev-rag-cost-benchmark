FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs

RUN uv pip install --system --no-cache .

# 로컬: docker run -v $(pwd)/results:/app/results --env-file .env <image> <subcommand>
# Cloud Run Job: 환경변수로 CACHE_BACKEND=gcs, GCS_CACHE_BUCKET=... 주입, ADC는 서비스 계정으로 자동 적용
ENTRYPOINT ["jev-rag-bench"]
