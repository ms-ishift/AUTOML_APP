FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 일부 ML 패키지 런타임 의존성(libgomp) 제공
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY requirements-optional.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt && \
    pip install -r requirements-optional.txt
COPY . .

RUN useradd -m appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# init_db 는 멱등(--seed: 시스템 사용자 + 샘플 프로젝트). 빌드(RUN)가 아니라 기동 시 실행:
# - SQLite/db 경로는 런타임 볼륨·빈 디렉터리에 맞춤
# - samples/classification.csv 는 COPY 로 이미지에 포함됨(.dockerignore 에 samples 없음)
CMD ["sh", "-c", "python scripts/sync_streamlit_config.py && python scripts/init_db.py --seed && python -m streamlit run app.py --server.address=0.0.0.0 --server.port=8501"]
