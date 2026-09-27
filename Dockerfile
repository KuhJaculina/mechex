# CPU image. The release trains on CPU at the smoke scale and on a single GPU at the
# reported scale, so the recipe keeps the CUDA runtime out of the default build.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/mechex

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install -r requirements.txt

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests

RUN python -m pip install -e .

# The cohort fixture and any derived artefacts are written here, never into the image layer.
VOLUME ["/opt/mechex/data", "/opt/mechex/runs"]

ENTRYPOINT ["python", "-m"]
CMD ["mechex.console.audit", "--skip-gates", "--print-summary"]
