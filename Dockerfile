ARG BASE_IMAGE=torch-runtime:2.11.0-cu128-py3.11
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# gcc/libc6-dev: Triton needs them to build launcher stubs for torch.compile
# (inductor). curl: compose healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gcc \
    libc6-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

# PyTorch (CUDA 12.8) and Triton come from the base image. Pin them with a
# constraints file so no dependency can swap in a different torch build.
# flash-attn is optional: without it the code falls back to PyTorch SDPA.
RUN pip list --format=freeze | grep -iE '^(torch|triton)==' > /tmp/constraints.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt -c /tmp/constraints.txt \
    && rm /tmp/requirements.txt /tmp/constraints.txt

WORKDIR /workspace/SkinTokens
# Model weights are baked in so the image is self-contained. They sit before the
# code layers so code edits do not invalidate (or re-push) these large layers.
# Layout matches src/paths.py defaults (MODEL_ROOT = project root).
COPY experiments ./experiments
COPY models ./models

# Only what the server needs at runtime (see .dockerignore allowlist).
COPY serve.py ./
COPY configs ./configs
COPY src ./src

EXPOSE 8000

CMD ["python", "serve.py"]
