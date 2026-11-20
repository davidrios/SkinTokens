ARG BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

# ---------------------------------------------------------------------------
# Model weights: fetched from Hugging Face in a separate stage so the build
# needs nothing on the host (no `python download.py --model` first). Docker
# caches this stage, so code edits never re-download the weights.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS weights

ARG HF_REPO=VAST-AI/SkinTokens
ARG LLM_REPO=Qwen/Qwen3-0.6B

RUN pip install --no-cache-dir huggingface_hub==2.4.0

WORKDIR /weights
RUN python - <<EOF
from huggingface_hub import hf_hub_download, snapshot_download

for name in (
    "experiments/skin_vae_2_10_32768/last.ckpt",
    "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
):
    hf_hub_download(repo_id="${HF_REPO}", filename=name, local_dir=".")

# Tokenizer/config only; the LLM weights come from the checkpoints above.
snapshot_download(
    repo_id="${LLM_REPO}",
    local_dir="models/Qwen3-0.6B",
    allow_patterns=["*.json", "merges.txt", "LICENSE"],
)
EOF
RUN rm -rf .cache models/Qwen3-0.6B/.cache

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE}

# PIP_BREAK_SYSTEM_PACKAGES: the base image's Python is the distro's (PEP 668);
# this container is single-purpose, so install into it directly.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

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
# Model weights sit before the code layers so code edits do not invalidate (or
# re-push) these large layers. Layout matches src/paths.py defaults
# (MODEL_ROOT = project root).
COPY --from=weights /weights/experiments ./experiments
COPY --from=weights /weights/models ./models

# Only what the server needs at runtime (see .dockerignore allowlist).
COPY serve.py ./
COPY configs ./configs
COPY src ./src

EXPOSE 8000

CMD ["python", "-m", "src.server.log_archive", "python", "serve.py"]
