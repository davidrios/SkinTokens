ARG BASE_IMAGE=torch-runtime:2.11.0-cu128-py3.11
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Runtime system libs (GL/X11 libs for open3d/trimesh), plus gcc/libc6-dev,
# which Triton needs to build its launcher stubs for torch.compile (inductor),
# and curl for the compose healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gcc \
    git \
    libc6-dev \
    libegl1 \
    libfreetype6 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libx11-6 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxkbcommon0 \
    libxrender1 \
    libxxf86vm1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

# PyTorch (CUDA 12.8) and Triton come from the base image. Pin them with a
# constraints file so no dependency can swap in a different torch build.
# flash-attn is optional: without it the code falls back to PyTorch SDPA.
RUN pip list --format=freeze | grep -iE '^(torch|triton)==' > /tmp/constraints.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt -c /tmp/constraints.txt \
    && rm /tmp/requirements.txt /tmp/constraints.txt

WORKDIR /workspace/SkinTokens
COPY . .

EXPOSE 8000

CMD ["python", "serve.py"]
