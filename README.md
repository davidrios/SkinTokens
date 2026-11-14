<h1 align="center">SkinTokens: A Learned Compact Representation <br/> for Unified Autoregressive Rigging</h1>

<p align="center">
<a href="https://arxiv.org/abs/2602.04805"><img src='https://img.shields.io/badge/arXiv-2602.04805-b31b1b.svg?logo=arxiv' alt='arXiv'></a>
<a href="https://zjp-shadow.github.io/works/SkinTokens/"><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
<a href="https://huggingface.co/VAST-AI/SkinTokens"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue' alt='HuggingFace Model'></a>
<a href="https://huggingface.co/spaces/VAST-AI/SkinTokens"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-yellow' alt='HuggingFace Demo'></a>
<a href="https://www.tripo3d.ai"><img src='https://img.shields.io/badge/Tripo-3D_Studio-ff7a00' alt='Tripo'></a>
</p>

<p align="center">
  <img src="assets/skintokens-teaser.png" width="100%" alt="TokenRig teaser: automated rigging with SkinTokens"/>
</p>

**SkinTokens** is a learned, compact, and discrete representation for skinning weights. Built on this representation, **TokenRig** is a unified autoregressive framework that models the entire rig, i.e., skeleton and skinning weights, as a single token sequence. Given an input 3D mesh, it generates a complete skeleton hierarchy and skin weights that can be directly imported into standard 3D pipelines for character animation and simulation.

SkinTokens is the successor to [UniRig](https://github.com/VAST-AI-Research/UniRig) (SIGGRAPH '25). While UniRig uses separate stages for skeleton prediction and skinning, SkinTokens unifies both into a single autoregressive sequence via learned discrete skin tokens, yielding **98%–133%** improvement in skinning accuracy and **17%–22%** improvement in bone prediction over state-of-the-art baselines.

## 🔮 Overview

TokenRig takes a **single 3D mesh** as input and autoregressively produces a fully rigged asset — a coherent skeleton hierarchy plus dense per-vertex skinning weights — in a single unified sequence. Method in three stages:

1. **Learn SkinTokens** — An FSQ-CVAE compresses sparse skinning weights into a compact discrete vocabulary.
2. **Unified Autoregressive Modeling** — A Qwen3-0.6B-based Transformer generates the full rig (skeleton followed by SkinTokens) as one interleaved sequence.
3. **RL Refinement via GRPO** — Tailored geometric and semantic rewards (volumetric joint coverage, bone-mesh containment, skinning sparsity, deformation smoothness) refine the model for out-of-distribution assets.

<p align="center">
  <img src="https://zjp-shadow.github.io/works/SkinTokens/static/images/pipeline.png" width="100%" alt="TokenRig pipeline overview"/>
</p>

<p align="center">
  <img src="assets/compare-skin.jpg" width="100%" alt="Qualitative comparison of skinning prediction: TokenRig vs. Puppeteer vs. UniRig, with average L1 error maps"/>
</p>

<p align="center"><em>Qualitative comparison of skinning prediction. TokenRig produces clean, locally coherent influence maps that closely match the ground truth, while baselines suffer from "bleeding" artifacts across disconnected mesh parts.</em></p>

See the [project page](https://zjp-shadow.github.io/works/SkinTokens/) for the full teaser video and additional qualitative comparisons ([skeleton generation](https://zjp-shadow.github.io/works/SkinTokens/static/images/compare_rigging.png) and [impact of GRPO](https://zjp-shadow.github.io/works/SkinTokens/static/images/compare_grpo.png)).

## 📦 Installation

### Prerequisites
- **Hardware**: An NVIDIA GPU with at least **14 GB** of memory is required for inference.
- **Software**:
  - Python >= 3.11
  - [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) >= 12.1
  - [uv](https://docs.astral.sh/uv/) is recommended for managing dependencies.

### Installation Steps

1. Clone the repo:
    ```sh
    git clone https://github.com/VAST-AI-Research/SkinTokens.git
    cd SkinTokens
    ```

2. Create a virtual environment and install PyTorch:
    ```sh
    uv venv --python 3.11
    source .venv/bin/activate
    uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
    ```

    > [!NOTE]
    > Adjust the CUDA version in the PyTorch index URL to match your driver. See [PyTorch Get Started](https://pytorch.org/get-started/locally/) for other CUDA versions.

3. Install dependencies:
    ```sh
    uv pip install -r requirements.txt
    ```

4. (Optional) Install [flash-attn](https://github.com/Dao-AILab/flash-attention):
    ```sh
    uv pip install flash-attn --no-build-isolation
    ```

    > [!NOTE]
    > flash-attn is optional. Without it, the LLM and VAE use PyTorch SDPA
    > (auto-detected). Force a backend with `SKINTOKENS_ATTN_IMPL=sdpa|eager|flash_attention_2`.

5. Download pretrained models:
    ```sh
    python download.py --model
    ```

    This downloads the TokenRig and FSQ-CVAE checkpoints to `experiments/`, and the Qwen3-0.6B config to `models/`.

## 🤖 Pretrained Models

We provide the following pretrained models on [Hugging Face](https://huggingface.co/VAST-AI/SkinTokens):

| Model | Description | Download |
| --- | --- | --- |
| `articulation_xl_quantization_256_token_4` | TokenRig autoregressive rigging model, trained on ArticulationXL 2.0 + VRoid Hub + ModelsResource and refined with GRPO (**recommended**) | [Download](https://huggingface.co/VAST-AI/SkinTokens/tree/main/experiments/articulation_xl_quantization_256_token_4) |
| `skin_vae_2_10_32768` | FSQ-CVAE (SkinTokens) — skin-weight tokenizer used to encode and decode skinning weights | [Download](https://huggingface.co/VAST-AI/SkinTokens/tree/main/experiments/skin_vae_2_10_32768) |

## 💡 Usage

### Hugging Face Space Demo

The easiest way to try TokenRig without any local setup is the hosted [Hugging Face Space](https://huggingface.co/spaces/VAST-AI/SkinTokens) — upload a mesh and get a rigged result in the browser.

### Gradio Demo (local)

```sh
python demo.py
```

Then open `http://127.0.0.1:1024` in your browser.

### Command Line

```sh
# Rig a single model
python demo.py --input examples/giraffe.glb --output results/giraffe.glb

# Batch process a directory (output defaults to outputs/{input_name}_bind.glb)
python demo.py --input examples/

# Process directory with custom output directory
python demo.py --input examples/ --output results/
```

Inputs may be `.glb`, `.gltf` (with embedded or adjacent resources) or `.obj`; any existing skin/skeleton in the input is ignored, and results are always a skinned-mesh-only `.glb` (materials and textures are not exported). Only triangle geometry is processed (triangle strips/fans are triangulated); points and lines are ignored, and files without triangles are rejected. Geometry I/O is native Python/NumPy (`src/rig_package/glb_io`) — Blender (`bpy`) is not required, and FBX is not supported (convert FBX to GLB first). Draco/meshopt-compressed GLBs and the `EXT_mesh_gpu_instancing` extension are not supported; expand instances before importing.

### Docker (NVIDIA GPU)

The Compose setup follows the same external-model-volume pattern as TRELLIS.2.
Project source is mounted at `/workspace/SkinTokens`, while checkpoints are read
from the external `models` volume at `/models/SkinTokens/experiments`.

```powershell
# Copy local experiments/ checkpoints into the shared Docker volume (once)
.\scripts\copy_weights_to_volume.ps1

# Build and start the service
docker compose build
docker compose up -d
```

The image is based on `torch-runtime:2.11.0-cu128-py3.11` (PyTorch 2.11 + CUDA
12.8 + Triton preinstalled). Override it with `$env:SKINTOKENS_BASE_IMAGE`.

Open `http://127.0.0.1:8087`. To run command-line inference inside the same
environment:

```powershell
docker compose run --rm skintokens python demo.py `
  --input examples/giraffe.glb --output results/giraffe.glb
```

Paths can be overridden with `SKINTOKENS_MODEL_ROOT`,
`SKINTOKENS_EXPERIMENTS_DIR`, and `SKINTOKENS_DATA_ROOT`. This also makes saved
checkpoint references such as `experiments/...` portable between host training
and containers.

Docker defaults to a lighter inference profile than the bare Python server:
`SKINTOKENS_DEFAULT_NUM_BEAMS=4`, one sample, and CUDA cache release after each
request. For the previous beam-search behavior, set:

```powershell
$env:SKINTOKENS_DEFAULT_NUM_BEAMS = "8"
$env:SKINTOKENS_EMPTY_CACHE_AFTER_INFERENCE = "0"
docker compose up -d
```

The server runs a pool of worker processes, each with its own copy of the
model on the GPU and serving one request at a time, so `SKINTOKENS_WORKERS`
requests (default 2) are parsed, sampled and exported in parallel;
further requests queue in FIFO order and receive `queue_ahead` heartbeats.
Each worker needs roughly 2-4 GB of VRAM. Measure concurrent throughput with:

```powershell
python scripts/load_test.py --server http://127.0.0.1:8087 -c 4 --json after.json
python scripts/load_test.py --compare before.json after.json
```

Each load-test run is capped at 120 seconds.

With `SKINTOKENS_FAST_DECODE=1` (Compose default) each decode step replays a
CUDA graph, and these switches control the rest of the step (defaults shown;
set them to compare or to fall back):

| Variable | Default | Effect |
|---|---|---|
| `SKINTOKENS_DECODE_ATTN` | `triton` (`math` without Triton) | Decode attention over the static cache: `triton` computes only the filled slots, `math` is fused grouped-query attention over the whole cache, `sdpa` is the previous memory-efficient SDPA path. |
| `SKINTOKENS_GPU_GRAMMAR` | `1` | Enforce the rig-token grammar with GPU state tensors instead of copying the sequence to the CPU every token. |
| `SKINTOKENS_GRAMMAR_COMPILE` | `1` with CUDA + Triton | Fuse the grammar step with temperature/top-k/top-p into one compiled function. |
| `SKINTOKENS_DEFERRED_STOP` | `1` | Check for the end of generation one step late on CUDA, so the host never waits for the GPU between tokens. |

For lower peak VRAM at the cost of much slower autoregressive decoding, set
`SKINTOKENS_USE_CACHE=0`. `/dev/shm` can be adjusted with
`SKINTOKENS_SHM_SIZE`; the Compose default is `1gb` because inference uses
single-worker dataloading.

> [!NOTE]
> This public repository currently contains no training launcher or training
> configuration, and both published model classes leave `training_step()`
> unimplemented. The image includes Lightning and mounts the project read-write,
> so upstream/private training code can be run with
> `docker compose run --rm skintokens python <training-entrypoint>`, but the
> released source itself cannot start model training as-is.

### Generation Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--top_k` | 1 | Top-k sampling |
| `--top_p` | 1 | Top-p (nucleus) sampling |
| `--temperature` | 0.1 | Sampling temperature |
| `--repetition_penalty` | 1.0 | Repetition penalty |
| `--num_beams` | 5 | Number of beams for beam search |

### Troubleshooting

- **Server fails to start**: Make sure `http_proxy` / `https_proxy` environment variables are unset or correctly configured.

## 🙏 Acknowledgements

- [UniRig](https://github.com/VAST-AI-Research/UniRig) — the predecessor to this work.
- [Qwen3](https://github.com/QwenLM/Qwen3) — the LLM architecture used by the TokenRig autoregressive backbone.
- [Michelangelo](https://github.com/NeuralCarver/Michelangelo) — 3D shape encoder.
- [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet) — shape-representation backbone used by the FSQ-CVAE.
- [FSQ](https://arxiv.org/abs/2309.15505) — Finite Scalar Quantization, the discretization scheme behind SkinTokens.
- [GRPO (DeepSeekMath)](https://arxiv.org/abs/2402.03300) — the policy-optimization method used for RL refinement.
- [Tripo](https://www.tripo3d.ai) — the 3D generative studio from Tripo, a broader context for this line of work.

We sincerely appreciate the contributions of these excellent projects and their authors. We believe open source helps accelerate research, lower barriers to innovation, and make progress more accessible to the broader community.

## License

This project is licensed under the [MIT License](LICENSE).

## 📜 Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@article{zhang2026skintokens,
  title   = {SkinTokens: A Learned Compact Representation for Unified Autoregressive Rigging},
  author  = {Zhang, Jia-Peng and Pu, Cheng-Feng and Guo, Meng-Hao and Cao, Yan-Pei and Hu, Shi-Min},
  journal = {arXiv preprint arXiv:2602.04805},
  year    = {2026}
}
```
