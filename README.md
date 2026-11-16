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

This is the [marklalon/SkinTokens fork](https://github.com/marklalon/SkinTokens) of [VAST-AI-Research/SkinTokens](https://github.com/VAST-AI-Research/SkinTokens). The paper, project page, hosted demo and pretrained weights linked above belong to the upstream project; they do not describe or demonstrate this fork's serving behavior.

This fork focuses on inference: native GLB/glTF I/O without Blender, a WebSocket service with separate CPU/GPU workers and continuous batching, CUDA-graph decoding, and compact rig payloads that clients apply to the original asset while keeping its materials and textures. `demo.py` is CLI-only; the local Gradio UI has been removed. Training and GRPO launchers are not included.

SkinTokens is the successor to [UniRig](https://github.com/VAST-AI-Research/UniRig) (SIGGRAPH '25). While UniRig uses separate stages for skeleton prediction and skinning, SkinTokens unifies both into a single autoregressive sequence via learned discrete skin tokens.

## 🔮 Overview

The upstream method takes a **single 3D mesh** as input and autoregressively produces a skeleton hierarchy plus skinning weights in a single sequence. Its research pipeline has three stages (this repository provides inference with pretrained models):

1. **Learn SkinTokens** — An FSQ-CVAE compresses sparse skinning weights into a compact discrete vocabulary.
2. **Unified Autoregressive Modeling** — A Qwen3-0.6B-based Transformer generates the full rig as one sequence: skeleton tokens followed by SkinTokens.
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
- **Hardware**: An NVIDIA GPU is recommended for inference and required for CUDA-graph decoding and continuous batching. VRAM use depends on the mesh, skeleton, worker count and active batch size; the upstream 14 GB requirement is not a measured minimum for this fork. A CPU fallback is available with `SKINTOKENS_DEVICE=cpu`, without CUDA acceleration.
- **Software**:
  - Python >= 3.11
  - A driver compatible with the installed PyTorch CUDA build (the commands below use CUDA 12.8). A local CUDA toolkit is needed when building CUDA extensions such as optional flash-attn.
  - [uv](https://docs.astral.sh/uv/) is recommended for managing dependencies.

### Installation Steps

1. Clone the repo:
    ```sh
    git clone https://github.com/marklalon/SkinTokens.git
    cd SkinTokens
    ```

2. Create a virtual environment and install PyTorch:
    ```sh
    uv venv --python 3.11
    source .venv/bin/activate
    uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
    ```

    > [!NOTE]
    > On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`. The CUDA build must be supported by your NVIDIA driver. Torchvision and torchaudio are not needed for inference.

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
    > (auto-detected). `SKINTOKENS_ATTN_IMPL=sdpa|eager|flash_attention_2` selects the LLM's standard attention backend; fast single-token decode has its own `SKINTOKENS_DECODE_ATTN` setting below.

5. Download pretrained models:
    ```sh
    python download.py --model
    ```

    By default this downloads the TokenRig and FSQ-CVAE checkpoints to `experiments/`, and the Qwen3-0.6B config/tokenizer files to `models/Qwen3-0.6B/` (not the separate Qwen model weights). The downloader places these under `SKINTOKENS_MODEL_ROOT` when set; `SKINTOKENS_EXPERIMENTS_DIR` changes checkpoint lookup, not the download destination.

## 🤖 Pretrained Models

The fork uses the upstream pretrained models on [Hugging Face](https://huggingface.co/VAST-AI/SkinTokens):

| Model | Description | Download |
| --- | --- | --- |
| `articulation_xl_quantization_256_token_4` | TokenRig autoregressive rigging model, trained on ArticulationXL 2.0 + VRoid Hub + ModelsResource and refined with GRPO (**recommended**) | [Download](https://huggingface.co/VAST-AI/SkinTokens/tree/main/experiments/articulation_xl_quantization_256_token_4) |
| `skin_vae_2_10_32768` | FSQ-CVAE (SkinTokens) — skin-weight tokenizer used to encode and decode skinning weights | [Download](https://huggingface.co/VAST-AI/SkinTokens/tree/main/experiments/skin_vae_2_10_32768) |

## 💡 Usage

### Hugging Face Space Demo

The upstream [Hugging Face Space](https://huggingface.co/spaces/VAST-AI/SkinTokens) is linked for reference. It runs independently of this fork.

### WebSocket Server and Python Client

```sh
python serve.py
```

The server listens on port 8087 by default. `GET /health` reports readiness and supported output formats; `WS /ws/generate` accepts generation requests. There is no browser UI or REST generation endpoint. After the model finishes loading, run the client in another terminal:

```sh
python skintokens_client.py --server 127.0.0.1:8087 --input examples/giraffe.glb --output results/giraffe.glb
```

The client needs `websockets` (included via `uvicorn[standard]`), NumPy, SciPy and this repository's `src` package, but no PyTorch or GPU on the client machine. By default it requests a compact rig payload and adds the rig to the original asset, preserving materials and textures. For GLB inputs it uploads a reduced geometry-only representation; the original file stays on the client for reconstruction.

### Command Line

```sh
# Rig a single model
python demo.py --input examples/giraffe.glb --output results/giraffe.glb

# Recursively process a directory, keeping relative paths under results/
python demo.py --input examples/ --output results/
```

`demo.py` exports a skinned-mesh-only `.glb`, without materials or textures. For a single file, the default output is `outputs/{input_name}_bind.glb`; specify an explicit output directory for directory inputs. Multiple samples add `_0`, `_1`, etc. to the output stem. Both the CLI and Python client ground the result by default; pass `--no-auto-ground` to preserve placement (the client requires `--output-format rig` for this option).

Inputs may be `.glb` or `.gltf`. Local CLI input can reference adjacent resources; server uploads and the Python client require embedded geometry buffers, since only one file is sent. Embed images as well for a self-contained result. Existing skins are not used to condition generation: the skeleton and weights are always generated anew. Only triangle geometry is processed (triangle strips/fans are triangulated); points and lines are ignored during inference, and files without triangles are rejected. Geometry I/O is native Python/NumPy/SciPy (`src/rig_package/glb_io`) — Blender (`bpy`) is not required, and FBX is not supported (convert FBX to GLB first). Draco/meshopt-compressed GLBs and the `EXT_mesh_gpu_instancing` extension are not supported; expand instances before importing.

### Docker (NVIDIA GPU)

The image is self-contained: the `Dockerfile` copies the project source and the
model checkpoints (`experiments/`, `models/`) into the image at build time.
Run `python download.py --model` on the host before building. The default base
image, `torch-runtime:2.11.0-cu128-py3.11`, must also be available locally or
from your registry; this repository does not build it. The image contains only
the service runtime, not `demo.py`, the client or `examples/`. Logs and failure
dumps live on the `skintokens-outputs` Docker volume; generated results are sent
to clients and saved by the client.

```powershell
# Build and start the service
docker compose build
docker compose up -d
```

The image is based on `torch-runtime:2.11.0-cu128-py3.11` (PyTorch 2.11 + CUDA
12.8 + Triton preinstalled). Override it with `$env:SKINTOKENS_BASE_IMAGE`.

Check `http://127.0.0.1:8087/health`, then run the client on the host:

```powershell
python skintokens_client.py --server 127.0.0.1:8087 `
  --input examples/giraffe.glb --output results/giraffe.glb
```

For local Python execution, paths can be overridden with `SKINTOKENS_MODEL_ROOT`,
`SKINTOKENS_EXPERIMENTS_DIR`, and `SKINTOKENS_DATA_ROOT`. This also makes saved
checkpoint references such as `experiments/...` portable between host training
and containers.

Requests default to one sample. Compose allows up to four samples per request
(`SKINTOKENS_MAX_NUM_SAMPLES=4`), while standalone `serve.py` allows one by
default. CUDA cache release is enabled. To disable it for a one-off container:

```powershell
docker compose run --rm --service-ports -e SKINTOKENS_EMPTY_CACHE_AFTER_INFERENCE=0 skintokens
```

Stop the existing service before using `--service-ports`. For the regular
service, edit `environment` in `docker-compose.yml` or use a Compose override.
Only values written as `${...}` in the Compose file inherit host environment
variables; settings such as fast decode, cache release and `shm_size` are
currently literal values.

The server runs each request in three stages on two pools of worker
processes: a CPU worker parses the upload and samples its point clouds, a GPU
worker samples the rig, and a CPU worker exports the result. The CPU and GPU
stages of different requests overlap: while one request is on the GPU, the
next is parsed and the previous one exported. Each GPU worker has its own copy
of the model; `SKINTOKENS_WORKERS` (default 1) sets the number of GPU workers
**on each visible GPU** (pinned round-robin). `SKINTOKENS_CPU_WORKERS`
(default 0 = twice the GPU workers) sets the number of CPU workers, which load
no model and see no GPU. A request waiting for a stage queues in FIFO order
(exports ahead of new parses) and receives `queue_ahead` heartbeats. Each GPU
worker holds its own model weights and KV caches; memory use grows with active
slots and skin decoding. Measure concurrent throughput on your hardware with:

```powershell
python load_test.py run --server http://127.0.0.1:8087 --concurrency 4 --requests 12 --out "$env:TEMP/skintokens-after.json"
python load_test.py compare "$env:TEMP/skintokens-before.json" "$env:TEMP/skintokens-after.json"
```

Save a baseline report as `skintokens-before.json` in the same temporary
directory before changing the server configuration and running the comparison.

A GPU worker batches its requests continuously
(`src/model/batch_decode.py`). It takes up to `SKINTOKENS_BATCH_SLOTS`
(default 4; 1, 2, 3 or 4) requests at once and decodes their point clouds
together, one per slot. A request joins as soon as a slot frees and leaves
when its rig ends, without waiting for the others. Each sample consumes a
decode row, so a multi-sample request may occupy several slots. Batching can
improve throughput; speedup depends on hardware and workload.
Each batch size's decode step is captured once as a CUDA graph. The decode
batch matches the number of active rows: three rows use batch 3, even with
four slots configured.
`SKINTOKENS_BATCH_SLOTS=0`, or fast decode off, or a CPU device, serves one
request at a time through `transformers`' `generate`, as before.

Start with one worker per GPU: additional workers duplicate model memory and
split requests that could otherwise share a batch. Benchmark before increasing
the worker count.

With `SKINTOKENS_FAST_DECODE_BACKEND=inductor` (the Compose default), the
step is compiled before it is captured: startup takes longer, but each step
can run faster than the unfused path. Standalone Python defaults to
`SKINTOKENS_FAST_DECODE_BACKEND=cudagraphs`, which does not require Triton.

> [!NOTE]
> Results are not exactly reproducible under batching. Decoding is greedy,
> and the kernels that compute a row depend on how many other rows decode
> beside it, so a near-tie between two tokens (in practice, two adjacent
> coordinate bins) can go either way and the same input and `seed` may give
> a slightly different rig depending on the server's load.
> `SKINTOKENS_DETERMINISTIC` does not remove this. Without batching
> (`SKINTOKENS_BATCH_SLOTS=0`), use `SKINTOKENS_DETERMINISTIC=1` for
> reproducibility across workers and restarts on the same hardware/software stack.

With fast decode on (the default) each decode step replays a CUDA graph, and
these switches control the rest of the step (defaults shown; set them to
compare or to fall back):

| Variable | Default | Effect |
|---|---|---|
| `SKINTOKENS_DECODE_ATTN` | `triton` (`math` without Triton) | Decode attention over the static cache: `triton` skips unfilled cache blocks, `math` uses tensor operations over the whole cache (fused with inductor), and `sdpa` uses PyTorch SDPA. |
| `SKINTOKENS_GPU_GRAMMAR` | `1` | Enforce the rig-token grammar with GPU state tensors instead of copying the sequence to the CPU every token. |
| `SKINTOKENS_GRAMMAR_COMPILE` | `1` with CUDA + Triton | Fuse the grammar step (state update and logit mask) into one compiled function. |
| `SKINTOKENS_DEFERRED_STOP` | `1` | Check for the end of generation one step late on CUDA, so the host never waits for the GPU between tokens. |
| `SKINTOKENS_DETERMINISTIC` | `0` | Set to `1` for deterministic kernels, at a performance cost. Without batching (`SKINTOKENS_BATCH_SLOTS=0`), this supports seed reproducibility across workers and restarts on the same hardware/software stack. With batching, results still depend on the other rows in the batch. Off, a seed only supports reproducibility within one worker process with the same execution path. |

`/dev/shm` can be adjusted with
the `shm_size` field in `docker-compose.yml` (currently `1gb`); there is no
`SKINTOKENS_SHM_SIZE` interpolation in the supplied Compose file.

> [!NOTE]
> This public repository currently contains no training launcher or training
> configuration, and both published model classes leave `training_step()`
> unimplemented. The image includes Lightning but only copies inference runtime
> files and does not bind-mount the source. Training requires additional code,
> configuration and an appropriate image or explicit source mount.

### Generation Parameters

Decoding is fixed to greedy single-beam search: there are no top-k, top-p,
temperature, repetition-penalty or beam settings. The only randomness is where
points are sampled on the input mesh, and that is controlled by the seed.
`demo.py` takes the sampling parameters below; `skintokens_client.py` and the
WebSocket API take those and `output`, the result format:

| Parameter | Default | Description |
| --- | --- | --- |
| `--num-samples` (WebSocket: `num_samples`) | 1 | Rigs per input. The server rejects values above `SKINTOKENS_MAX_NUM_SAMPLES` (standalone default: 1; Compose: 4; configurable from 1 to 8). Each sample uses its own point sampling; different rigs are not guaranteed. The local CLI only enforces a minimum of 1. |
| `--seed` (WebSocket: `seed`) | random | Seed for point sampling, in `[0, 2**63)`. Without it a random seed is printed by the CLI/client and returned in every server `done` message. Exact rig reproducibility also depends on deterministic kernels, the execution path and batching, as described above. |
| `--output-format` (WebSocket: `output`) | `rig` | What the server sends: `rig`, a rig payload that the client turns into the uploaded file with the skeleton and skin added (see [Server Results: Rig Payloads](#server-results-rig-payloads)); or `glb`, the server's skinned-mesh-only GLB, served only with `SKINTOKENS_ENABLE_GLB_OUTPUT=1`. |

### GLB Weight Encoding

GLB exports use normalized 16-bit integer skin weights by default (8 bytes per
vertex for four influences, versus 16 bytes with float32). Integer weights sum
exactly to 65535 per vertex before decoding, avoiding weight-sum drift during
skinning. Python callers of `transfer_rigging`, `GlbParser.export_asset`, or
`export_skinned_glb` can set `weight_bits=16` (default) or `32` for float32 weights.
`do_not_normalize=True` (`normalize=False` in `export_skinned_glb`) always
keeps float32 weights to preserve the supplied sums. Total file savings depend
on geometry and index sizes.

### Server Results: Rig Payloads

Most of a rigged GLB is the input's own positions, normals and triangles,
which the client already has. So the server returns a **rig payload** by default:
just the skeleton and the per-vertex joint influences, with the binary arrays
zlib-compressed. Payload size depends on the asset. It uses the same four
influences and normalized 16-bit weights as the default server GLB export,
without an additional lossy compression step. The client adds the rig to its
original asset, preserving data that the server's skinned-mesh-only GLB drops.

| Client | How |
| --- | --- |
| Python | `skintokens_client.py` does it by default. In code: `apply_rig(uploaded_bytes, payload, name="input.glb")` from `src.rig_package.glb_io` returns the input as a GLB with the rig added. (`build_glb` with the same arguments returns the server's GLB instead, byte for byte.) |
| three.js | [`clients/threejs/skintokens-rig.js`](clients/threejs/skintokens-rig.js): `await applyRigPayload(gltf, await decodeRigPayload(payload))` turns the meshes `GLTFLoader` loaded from the uploaded file into `SkinnedMesh`es on the new skeleton. |

`apply_rig` preserves the input's node hierarchy, materials, textures, UVs,
animation data, extensions and extras while replacing its skin bindings.
Preserved animations are not retargeted to the generated skeleton. Embedded `.gltf`
buffers move into the GLB's binary chunk, with their bytes unchanged. It only
adds the following:
- **Skeleton:** an armature node holding the joints, added to the default scene.
- **Skins:** each inverse bind matrix includes the mesh node's placement, so vertex data is left as stored. glTF ignores a skinned mesh node's own transform. Skins of the input that no node refers to any more are dropped.
- **`JOINTS_0`/`WEIGHTS_0`:** added to every primitive of the rigged meshes, replacing existing joint/weight attributes (including additional influence sets). Points and lines take the influences of their nearest rigged vertex.
- **Mesh copies:** a mesh used by several nodes gets one copy per extra node, sharing its geometry, so each use can have its own weights.

The Python API `apply_rig` defaults to `auto_ground=False`; the Python CLI
client passes `auto_ground=True` unless `--no-auto-ground` is set. The three.js
helper preserves placement unless passed `{ ground: true }`. Grounding applies
the payload's world-space shift (Y-up in glTF) to the scene roots and armature.
The glTF validator may warn `NODE_SKINNED_MESH_NON_ROOT` or
`NODE_SKINNED_MESH_LOCAL_TRANSFORMS` for preserved transformed hierarchies;
the inverse bind matrices account for mesh placement.

**Protocol.** The request header names the format: `{"filename": ..., "output": "rig"}`.
`output` defaults to `"rig"`, and `/health` lists the formats the server
supports in `output_formats`. Each result is a `done` message (`"format":
"rig"`, `"payload_size": N`) followed by one binary frame. The whole GLB
(`"output": "glb"`) is served only when the server sets
`SKINTOKENS_ENABLE_GLB_OUTPUT=1`; any other request is refused before its
upload.

**Format** (version 1, little-endian; full spec in
[`rig_payload.py`](src/rig_package/glb_io/rig_payload.py)). The container is
laid out like a GLB: `"SKRG"`, `u32 version`, `u32 total length`, then a JSON
chunk and a BIN chunk.
- **JSON chunk:**
  - `joint_names`, `parents` (`-1` for roots), and `joint_heads`, which are bind-pose heads in glTF world space (Y-up).
  - `ground_offset`, the shift the server applies to meshes and joints together to put the model on the ground.
  - `vertex_count`, and `influences` (4).
  - The byte ranges of the two arrays, and `bin.compression`, which is `zlib` (RFC 1950, which browsers call `"deflate"` in `DecompressionStream`).
  - `layout`.
- **BIN chunk:** two row-major `(vertex_count, 4)` arrays: `joint_indices` (`uint8`, or `uint16` above 256 joints) and `joint_weights` (normalized `uint16`, each row summing to 65535).

Rows follow `layout`, which spells out the vertex order:
- The default scene's mesh nodes, depth-first.
- Within each node, its triangle primitives in order. Each primitive's `index` is its position in the glTF mesh's `primitives` list; points, lines and primitives without triangles are skipped.
- A mesh used by several nodes appears once per node.
- `skinned_input` marks nodes that were skinned in the input. The server used their vertices as stored and ignored their node transform.

Clients check `layout` against their own copy of the file and reject mismatched
layouts. This is not a content hash: two different files with the same layout
can pass the check. Apply a payload only to the asset used for its request.

### Troubleshooting

- **Missing checkpoints/config**: Run `python download.py --model` before local inference or a Docker build. Check path overrides if models are stored elsewhere.
- **Docker base image unavailable**: Provide the default `torch-runtime:2.11.0-cu128-py3.11` image or set `SKINTOKENS_BASE_IMAGE` to a compatible Linux image with PyTorch, CUDA and Triton installed.
- **Client rejected / still loading**: Check `/health` and server logs. Compose limits uploads to 32 MiB and samples to 4 by default; standalone defaults are 64 MiB and 1 sample. Geometry limits are configured by `SKINTOKENS_INPUT_*` (see [`limits.py`](src/rig_package/glb_io/limits.py)).
- **External glTF buffers rejected**: Embed buffers or convert the asset to a self-contained GLB before sending it to the service.
- **Need materials/textures in the output**: Use `skintokens_client.py` with the default `--output-format rig`. Local `demo.py` and server `output=glb` export only the skinned mesh.

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
