"""SkinTokens / TokenRig WebSocket client — stream progress and save a rigged GLB.

Example:
    python skintokens_client.py --input assets/character.glb --output out.glb
    python skintokens_client.py --input model.glb --server http://HOST:8087

By default the server sends a rig payload (skeleton and skin only, several
times smaller than a GLB) and this client adds it to the input file: the output
is the input GLB/glTF (as a GLB) with its node structure, materials, textures
and UVs kept, plus the skeleton and skin. That needs NumPy, SciPy and this
repository's ``src`` package next to the script. ``--output-format glb`` asks
for the server's GLB instead (the skinned mesh only, no materials; served only
with SKINTOKENS_ENABLE_GLB_OUTPUT=1) and needs only ``websockets``.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from contextlib import suppress
from pathlib import Path


def _websocket_url(server: str) -> str:
    base = server.rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://"):]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://"):]
    elif not base.startswith(("ws://", "wss://")):
        base = "ws://" + base
    return base + "/ws/generate"


class ProgressDisplay:
    """Render progress in place on a terminal, or as lines when redirected."""

    def __init__(self, width: int = 30):
        self.width = width
        self.last_length = 0
        self.active = False

    def update(self, percent: int, step: str, elapsed: float | None = None) -> None:
        percent = max(0, min(100, int(percent)))
        filled = round(self.width * percent / 100)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed_text = f"  {elapsed:.1f}s" if elapsed is not None else ""
        line = f"[client] [{bar}] {percent:3d}%  {step}{elapsed_text}"

        if sys.stdout.isatty():
            sys.stdout.write("\r" + line.ljust(self.last_length))
            sys.stdout.flush()
            self.last_length = max(self.last_length, len(line))
            self.active = True
        else:
            print(line)

    def finish(self) -> None:
        if self.active:
            print()
            self.active = False


def _sample_output_path(output_path: str, sample_index: int, total_samples: int) -> str:
    if total_samples <= 1:
        return output_path
    path = Path(output_path)
    return str(path.with_name(f"{path.stem}_{sample_index}{path.suffix or '.glb'}"))


def _write(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def _rebuilder(file_data: bytes, filename: str):
    """A function turning a rig payload into ``file_data`` rigged: the input
    file with the skeleton and skin added, its materials, textures and node
    structure kept.

    The input is parsed once (as the server parsed the upload) for all samples.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from src.rig_package.glb_io import apply_rig, load_scene
        # apply_rig imports SciPy lazily, for primitives without rig rows, so
        # probe it here where the failure can still be explained.
        import scipy.spatial  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"applying the rig needs NumPy, SciPy and this repository's src package ({exc}); "
            "or pass --output-format glb if the server serves GLBs"
        ) from exc
    scene = load_scene(file_data, name="input" + Path(filename).suffix.lower())
    return lambda payload: apply_rig(scene, payload)


async def _run(args) -> None:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError(
            "missing dependency 'websockets'; install it with: pip install websockets"
        ) from exc

    ws_url = _websocket_url(args.server)
    with open(args.input, "rb") as f:
        file_data = f.read()

    filename = os.path.basename(args.input)

    payload = {
        "filename": filename,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "output": args.output_format,
    }
    # Parsed before connecting, so an input that cannot be rigged fails early.
    rebuild = _rebuilder(file_data, filename) if args.output_format == "rig" else None

    progress = ProgressDisplay()
    started_at = time.monotonic()
    print(f"[client] connecting {ws_url}")
    try:
        async with websockets.connect(
            ws_url,
            # Rigged GLBs grow with the mesh (about 70 bytes per vertex).
            max_size=1024 * 1024 * 1024,
            open_timeout=args.timeout,
            # GLB data barely compresses; deflating it costs seconds. Rig
            # payloads arrive compressed already.
            compression=None,
        ) as ws:
            await ws.send(json.dumps(payload))
            await ws.send(file_data)

            glb_count = 0
            expected_samples = args.num_samples
            pending_sample_index = 0
            pending_format = args.output_format
            try:
                async for raw_message in ws:
                    if isinstance(raw_message, bytes):
                        sample_output = _sample_output_path(
                            args.output, pending_sample_index, expected_samples
                        )
                        glb_count += 1
                        if pending_format == "rig":
                            glb = rebuild(raw_message)
                            print(f"[client] received {len(raw_message)} bytes (rig payload); "
                                  f"added the rig to the input ({len(glb)} bytes GLB)")
                        else:
                            glb = raw_message
                        _write(sample_output, glb)
                        print(f"[client] saved {len(glb)} bytes -> {sample_output}")
                        if glb_count >= expected_samples:
                            return
                        continue

                    message = json.loads(raw_message)
                    stage = message.get("stage", "unknown")

                    if stage == "queued":
                        if message.get("queued"):
                            print("[client] queued; waiting for the GPU")
                        else:
                            print("[client] GPU is available; starting generation")
                    elif stage == "processing":
                        progress.update(
                            message.get("progress", 0),
                            message.get("step", "processing"),
                            message.get("elapsed_sec"),
                        )
                    elif stage == "done":
                        expected_samples = int(message.get("num_samples", expected_samples))
                        pending_sample_index = int(message.get("sample_index", glb_count))
                        pending_format = message.get("format", args.output_format)
                        if pending_format not in ("rig", "glb"):
                            raise RuntimeError(f"server sent an unknown format {pending_format!r}")
                        if pending_format == "rig" and rebuild is None:
                            rebuild = _rebuilder(file_data, filename)
                        progress.update(100, "complete", message.get("elapsed_sec"))
                        progress.finish()
                        if pending_sample_index == 0 and message.get("seed") is not None:
                            print(f"[client] seed {message['seed']} (pass --seed to reproduce)")
                        # The result frame (rig payload or GLB) follows the done message
                        continue
                    elif stage == "cancelled":
                        raise asyncio.CancelledError(message.get("message"))
                    elif stage == "error":
                        err_msg = message.get("message") or "unknown server error"
                        raise RuntimeError(err_msg)
                    else:
                        print(f"[client] server stage: {stage}")
            except asyncio.CancelledError:
                with suppress(Exception):
                    await asyncio.shield(ws.send(json.dumps({"type": "cancel"})))
                raise

            raise RuntimeError("WebSocket closed before the result was received")
    finally:
        progress.finish()
        print(f"[client] request elapsed: {time.monotonic() - started_at:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SkinTokens / TokenRig WebSocket client with live progress"
    )
    parser.add_argument("--input", required=True, help="Input 3D file path (GLB/GLTF)")
    parser.add_argument(
        "--output",
        default=None,
        help="Output GLB path (default: outputs/<input_name>_skined.glb)",
    )
    parser.add_argument("--server", default="http://localhost:8087", help="Server base URL")

    parser.add_argument("--num-samples", type=int, default=1,
                        help="Rigs to generate, each from a different point sampling of the mesh "
                             "(limited by server SKINTOKENS_MAX_NUM_SAMPLES, default: 1)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the point sampling; the same seed and --num-samples reproduce the same "
                             "outputs (default: random, printed after the run)")

    parser.add_argument("--output-format", choices=("rig", "glb"), default="rig",
                        help="What the server sends: 'rig' (default), the skeleton and skin, "
                             "added here to the input file (materials and textures kept); "
                             "'glb', the server's skinned-mesh-only GLB (served only with "
                             "SKINTOKENS_ENABLE_GLB_OUTPUT=1)")
    parser.add_argument("--timeout", type=int, default=30, help="Connection timeout seconds")
    args = parser.parse_args()

    if args.output is None:
        input_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"outputs/{input_name}_skined.glb"

    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\n[client] cancelled", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        err_msg = str(exc).strip()
        if not err_msg:
            err_msg = f"{type(exc).__name__} (no detail – check server logs)"
        print(f"[client] error: {err_msg}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
