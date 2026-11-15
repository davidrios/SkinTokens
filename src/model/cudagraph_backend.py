"""A ``cudagraphs`` compile backend that does not copy the weights or KV cache.

torch's own ``cudagraphs`` backend (``torch._dynamo.backends.cudagraphs``)
passes only the first ``fixed`` graph inputs to CUDA-graph trees as static.
With ``inline_inbuilt_nn_modules`` (the default since torch 2.4) parameters are
ordinary graph inputs, so ``fixed`` is 0 and nothing is static: the recording
keeps a copy of every weight and of the static KV cache in its private pool
(~1.1 GiB for the 0.6B LLM) and copies them all into it before every replay,
i.e. every decoded token. The cache's in-place update is also taken out of the
graph (functionalized), so each step writes out a whole new cache and copies it
back.

``skintokens_cudagraphs`` is that backend with two changes, both what inductor
already does: the static inputs are the ones dynamo marked
(``TracingContext.fw_metadata.static_input_indices``: the parameters and the
tensors given ``torch._dynamo.mark_static_address``, as transformers does for
its static cache), and mutations of inputs stay in the graph. A static input
keeps its address across replays, so the graph reads and updates the real
weights and cache in place.
"""
from __future__ import annotations

import contextlib
import threading
from typing import Any, Sequence

import torch

BACKEND_NAME = "skintokens_cudagraphs"

_lock = threading.Lock()


def _static_input_idxs(fixed: Sequence[int]) -> list[int]:
    context = torch._guards.TracingContext.try_get()
    marked = context.fw_metadata.static_input_indices if context and context.fw_metadata else ()
    return sorted(set(fixed) | set(marked))


@contextlib.contextmanager
def _static_marked_inputs():
    """Within it, torch's cudagraphs backend treats the marked inputs as static."""
    import torch._inductor.cudagraph_trees as trees

    original = trees.cudagraphify_impl

    def cudagraphify_impl(model, inputs, static_input_idxs, *args, **kwargs):
        return original(model, inputs, _static_input_idxs(static_input_idxs), *args, **kwargs)

    # The backend imports cudagraphify_impl from the module when it compiles,
    # so the swap covers exactly the compilation made inside this block.
    with _lock:
        trees.cudagraphify_impl = cudagraphify_impl
        try:
            with torch._dynamo.config.patch(
                cudagraph_backend_keep_input_mutation=True,
                cudagraph_backend_support_input_mutation=True,
            ):
                yield
        finally:
            trees.cudagraphify_impl = original


def skintokens_cudagraphs(gm: torch.fx.GraphModule, example_inputs: Sequence[Any]) -> Any:
    from torch._dynamo.backends.cudagraphs import cudagraphs

    with _static_marked_inputs():
        return cudagraphs(gm, example_inputs)


def register() -> str:
    """Register the backend with torch.compile (idempotent); returns its name."""
    from torch._dynamo.backends.registry import _COMPILER_FNS, register_backend

    if BACKEND_NAME not in _COMPILER_FNS:
        register_backend(name=BACKEND_NAME, compiler_fn=skintokens_cudagraphs)
    return BACKEND_NAME
