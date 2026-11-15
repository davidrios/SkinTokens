"""Continuous-batching greedy decode for TokenRig's LLM.

``TokenRig.generate`` decodes one request's point clouds as one batch with
``transformers.generate``: the batch runs until its longest row ends, and
nothing can join it meanwhile. :class:`BatchDecoder` instead keeps a fixed
number of row *slots* (up to 4) in one static KV cache. A row (one point
cloud's prompt) is prefilled on its own and inserted into a free slot between
two decode steps; a row that ends frees its slot for the next one, whatever the
other rows are doing. Decode attention is memory bound at these batch sizes,
so a step over 4 rows costs little more than a step over one.

How a step works:

* Each slot has its own write position (``pos``). The cache passed to the
  transformer (:class:`_SlotCache`) writes each row's new key/value at that
  row's position, and the attention mask lets row ``b`` see slots
  ``0..pos[b]``. RoPE gets the per-row positions through ``position_ids``.
* The grammar (``src/model/grammar.py``) runs as the same per-row state
  machine as in ``generate``, then the argmax is taken: greedy decoding.
* Rows that have ended keep running (their outputs are dropped) until the host
  notices, so a step needs no host round-trip. The host reads the rows' status
  through an asynchronous copy and frees ended rows a few steps later.
* Occupied slots are kept compact (``0..n-1``), so a step runs on the smallest
  *bucket* (1, 2, 3 or 4 rows) that holds them. Each bucket's step is captured
  once as a CUDA graph and replayed.

Batching changes which kernels run for a row (their shapes depend on the
bucket), so a row's result can differ in rare near-ties from the same row
decoded alone, depending on how many other rows were running.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from . import grammar as grammar_module

logger = logging.getLogger(__name__)

BUCKETS = (1, 2, 3, 4)
# Decode steps the host may queue beyond the last status it has read: bounds
# how late an ended row (or a cancellation) is noticed.
MAX_STEPS_AHEAD = 8
# Status rows (see _status): done, generated tokens, in skin phase, tokens up to
# the switch token, bones.
_DONE, _GENERATED, _IN_SKIN, _SWITCH_LEN, _BONES = range(5)


class _SlotCache:
    """A transformers cache that writes each row's one new key/value at that
    row's own position (``transformers.StaticCache`` writes every row at one
    shared position) and returns the whole static buffers."""

    is_compileable = False

    def __init__(self, keys: Sequence[Tensor], values: Sequence[Tensor], pos: Tensor) -> None:
        self.keys = keys
        self.values = values
        self.pos = pos
        self.rows = torch.arange(pos.shape[0], device=pos.device)

    def update(self, key_states: Tensor, value_states: Tensor, layer_idx: int, *args, **kwargs):
        keys, values = self.keys[layer_idx], self.values[layer_idx]
        keys[self.rows, :, self.pos] = key_states[:, :, 0]
        values[self.rows, :, self.pos] = value_states[:, :, 0]
        return keys, values


@dataclass
class _Row:
    key: Any
    slot: int
    # The latest status read for this row (see _status), or None.
    status: Optional[List[int]] = None


@dataclass
class Finished:
    """A row that ended: its generated token ids (on the host, prompt not
    included, final eos included unless it ran out of tokens)."""
    key: Any
    generated_ids: Tensor


@dataclass
class _Snapshot:
    host: Tensor
    event: Any
    rows: Dict[int, Any] = field(default_factory=dict)  # slot -> row key at request time
    step: int = 0


class BatchDecoder:
    """Greedy decoding of up to ``slots`` rows at once, rows joining and
    leaving between steps. Must be used from the thread that captured its
    graphs (:meth:`capture`), with the model's LLM attention set to the
    length-aware decode attention (``src/model/decode_attention.py``)."""

    def __init__(self, model, slots: int, max_new_tokens: int, prompt_length: int,
                 compile_step: bool = False) -> None:
        if slots not in BUCKETS:
            raise ValueError(f"slots must be one of {BUCKETS}, got {slots}")
        tokenizer = model.tokenizer
        if not grammar_module.supports(tokenizer):
            raise ValueError("continuous batching needs the GPU grammar (a TokenizerPart tokenizer)")
        self.model = model
        self.transformer = model.transformer
        self.slots = slots
        self.buckets = tuple(b for b in BUCKETS if b <= slots)
        self.max_new_tokens = int(max_new_tokens)
        self.prompt_length = int(prompt_length)
        self.length = self.prompt_length + self.max_new_tokens  # cache slots per row
        self.eos = int(model.eos)
        self.device = model.device
        # Compile the step with inductor (fusing its many small pointwise ops)
        # before capturing it; needs Triton.
        self.compile_step = compile_step

        cfg = self.transformer.config
        dtype = next(self.transformer.parameters()).dtype
        layers = cfg.num_hidden_layers
        kv_heads = cfg.num_key_value_heads
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        shape = (slots, kv_heads, self.length, head_dim)
        self._keys = [torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(layers)]
        self._values = [torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(layers)]
        self._layer_types = sorted(set(getattr(cfg, "layer_types", None) or ["full_attention"]))

        # The grammar's tables and per-prompt start states.
        width = self.transformer.get_output_embeddings().weight.shape[0]
        self._grammar = grammar_module.GpuGrammarLogitsProcessor(
            tokenizer=tokenizer, switch_token_id=tokenizer.eos, eos_token_id=self.eos,
            tokens_per_skin=model.tokens_per_skin, init=[tokenizer.bos])
        self._transitions = torch.from_numpy(self._grammar._build_transitions(width)).to(self.device)
        self._masks = torch.from_numpy(self._grammar._build_masks(width)).to(self.device)
        self._grammar_kwargs = dict(switch_id=int(tokenizer.eos),
                                    branch_id=int(tokenizer.token_id_branch),
                                    tokens_per_skin=int(model.tokens_per_skin))
        self._init_states: Dict[tuple, tuple] = {}

        long = dict(dtype=torch.long, device=self.device)
        self._tok = torch.zeros(slots, **long)
        self._pos = torch.zeros(slots, **long)
        self._state = torch.zeros(slots, **long)
        self._is_branch = torch.zeros(slots, dtype=torch.bool, device=self.device)
        self._bones = torch.zeros(slots, **long)
        self._skin_count = torch.zeros(slots, **long)
        self._switch_len = torch.zeros(slots, **long)
        self._position = torch.zeros(slots, **long)
        self._generated = torch.zeros(slots, **long)
        self._done = torch.ones(slots, dtype=torch.bool, device=self.device)
        self._out = torch.zeros((slots, self.max_new_tokens), **long)
        self._status = torch.zeros((5, slots), **long)
        self._cache_slots = torch.arange(self.length, **long)

        self._rows: List[_Row] = []  # occupied slots 0..n-1, in slot order
        self._graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        # Each graph's cache: its index tensors were made outside the graph,
        # which reads them in place, so they must outlive it.
        self._caches: Dict[int, _SlotCache] = {}
        self._steps = 0
        self._snapshot: Optional[_Snapshot] = None
        self._spare_host: Optional[Tensor] = None

    # ── capacity ─────────────────────────────────────────────────────────────

    @property
    def active(self) -> int:
        return len(self._rows)

    @property
    def free(self) -> int:
        return self.slots - len(self._rows)

    def kv_cache_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._keys + self._values)

    # ── rows ─────────────────────────────────────────────────────────────────

    def _init_state(self, start_tokens: Sequence[int]) -> tuple:
        key = tuple(int(t) for t in start_tokens)
        state = self._init_states.get(key)
        if state is None:
            state = self._init_states[key] = self._grammar._run_init(np.asarray(key, dtype=np.int64))
        return state

    @torch.inference_mode()
    def insert(self, key: Any, inputs_embeds: Tensor, start_tokens: Sequence[int]) -> None:
        """Prefill a row and put it in the next free slot. *inputs_embeds* is
        its ``(L, hidden)`` prompt, *start_tokens* the token part of it."""
        from transformers import DynamicCache

        if not self.free:
            raise RuntimeError("no free slot")
        length = inputs_embeds.shape[0]
        if length > self.prompt_length:
            raise ValueError(f"prompt of {length} positions; the cache holds {self.prompt_length}")
        slot = len(self._rows)
        cache = DynamicCache(config=self.transformer.config)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.transformer(inputs_embeds=inputs_embeds.unsqueeze(0), past_key_values=cache,
                                   use_cache=True, logits_to_keep=1)
        for layer, (keys, values) in enumerate(zip(self._keys, self._values)):
            keys[slot, :, :length].copy_(cache.layers[layer].keys[0])
            values[slot, :, :length].copy_(cache.layers[layer].values[0])
        del cache

        state, is_branch, bones, skin = self._init_state(start_tokens)
        one = dict(dtype=torch.long, device=self.device)
        row_state = torch.tensor([state], **one)
        row_branch = torch.tensor([is_branch], dtype=torch.bool, device=self.device)
        row_bones = torch.tensor([bones], **one)
        row_skin = torch.tensor([skin], **one)
        zero = torch.zeros(1, **one)
        scores = out.logits[:, -1].float()
        scores, row_state, row_branch, row_bones, row_skin, _, _ = grammar_module.grammar_step(
            scores, row_state, row_state, row_branch, row_bones, row_skin, zero, zero,
            self._transitions, self._masks, advance=False, **self._grammar_kwargs)
        first = scores.argmax(-1)

        self._tok[slot:slot + 1].copy_(first)
        self._pos[slot] = length
        self._state[slot:slot + 1].copy_(row_state)
        self._is_branch[slot:slot + 1].copy_(row_branch)
        self._bones[slot:slot + 1].copy_(row_bones)
        self._skin_count[slot:slot + 1].copy_(row_skin)
        self._switch_len[slot] = 0
        self._position[slot] = 0
        self._out[slot, 0:1].copy_(first)
        self._generated[slot] = 1
        self._done[slot:slot + 1].copy_((first == self.eos) | (self.max_new_tokens <= 1))
        self._write_status(slot)
        self._rows.append(_Row(key, slot))

    def _write_status(self, slot: int) -> None:
        s = slice(slot, slot + 1)
        self._status[_DONE, s].copy_(self._done[s])
        self._status[_GENERATED, s].copy_(self._generated[s])
        self._status[_IN_SKIN, s].copy_(self._state[s] == grammar_module.SKIN)
        self._status[_SWITCH_LEN, s].copy_(self._switch_len[s])
        self._status[_BONES, s].copy_(self._bones[s])

    def remove(self, key: Any) -> bool:
        """Drop a row (a cancelled request's) from its slot; False if absent."""
        for row in self._rows:
            if row.key == key:
                self._free(row)
                return True
        return False

    def _free(self, row: _Row) -> None:
        """Free *row*'s slot, moving the last row into it to keep slots compact."""
        last = self._rows[-1]
        slot = row.slot
        self._rows.remove(row)
        if last is not row:
            self._move(last.slot, slot)
            last.slot = slot
            self._rows.sort(key=lambda r: r.slot)
            vacated = len(self._rows)
        else:
            vacated = slot
        self._done[vacated] = True

    def _move(self, src: int, dst: int) -> None:
        for buffers in (self._keys, self._values):
            for t in buffers:
                t[dst].copy_(t[src])
        for t in (self._tok, self._pos, self._state, self._is_branch, self._bones,
                  self._skin_count, self._switch_len, self._position, self._generated,
                  self._done, self._out):
            t[dst] = t[src]
        self._status[:, dst] = self._status[:, src]

    def progress(self, key: Any) -> Optional[Dict[str, int]]:
        """The latest status read for *key*'s row (approximate, a few steps
        old): ``generated``, ``in_skin``, ``switch_len`` and ``bones``."""
        for row in self._rows:
            if row.key == key and row.status is not None:
                s = row.status
                return {"generated": s[_GENERATED], "in_skin": s[_IN_SKIN],
                        "switch_len": s[_SWITCH_LEN], "bones": s[_BONES]}
        return None

    # ── stepping ─────────────────────────────────────────────────────────────

    def _bucket(self) -> int:
        n = len(self._rows)
        return next(b for b in self.buckets if b >= n)

    def _step(self, b: int, cache: _SlotCache) -> None:
        """One decode step of slots ``0..b-1``, all in place (graph-capturable)."""
        tok, pos = self._tok[:b], self._pos[:b]
        embeds = self.transformer.get_input_embeddings()(tok[:, None])
        visible = (self._cache_slots[None, :] <= pos[:, None])[:, None, None, :]
        out = self.transformer(
            inputs_embeds=embeds, position_ids=pos[:, None], past_key_values=cache,
            attention_mask={t: visible for t in self._layer_types}, use_cache=True)
        scores = out.logits[:, -1].float()
        scores, state, is_branch, bones, skin, switch_len, position = grammar_module.grammar_step(
            scores, tok, self._state[:b], self._is_branch[:b], self._bones[:b],
            self._skin_count[:b], self._switch_len[:b], self._position[:b],
            self._transitions, self._masks, advance=True, **self._grammar_kwargs)
        new = scores.argmax(-1)
        done = self._done[:b]
        active = ~done
        generated = self._generated[:b]
        index = generated.clamp(max=self.max_new_tokens - 1)[:, None]
        out_ids = self._out[:b]
        out_ids.scatter_(1, index, torch.where(active, new, out_ids.gather(1, index)[:, 0])[:, None])
        generated_next = generated + active.long()
        done_next = done | (new == self.eos) | (generated_next >= self.max_new_tokens)
        generated.copy_(generated_next)
        done.copy_(done_next)
        tok.copy_(new)
        pos.copy_((pos + 1).clamp(max=self.length - 1))
        self._state[:b].copy_(state)
        self._is_branch[:b].copy_(is_branch)
        self._bones[:b].copy_(bones)
        self._skin_count[:b].copy_(skin)
        self._switch_len[:b].copy_(switch_len)
        self._position[:b].copy_(position)
        status = self._status[:, :b]
        status[_DONE].copy_(done_next)
        status[_GENERATED].copy_(generated_next)
        status[_IN_SKIN].copy_(state == grammar_module.SKIN)
        status[_SWITCH_LEN].copy_(switch_len)
        status[_BONES].copy_(bones)

    def _cache(self, b: int) -> _SlotCache:
        return _SlotCache([k[:b] for k in self._keys], [v[:b] for v in self._values],
                          self._pos[:b])

    @torch.inference_mode()
    def capture(self) -> None:
        """Capture each bucket's step as a CUDA graph (on this thread, once)."""
        pool = torch.cuda.graph_pool_handle()
        step = self._step
        if self.compile_step:
            # Plain inductor kernels, no CUDA graphs of its own: the graph
            # captured below wraps them. One compilation per bucket.
            step = torch.compile(self._step, backend="inductor", dynamic=False)
        self._done.fill_(True)
        # Autocast wraps the capture (not the other way round), as in generate().
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for b in self.buckets:
                cache = self._caches[b] = self._cache(b)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):  # compile kernels, settle allocations
                        step(b, cache)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    step(b, cache)
                self._graphs[b] = graph
        torch.cuda.synchronize()
        self._reset()

    def _reset(self) -> None:
        self._rows.clear()
        self._snapshot = None
        for t in (self._tok, self._pos, self._generated, self._status):
            t.zero_()
        self._done.fill_(True)

    def step(self) -> List[Finished]:
        """Run one decode step over the occupied slots; returns the rows found
        ended (read a few steps late), whose slots are free again."""
        if not self._rows:
            return []
        self._graphs[self._bucket()].replay()
        self._steps += 1
        return self._read_status(wait=False)

    def _request_snapshot(self) -> None:
        host = self._spare_host
        if host is None:
            host = torch.empty(self._status.shape, dtype=torch.long, pin_memory=True)
        self._spare_host = None
        host.copy_(self._status, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self._snapshot = _Snapshot(host, event, {r.slot: r.key for r in self._rows}, self._steps)

    def _read_status(self, wait: bool) -> List[Finished]:
        snapshot = self._snapshot
        if snapshot is None:
            self._request_snapshot()
            return []
        if not snapshot.event.query():
            if not wait and self._steps - snapshot.step < MAX_STEPS_AHEAD:
                return []
            snapshot.event.synchronize()
        values = snapshot.host.tolist()
        self._spare_host = snapshot.host
        self._snapshot = None
        finished = []
        by_key = {row.key: row for row in self._rows}
        for slot, key in snapshot.rows.items():
            row = by_key.get(key)
            if row is None:
                continue  # removed since
            row.status = [values[i][slot] for i in range(5)]
            if row.status[_DONE]:
                count = row.status[_GENERATED]
                ids = self._out[row.slot, :count].to("cpu")
                finished.append(Finished(key, ids))
        for item in finished:
            self._free(by_key[item.key])
        if self._rows:
            self._request_snapshot()
        return finished
