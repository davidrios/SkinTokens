"""Rig-token grammar as a GPU state machine, plus sync-free decode helpers.

:class:`VocabSwitchingLogitsProcessor` (``tokenrig.py``) enforces the token
grammar on the host: every step it copies the whole sequence to the CPU (a
device sync) and re-scans it in Python. That sync stalls the GPU each token and
the scans grow with the sequence.

:class:`GpuGrammarLogitsProcessor` enforces the same grammar with per-row state
tensors on the device, advanced by each new token through lookup tables, so a
decode step needs no host round-trip. The grammar (``TokenizerPart``):

* skeleton phase, states 1..6 of ``TokenizerPart.next_posible_token``; a bone
  is counted on each ``expect_joint_3`` token that does not follow a branch
  (``bones_in_sequence``);
* the switch token (the tokenizer's eos) starts the skin phase: skin tokens
  until ``bones * tokens_per_skin`` of them, then only the final eos.

:func:`enable_cuda_deferred_stop` lets transformers check for the end of
generation one step late on CUDA too (it already does on MPS), removing its own
per-step sync; generation then runs one extra (discarded) step.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import torch
from transformers import LogitsProcessor

logger = logging.getLogger(__name__)

# Skeleton-phase states (TokenizerPart.next_posible_token).
BOS, CLS_PART_JOINT, PART_JOINT, JOINT_2, JOINT_3, BRANCH_PART_JOINT, JOINT = range(7)
SKIN = 7
# Mask table rows: the 7 skeleton states, SKIN, and FINAL (only the final eos).
FINAL = 8
N_STATES = 8


def gpu_grammar_enabled() -> bool:
    return os.environ.get("SKINTOKENS_GPU_GRAMMAR", "1").strip().lower() not in {"0", "false", "no", "off"}


def supports(tokenizer) -> bool:
    from src.tokenizer.tokenizer_part import TokenizerPart
    return isinstance(tokenizer, TokenizerPart)


def grammar_step(scores, tok, state, is_branch, bones, skin_count, switch_len, position,
                 transitions, masks, *, switch_id: int, branch_id: int, tokens_per_skin: int,
                 advance: bool, temperature: float, top_k: int, top_p: float):
    """One decode step of the grammar, as a pure function (compiled as one graph).

    Advances the per-row state by ``tok`` (when ``advance``), masks ``scores``
    to the tokens the grammar allows next, then applies temperature, top-k and
    top-p exactly as transformers' warpers do (in that order, after the other
    logits processors), so they can be switched off in ``generate``.
    """
    if advance:
        in_skin = state == SKIN
        at_joint_3 = state == JOINT_3
        bones = bones + (at_joint_3 & ~is_branch & ~in_skin).long()
        is_branch = (is_branch & ~at_joint_3) | ((state == BRANCH_PART_JOINT) & (tok == branch_id))
        skin_count = skin_count + in_skin.long()
        switching = ~in_skin & (tok == switch_id)
        position = position + 1
        switch_len = torch.where(switching, position, switch_len)
        state = transitions[state, tok]
    done = skin_count >= bones * tokens_per_skin
    row = torch.where(state == SKIN, torch.where(done, FINAL, SKIN), state)
    out = scores + masks[row]
    if temperature != 1.0:
        out = out / temperature
    if top_k > 0:
        values, indices = torch.topk(out, min(top_k, out.shape[-1]))  # descending
        if top_p < 1.0:
            ascending = values.flip(-1)
            cumulative = ascending.softmax(dim=-1).cumsum(dim=-1)
            remove = cumulative <= (1 - top_p)
            remove[..., -1] = False  # min_tokens_to_keep=1
            values = values.masked_fill(remove.flip(-1), float("-inf"))
        out = torch.full_like(out, float("-inf")).scatter(-1, indices, values)
    return out, state, is_branch, bones, skin_count, switch_len, position


def _compile_enabled() -> bool:
    raw = os.environ.get("SKINTOKENS_GRAMMAR_COMPILE")
    if raw is not None:
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    import importlib.util
    return torch.cuda.is_available() and importlib.util.find_spec("triton") is not None


_compiled_step = None


def _get_step(device: torch.device):
    """The compiled step on CUDA (one fused graph instead of ~40 launches)."""
    global _compiled_step
    if device.type != "cuda" or not _compile_enabled():
        return grammar_step
    if _compiled_step is None:
        _compiled_step = torch.compile(grammar_step, fullgraph=True, dynamic=False)
    return _compiled_step


class GpuGrammarLogitsProcessor(LogitsProcessor):
    """Drop-in for ``VocabSwitchingLogitsProcessor`` with a ``TokenizerPart``.

    ``input_ids`` passed by ``generate`` hold the generated tokens only (the
    prompt, ``init``, went in as embeddings); each call consumes the tokens
    appended since the previous call.
    """

    def __init__(self, tokenizer, switch_token_id, eos_token_id, tokens_per_skin, init):
        self.tokenizer = tokenizer
        self.switch_token_id = int(switch_token_id)
        self.eos_token_id = int(eos_token_id)
        self.tokens_per_skin = int(tokens_per_skin)
        if isinstance(init, torch.Tensor):
            init = init.detach().cpu().numpy()
        self.init_ids = np.asarray(init, dtype=np.int64)
        self._init_state = self._run_init(self.init_ids)
        self._tables_key = None
        self._seen = 0
        self.state = None
        # Warpers folded into the step (see configure_sampling); off by default.
        self._sampling = (1.0, 0, 1.0)
        self._compile_failed = False

    def configure_sampling(self, temperature: float, top_k: int, top_p: float) -> None:
        """Apply temperature/top-k/top-p inside the grammar step; the caller
        then disables transformers' own warpers for this generation."""
        self._sampling = (float(temperature), int(top_k), float(top_p))

    # ── host-side reference, used once for the prompt ───────────────────────

    def _step_host(self, state, is_branch, bones, skin_count, tok):
        """One transition on plain ints (mirrors the device update)."""
        if state == SKIN:
            return SKIN, is_branch, bones, skin_count + 1
        if state == JOINT_3:
            bones += 0 if is_branch else 1
            is_branch = False
        if state == BRANCH_PART_JOINT and tok == self.tokenizer.token_id_branch:
            is_branch = True
        return int(self._transitions_np[state, tok]), is_branch, bones, skin_count

    def _run_init(self, ids):
        self._transitions_np = self._build_transitions(
            max(int(ids.max(initial=0)) + 1, self.eos_token_id + 1))
        if ids.size == 0:
            return BOS, False, 0, 0
        if int(ids[0]) != self.tokenizer.token_id_bos:
            raise ValueError("prompt does not start with bos")
        state, is_branch, bones, skin = CLS_PART_JOINT, False, 0, 0
        for tok in ids[1:]:
            state, is_branch, bones, skin = self._step_host(state, is_branch, bones, skin, int(tok))
        return state, is_branch, bones, skin

    # ── tables ───────────────────────────────────────────────────────────────

    def _token_classes(self, width):
        t = self.tokenizer
        joint = np.zeros(width, bool)
        joint[:min(t.num_discrete, width)] = True
        cls = np.zeros(width, bool)
        for v in [t.token_id_cls_none, *t.cls_token_id.values()]:
            if v < width:
                cls[v] = True
        part = np.zeros(width, bool)
        for v in [t.token_id_spring, *t.parts_token_id.values()]:
            if v < width:
                part[v] = True
        return joint, cls, part

    def _build_transitions(self, width):
        """``[state, token] -> next state`` for the skeleton phase (the switch
        token and the skin phase are handled separately)."""
        joint, cls, _ = self._token_classes(width)
        t = np.zeros((N_STATES, width), np.int64)
        t[BOS] = CLS_PART_JOINT
        t[CLS_PART_JOINT] = np.where(joint, JOINT_2, np.where(cls, PART_JOINT, JOINT))
        t[PART_JOINT] = np.where(joint, JOINT_2, PART_JOINT)
        t[JOINT_2] = JOINT_3
        t[JOINT_3] = BRANCH_PART_JOINT
        t[BRANCH_PART_JOINT] = np.where(joint, JOINT_2, JOINT)  # branch and parts → JOINT
        t[JOINT] = JOINT_2
        t[SKIN] = SKIN
        t[:, self.switch_token_id] = SKIN
        return t

    def _build_masks(self, width):
        """Additive masks, one row per state plus SKIN and FINAL."""
        t = self.tokenizer
        joint, cls, part = self._token_classes(width)
        branch = np.zeros(width, bool)
        branch[t.token_id_branch] = True
        switch = np.zeros(width, bool)
        switch[self.switch_token_id] = True
        bos = np.zeros(width, bool)
        bos[t.token_id_bos] = True
        skin = np.zeros(width, bool)
        skin[t.vocab_size:self.eos_token_id] = True
        final = np.zeros(width, bool)
        final[self.eos_token_id] = True
        allowed = np.stack([
            bos,                               # BOS
            cls | part | joint,                # CLS_PART_JOINT
            part | joint | switch,             # PART_JOINT
            joint,                             # JOINT_2
            joint,                             # JOINT_3
            joint | part | branch | switch,    # BRANCH_PART_JOINT
            joint,                             # JOINT
            skin,                              # SKIN
            final,                             # FINAL
        ])
        return np.where(allowed, 0.0, -np.inf).astype(np.float32)

    def _ensure(self, scores: torch.Tensor) -> None:
        key = (scores.device, scores.dtype, scores.shape[-1], scores.shape[0])
        if key == self._tables_key:
            return
        width = scores.shape[-1]
        device = scores.device
        self._transitions = torch.from_numpy(self._build_transitions(width)).to(device)
        self._masks = torch.from_numpy(self._build_masks(width)).to(device, scores.dtype)
        b = scores.shape[0]
        state, is_branch, bones, skin = self._init_state
        self.state = torch.full((b,), state, dtype=torch.long, device=device)
        self.is_branch = torch.full((b,), is_branch, dtype=torch.bool, device=device)
        self.bones = torch.full((b,), bones, dtype=torch.long, device=device)
        self.skin_count = torch.full((b,), skin, dtype=torch.long, device=device)
        # Generated tokens up to and including the switch token (0 = not yet).
        self.switch_len = torch.zeros((b,), dtype=torch.long, device=device)
        self.position = torch.zeros((), dtype=torch.long, device=device)
        self._branch_id = int(self.tokenizer.token_id_branch)
        self._seen = 0
        self._tables_key = key

    # ── device update ────────────────────────────────────────────────────────

    def _step(self, scores, tok, advance, warp):
        temperature, top_k, top_p = self._sampling if warp else (1.0, 0, 1.0)
        args = (scores, tok, self.state, self.is_branch, self.bones, self.skin_count,
                self.switch_len, self.position, self._transitions, self._masks)
        kwargs = dict(switch_id=self.switch_token_id, branch_id=self._branch_id,
                      tokens_per_skin=self.tokens_per_skin, advance=advance,
                      temperature=temperature, top_k=top_k, top_p=top_p)
        step = grammar_step if self._compile_failed else _get_step(scores.device)
        try:
            result = step(*args, **kwargs)
        except Exception:
            if step is grammar_step:
                raise
            logger.warning("compiled grammar step failed; using the eager one", exc_info=True)
            self._compile_failed = True
            result = grammar_step(*args, **kwargs)
        (out, self.state, self.is_branch, self.bones, self.skin_count,
         self.switch_len, self.position) = result
        return out

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        self._ensure(scores)
        length = input_ids.shape[1]
        if length < self._seen:  # a new generation reusing this processor
            self._tables_key = None
            self._ensure(scores)
        # Normally exactly one new token per call (none on the first). The token
        # is copied out of ``input_ids``: as a view its strides change every step,
        # which would recompile the step each time.
        for position in range(self._seen, length - 1):
            self._step(scores, input_ids[:, position].clone(memory_format=torch.contiguous_format), advance=True, warp=False)
        if length > self._seen:
            out = self._step(scores, input_ids[:, length - 1].clone(memory_format=torch.contiguous_format), advance=True, warp=True)
        else:
            out = self._step(scores, self.state, advance=False, warp=True)
        self._seen = length
        return out

    def progress_state(self) -> Optional[torch.Tensor]:
        """Row 0's ``[in skin phase, generated tokens up to the switch, bones]``
        on the device (for progress estimates), or None before the first step."""
        if self.state is None:
            return None
        return torch.stack([(self.state[0] == SKIN).long(), self.switch_len[0], self.bones[0]])


class AsyncSnapshot:
    """Reads a small device tensor without waiting for the device: each
    request starts a non-blocking copy, and the value becomes readable once
    the copy has finished (checked, never waited for)."""

    def __init__(self) -> None:
        self._host: Optional[torch.Tensor] = None
        self._event = None
        self._pending = False
        self.value: Optional[list] = None

    def request(self, tensor: torch.Tensor) -> None:
        if self._pending:
            return
        if tensor.device.type != "cuda":
            self.value = tensor.tolist()
            return
        if self._host is None or self._host.shape != tensor.shape:
            self._host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
            self._event = torch.cuda.Event()
        self._host.copy_(tensor, non_blocking=True)
        self._event.record()
        self._pending = True

    def poll(self) -> Optional[list]:
        if self._pending and self._event.query():
            self.value = self._host.tolist()
            self._pending = False
        return self.value


_deferred_patched = False


def enable_cuda_deferred_stop() -> bool:
    """Let ``generate`` defer its end-of-generation check on CUDA (one extra,
    discarded step instead of a device sync every step). Returns whether it is on."""
    global _deferred_patched
    if _deferred_patched:
        return True
    if os.environ.get("SKINTOKENS_DEFERRED_STOP", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    try:
        from transformers.generation import utils as gen_utils
        deferred = gen_utils.DeferredStopCheck
    except (ImportError, AttributeError):
        logger.info("transformers has no DeferredStopCheck; stop checks stay synchronous")
        return False
    original = deferred.is_supported

    def is_supported(device, cache, cache_is_returned, is_assistant):
        if device.type == "cuda" and not is_assistant:
            return cache is None or cache.is_croppable or not cache_is_returned
        return original(device, cache, cache_is_returned, is_assistant)

    deferred.is_supported = staticmethod(is_supported)
    _deferred_patched = True
    return True
