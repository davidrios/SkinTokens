"""The continuous-batching decoder (src/model/batch_decode.py) on a tiny LLM.

Covered guarantees:
  * every token a row emits is the grammar-masked argmax of a plain
    (no-cache, unbatched) forward over that row's own prefix, up to bf16
    rounding, however rows join, leave and move between slots;
  * the rows' tokens obey the grammar; a row ends at its final eos or after
    ``max_new_tokens``, and its slot is reused;
  * a removed row frees its slot at once and the others keep decoding.
"""
import random
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")
if not torch.cuda.is_available():
    pytest.skip("the batched decoder captures CUDA graphs", allow_module_level=True)

from src.model import decode_attention
from src.model.batch_decode import BatchDecoder, MAX_STEPS_AHEAD
from src.model.tokenrig import VocabSwitchingLogitsProcessor
from src.tokenizer.tokenizer_part import TokenizerPart

SKIN_CODES = 7
TOKENS_PER_SKIN = 2
PROMPT = 6  # condition positions before [bos, cls]
# Logit gap allowed between the emitted token and the reference argmax: the
# two forwards round differently in bf16 (more so with inductor's fused
# kernels), by a few units in the last place of the logits' size.
TOLERANCE = 0.06
RELATIVE_TOLERANCE = 0.015


def make_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    tok = TokenizerPart.parse(num_discrete=12, continuous_range=(-1.0, 1.0),
                              cls_token_id={"vroid": 0, "mixamo": 1},
                              parts_token_id={"body": 0, "hand": 1, "tail": 2})
    eos = tok.vocab_size + SKIN_CODES
    decode_attention.register()
    config = Qwen3Config(vocab_size=eos + 1, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=32, max_position_embeddings=512, tie_word_embeddings=True,
                         # Large weights, so the context moves the logits by
                         # whole units (the default init leaves them all near 0).
                         initializer_range=0.5)
    config._attn_implementation = decode_attention.DECODE_ATTN_IMPL
    torch.manual_seed(0)
    transformer = Qwen3ForCausalLM(config).to("cuda", torch.bfloat16).eval()
    for module in transformer.modules():
        if getattr(module, "config", None) is not None:
            module.config._attn_implementation = decode_attention.DECODE_ATTN_IMPL
    return SimpleNamespace(tokenizer=tok, transformer=transformer, eos=eos,
                           tokens_per_skin=TOKENS_PER_SKIN, device=torch.device("cuda"))


@pytest.fixture(scope="module")
def model():
    return make_model()


def make_prompt(model, seed):
    g = torch.Generator().manual_seed(seed)
    start = [model.tokenizer.bos, model.tokenizer.token_id_cls_none]
    cond = torch.randn(PROMPT, 64, generator=g).to("cuda", torch.bfloat16)
    embed = model.transformer.get_input_embeddings()(torch.tensor(start, device="cuda"))
    return torch.cat([cond, embed]), start


def check_row(model, embeds, start, generated, max_new_tokens):
    """Every emitted token against a no-cache forward over the row's prefix."""
    tok = model.tokenizer
    generated = generated.tolist()
    assert 1 <= len(generated) <= max_new_tokens
    assert generated[-1] == model.eos or len(generated) == max_new_tokens
    assert model.eos not in generated[:-1]
    host = VocabSwitchingLogitsProcessor(tokenizer=tok, switch_token_id=tok.eos,
                                         eos_token_id=model.eos,
                                         tokens_per_skin=TOKENS_PER_SKIN, init=start)
    emb = model.transformer.get_input_embeddings()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inputs = torch.cat([embeds, emb(torch.tensor(generated[:-1], device="cuda"))])
        logits = model.transformer(inputs_embeds=inputs[None], use_cache=False).logits[0].float()
    for t, token in enumerate(generated):
        scores = logits[embeds.shape[0] - 1 + t][None].cpu()
        masked = host(torch.tensor([generated[:t]], dtype=torch.long), scores.clone())[0]
        assert torch.isfinite(masked[token]), f"token {t} breaks the grammar"
        best = masked.max()
        assert masked[token] >= best - TOLERANCE - RELATIVE_TOLERANCE * best.abs(),             f"token {t} is not the argmax"


def check_cached_row(model, embeds, start, generated, max_new_tokens):
    """Independent HF DynamicCache, with the same single-query attention.

    With these deliberately oversized random weights, SDPA's bf16 attention
    rounding can exceed check_row's tolerance even for one row without moves.
    Explicit masks select the decode backend for this reference as well.
    """
    from transformers import DynamicCache

    ids = generated.tolist()
    assert 1 <= len(ids) <= max_new_tokens
    assert ids[-1] == model.eos or len(ids) == max_new_tokens
    assert model.eos not in ids[:-1]
    host = VocabSwitchingLogitsProcessor(
        tokenizer=model.tokenizer, switch_token_id=model.tokenizer.eos,
        eos_token_id=model.eos, tokens_per_skin=TOKENS_PER_SKIN, init=start)
    cache = DynamicCache(config=model.transformer.config)
    emb = model.transformer.get_input_embeddings()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for t, token in enumerate(ids):
            inputs = embeds[None] if t == 0 else emb(torch.tensor([[ids[t - 1]]], device="cuda"))
            mask = None if t == 0 else {"full_attention": torch.ones(
                1, 1, 1, embeds.shape[0] + t, dtype=torch.bool, device="cuda")}
            logits = model.transformer(inputs_embeds=inputs, past_key_values=cache,
                                       attention_mask=mask, use_cache=True, logits_to_keep=1).logits
            scores = host(torch.tensor([ids[:t]], dtype=torch.long), logits[:, -1].float().cpu())[0]
            assert torch.isfinite(scores[token]), f"token {t} breaks the grammar"
            best = scores.max()
            assert scores[token] >= best - TOLERANCE - RELATIVE_TOLERANCE * best.abs(), \
                f"token {t} is not the independent-cache argmax"


def run_rows(decoder, arrivals, steps_limit=2000, remove_at=None):
    """Feed rows as ``{step: [(key, embeds, start)]}`` and step until all end."""
    finished, inserted, step = {}, {}, 0
    pending = []
    while step < steps_limit:
        pending.extend(arrivals.pop(step, []))
        while pending and decoder.free:
            key, embeds, start = pending.pop(0)
            decoder.insert(key, embeds, start)
            inserted[key] = (embeds, start)
        if remove_at and step == remove_at[0]:
            assert decoder.remove(remove_at[1])
        for item in decoder.step():
            finished[item.key] = item.generated_ids
        step += 1
        if not pending and not arrivals and not decoder.active:
            break
    return finished, inserted


@pytest.mark.parametrize("slots", [1, 2, 3, 4])
def test_rows_joining_and_leaving_decode_like_a_plain_forward(model, slots):
    max_new = 48
    decoder = BatchDecoder(model, slots, max_new, prompt_length=PROMPT + 2)
    decoder.capture()
    rng = random.Random(slots)
    arrivals = {}
    for i in range(6):
        arrivals.setdefault(rng.randrange(0, 60), []).append((i, *make_prompt(model, i)))
    finished, inserted = run_rows(decoder, arrivals)
    assert set(finished) == set(range(6))
    for key, ids in finished.items():
        check_row(model, *inserted[key], ids, max_new)
    assert decoder.active == 0 and decoder.free == slots


@pytest.mark.skipif(not decode_attention._has_triton(), reason="inductor needs Triton")
def test_the_compiled_step_decodes_like_a_plain_forward(model):
    """SKINTOKENS_FAST_DECODE_BACKEND=inductor: the step is compiled, then captured."""
    max_new = 32
    decoder = BatchDecoder(model, 4, max_new, prompt_length=PROMPT + 2, compile_step=True)
    decoder.capture()
    arrivals = {0: [(i, *make_prompt(model, 30 + i)) for i in range(3)],
                7: [(3, *make_prompt(model, 40))]}
    finished, inserted = run_rows(decoder, arrivals)
    assert set(finished) == {0, 1, 2, 3}
    for key, ids in finished.items():
        check_row(model, *inserted[key], ids, max_new)


@pytest.mark.parametrize("slots", [3, 4])
@pytest.mark.parametrize("compile_step", [False, True])
def test_three_rows_replay_batch_three(model, slots, compile_step):
    if compile_step and not decode_attention._has_triton():
        pytest.skip("inductor needs Triton")
    max_new = 32
    decoder = BatchDecoder(model, slots, max_new, prompt_length=PROMPT + 2,
                           compile_step=compile_step)
    decoder.capture()
    replayed = []
    for bucket, graph in list(decoder._graphs.items()):
        def replay(bucket=bucket, graph=graph):
            replayed.append(bucket)
            graph.replay()
        decoder._graphs[bucket] = SimpleNamespace(replay=replay)
    arrivals = {0: [(i, *make_prompt(model, 50 + i)) for i in range(3)]}
    finished, inserted = run_rows(decoder, arrivals)
    assert replayed[0] == 3 and 4 not in replayed
    assert set(finished) == {0, 1, 2}
    for key, ids in finished.items():
        check_row(model, *inserted[key], ids, max_new)


def test_a_removed_row_frees_its_slot_and_the_moved_row_keeps_decoding(model):
    max_new = 40
    decoder = BatchDecoder(model, 4, max_new, prompt_length=PROMPT + 2)
    decoder.capture()
    # Rows 0-3 fill the slots; removing row 1 (slot 1) moves row 3 into it.
    arrivals = {0: [(i, *make_prompt(model, 10 + i)) for i in range(4)],
                5: [(4, *make_prompt(model, 20))]}
    finished, inserted = run_rows(decoder, arrivals, remove_at=(3, 1))
    assert 1 not in finished
    assert set(finished) == {0, 2, 3, 4}
    for key, ids in finished.items():
        check_row(model, *inserted[key], ids, max_new)


def test_insert_needs_a_free_slot_and_a_prompt_that_fits(model):
    decoder = BatchDecoder(model, 1, 8, prompt_length=PROMPT + 2)
    decoder.capture()
    embeds, start = make_prompt(model, 0)
    with pytest.raises(ValueError, match="positive"):
        decoder.insert("empty", embeds[:0], start)
    with pytest.raises(ValueError):
        decoder.insert("long", torch.cat([embeds, embeds]), start)
    decoder.insert("a", embeds, start)
    with pytest.raises(RuntimeError):
        decoder.insert("b", embeds, start)
    assert not decoder.remove("b")
    assert decoder.remove("a") and decoder.free == 1


@pytest.mark.parametrize("slots", [0, 5])
def test_slots_must_be_a_supported_batch_size(model, slots):
    with pytest.raises(ValueError):
        BatchDecoder(model, slots, 8, prompt_length=PROMPT + 2)


@pytest.mark.parametrize("max_new,prompt_length", [(0, 8), (-1, 8), (8, 0), (8, -1)])
def test_lengths_must_be_positive(model, max_new, prompt_length):
    with pytest.raises(ValueError, match="positive"):
        BatchDecoder(model, 1, max_new, prompt_length=prompt_length)


def test_old_snapshot_cannot_finish_a_reinserted_key(model):
    decoder = BatchDecoder(model, 1, 32, prompt_length=PROMPT + 3)
    decoder.capture()
    embeds, start = make_prompt(model, 70)
    # A prompt already in skin phase with zero bones must emit final eos first.
    switch = model.transformer.get_input_embeddings()(torch.tensor([model.tokenizer.eos], device="cuda"))
    decoder.insert("reused", torch.cat([embeds, switch]), start + [model.tokenizer.eos])
    decoder._request_snapshot()
    assert decoder.remove("reused")
    decoder.insert("reused", embeds, start)
    assert decoder._read_status(wait=True) == []
    assert decoder.active == 1
    finished, _ = run_rows(decoder, {})
    check_row(model, embeds, start, finished["reused"], 32)


def test_duplicate_active_keys_are_rejected(model):
    decoder = BatchDecoder(model, 2, 8, prompt_length=PROMPT + 2)
    embeds, start = make_prompt(model, 71)
    decoder.insert("same", embeds, start)
    with pytest.raises(ValueError, match="key"):
        decoder.insert("same", embeds, start)
    assert decoder.active == 1


@pytest.mark.parametrize("compile_step", [False, True])
@pytest.mark.parametrize("prefix_len,max_new", [
    *[(0, limit) for limit in [1, 2, 3, 6, 11, 12, 13, 15, 16, 17, 18]],
    (4, 18), (12, 18), (14, 18), (16, 18),
])
def test_exact_tokens_at_skeleton_branch_skin_and_eos_boundaries(model, compile_step, prefix_len, max_new):
    """Script logits, but run real prefill, KV writes, grammar and CUDA graphs."""
    if compile_step and not decode_attention._has_triton():
        pytest.skip("inductor needs Triton")
    tok = model.tokenizer
    sequence = [0, 1, 2, tok.token_id_branch, 3, 4, 5,
                tok.parts_token_id["body"], 6, 7, 8, tok.eos,
                tok.vocab_size, tok.vocab_size + 1, tok.vocab_size + 2,
                tok.vocab_size + 3, model.eos]
    table = torch.zeros(128, model.eos + 1, device="cuda")
    table[:, model.eos] = 100
    for i, token in enumerate(sequence):
        table[PROMPT + 1 + i].zero_()
        table[PROMPT + 1 + i, token] = 100

    def scripted_logits(module, args, kwargs, output):
        positions = kwargs.get("position_ids")
        if positions is None:
            positions = torch.tensor([[kwargs["inputs_embeds"].shape[1] - 1]], device="cuda")
        output.logits = table[positions]
        return output

    hook = model.transformer.register_forward_hook(scripted_logits, with_kwargs=True)
    try:
        decoder = BatchDecoder(model, 4, max_new, prompt_length=PROMPT + 2 + prefix_len,
                               compile_step=compile_step)
        decoder.capture()
        def prompt(i):
            embeds, start = make_prompt(model, 80 + i)
            prefix = sequence[:prefix_len]
            extra = model.transformer.get_input_embeddings()(torch.tensor(prefix, device="cuda", dtype=torch.long))
            return i, torch.cat([embeds, extra]), start + prefix
        arrivals = {0: [prompt(i) for i in range(4)], 2: [prompt(4)]}
        finished, _ = run_rows(decoder, arrivals)
        assert set(finished) == set(range(5))
        for ids in finished.values():
            assert ids.tolist() == sequence[prefix_len:prefix_len + max_new]
    finally:
        hook.remove()


@pytest.mark.parametrize("compile_step", [False, True])
def test_different_prompt_lengths_and_repeated_compaction(model, compile_step):
    if compile_step and not decode_attention._has_triton():
        pytest.skip("inductor needs Triton")
    decoder = BatchDecoder(model, 4, 48, prompt_length=PROMPT + 2,
                           compile_step=compile_step)
    decoder.capture()
    prompts = {i: make_prompt(model, 100 + i) for i in range(12)}
    prompts = {i: (embeds[i % PROMPT:], start) for i, (embeds, start) in prompts.items()}
    for i in range(4):
        decoder.insert(i, *prompts[i])
    decoder.step()  # keep a snapshot from before both moves and slot reuse
    assert decoder.remove(0) and decoder.remove(1)
    arrivals = {0: [(i, *prompts[i]) for i in range(4, 12)]}
    finished, _ = run_rows(decoder, arrivals)
    assert set(finished) == set(range(2, 12))
    for key, ids in finished.items():
        check_cached_row(model, *prompts[key], ids, 48)


def test_finished_rows_survive_many_extra_replays_and_finish_together(model):
    decoder = BatchDecoder(model, 4, 1, prompt_length=PROMPT + 2)
    decoder.capture()
    for i in range(4):
        decoder.insert(i, *make_prompt(model, 120 + i))
    expected = decoder._out.cpu().clone()
    for _ in range(32):
        decoder._graphs[4].replay()
    decoder._request_snapshot()
    finished = decoder._read_status(wait=True)
    assert {item.key for item in finished} == set(range(4))
    for item in finished:
        assert torch.equal(item.generated_ids, expected[item.key])
    assert decoder.active == 0 and decoder.step() == []


def test_host_waits_for_a_delayed_snapshot_at_the_steps_ahead_limit(model):
    decoder = BatchDecoder(model, 1, 1, prompt_length=PROMPT + 2)
    decoder.capture()
    decoder.insert("short", *make_prompt(model, 130))
    assert decoder.step() == []
    snapshot = decoder._snapshot
    event, waited = snapshot.event, []
    snapshot.event = SimpleNamespace(query=lambda: False,
                                     synchronize=lambda: (waited.append(True), event.synchronize()))
    for _ in range(MAX_STEPS_AHEAD - 1):
        assert decoder.step() == []
    assert not waited
    finished = decoder.step()
    assert waited == [True] and [item.key for item in finished] == ["short"]
    assert len(finished[0].generated_ids) == 1 and decoder.active == 0
