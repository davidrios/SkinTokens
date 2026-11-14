"""The GPU grammar processor matches the host one token for token."""
import random

import pytest
import torch

pytest.importorskip("transformers")

from src.model.grammar import GpuGrammarLogitsProcessor
from src.model.tokenrig import SamplingProgressCriteria, VocabSwitchingLogitsProcessor, get_logits_processor
from src.tokenizer.tokenizer_part import TokenizerPart

SKIN_CODES = 7
TOKENS_PER_SKIN = 2


def make_tokenizer():
    return TokenizerPart.parse(num_discrete=12, continuous_range=(-1.0, 1.0),
                               cls_token_id={"vroid": 0, "mixamo": 1},
                               parts_token_id={"body": 0, "hand": 1, "tail": 2})


def make_pair(tok, prompt):
    eos = tok.vocab_size + SKIN_CODES  # the model's final eos (last id)
    args = dict(tokenizer=tok, switch_token_id=tok.eos, eos_token_id=eos,
                tokens_per_skin=TOKENS_PER_SKIN, init=torch.tensor(prompt))
    return VocabSwitchingLogitsProcessor(**args), GpuGrammarLogitsProcessor(**args), eos


def test_gpu_grammar_is_selected(monkeypatch):
    monkeypatch.setenv("SKINTOKENS_GPU_GRAMMAR", "1")
    tok = make_tokenizer()
    processor = get_logits_processor(tok, tok.vocab_size + SKIN_CODES, TOKENS_PER_SKIN,
                                     torch.tensor([tok.bos, tok.token_id_cls_none]))[0]
    assert isinstance(processor, GpuGrammarLogitsProcessor)


def walk(seed, prompt, batch, device="cpu", max_steps=160):
    tok = make_tokenizer()
    host, gpu, eos = make_pair(tok, prompt)
    width = eos + 1 + 3  # the model's logits may be wider than its vocab
    rng = random.Random(seed)
    ids = torch.zeros((batch, 0), dtype=torch.long, device=device)
    finished = [False] * batch
    for step in range(max_steps):
        scores = torch.zeros((batch, width), device=device)
        expected = host(ids.cpu(), scores.cpu())
        got = gpu(ids, scores).cpu()
        assert torch.equal(torch.isfinite(expected), torch.isfinite(got)), f"step {step}"
        next_ids = []
        for row in range(batch):
            if finished[row]:
                next_ids.append(tok.pad)  # what generate appends once a row ended
                continue
            allowed = torch.nonzero(torch.isfinite(expected[row])).flatten().tolist()
            # Bias towards ending the skeleton so walks reach the skin phase.
            if tok.eos in allowed and step > 25 and rng.random() < 0.3:
                choice = tok.eos
            else:
                choice = rng.choice(allowed)
            finished[row] = choice == eos
            next_ids.append(choice)
        ids = torch.cat([ids, torch.tensor(next_ids, device=device)[:, None]], dim=1)
        if all(finished):
            break
    return ids, gpu


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("prompt_kind", ["cls", "cls_none", "bos_only"])
def test_gpu_grammar_matches_host(seed, prompt_kind):
    tok = make_tokenizer()
    prompt = {"cls": [tok.bos, tok.cls_token_id["mixamo"]],
              "cls_none": [tok.bos, tok.token_id_cls_none],
              "bos_only": [tok.bos]}[prompt_kind]
    walk(seed, prompt, batch=3)


def test_walks_reach_skin_phase_and_final_eos():
    tok = make_tokenizer()
    eos = tok.vocab_size + SKIN_CODES
    ended = sum(int((walk(seed, [tok.bos, tok.token_id_cls_none], batch=1)[0] == eos).any())
                for seed in range(20))
    assert ended >= 10


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_grammar_on_cuda():
    tok = make_tokenizer()
    walk(3, [tok.bos, tok.token_id_cls_none], batch=4, device="cuda")


def test_progress_estimate_matches_host_estimate():
    tok = make_tokenizer()
    prompt = [tok.bos, tok.token_id_cls_none]
    ids, gpu = walk(11, prompt, batch=1)
    eos = tok.vocab_size + SKIN_CODES
    kwargs = dict(progress_callback=lambda p, s: None, tokenizer=tok, final_eos_token_id=eos,
                  tokens_per_skin=TOKENS_PER_SKIN, start_tokens=torch.tensor(prompt))
    host_crit = SamplingProgressCriteria(**kwargs)
    gpu_crit = SamplingProgressCriteria(**kwargs, grammar=gpu)
    gpu_crit._snapshot.request(gpu.progress_state())
    assert gpu_crit._estimate_from_grammar() == host_crit._estimated_generated_total(ids[0])
