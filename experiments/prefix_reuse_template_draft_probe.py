import argparse
import time
from dataclasses import dataclass
import importlib

import mlx.core as mx
import template_draft_engine as engine

d = importlib.import_module("mlx_vlm.generate.dispatch")


def generate_prefix(repeat_lines: int) -> str:
    lines = []
    for i in range(repeat_lines):
        lines.append(f"[INFO] module=agent step={i:06d} status=ok message=\"synthetic long context line for prefix reuse template draft benchmark\"")
    return "\n".join(lines)


SUFFIX_CASES = [
    {
        "name": "exact_pytest_plan",
        "text": """次の確認手順をbashブロックだけで出してください。
前提:
- pytestの失敗内容を短く確認したい
- gitの差分も確認したい
- 出力は次の3行だけ
- 説明文は不要
- コマンドはこの順番にする:
  1. git status --short
  2. pytest --tb=short
  3. git diff
"""
    },
    {
        "name": "medium_pytest_plan",
        "text": """pytest失敗の原因を安全に確認するため、次に実行する確認手順を3行のbashブロックで出してください。
前提:
- repo=gemma4-mlx-speculative
- destructive commandは禁止
- 3コマンドだけ出す
- 説明文は不要
"""
    },
    {
        "name": "git_status",
        "text": """作業ツリーの状態を確認するため、次に実行すべきコマンドを1つだけ出してください。
説明文は不要。
"""
    }
]


@dataclass
class Result:
    text: str
    token_ids: list[int]
    elapsed_sec: float
    prefill_sec: float
    decode_sec: float
    accepted: int = 0
    drafted: int = 0
    rejected: int = 0


def run_baseline_greedy(target_model, tokenizer, input_ids, stop_ids, max_tokens, max_kv_size, prefix_len):
    lm = engine.get_lm(target_model)
    total_start = time.perf_counter()

    prompt_cache = engine.make_cache(lm, max_kv_size)

    input_arr = mx.array([input_ids])
    emb = target_model.get_input_embeddings(input_arr, None, mask=None)
    inputs_embeds = emb.inputs_embeds
    extra = {
        k: v for k, v in emb.to_dict().items() if k != "inputs_embeds" and v is not None
    }

    if input_arr.shape[1] > 1:
        p = prefix_len
        if p > 0:
            lm(
                input_arr[:, :p],
                inputs_embeds=inputs_embeds[:, :p],
                cache=prompt_cache,
                n_to_process=p,
                **extra,
            )
            engine.eval_cache(prompt_cache)
            
        rem = input_arr.shape[1] - 1
        if rem > p:
            lm(
                input_arr[:, p:rem],
                inputs_embeds=inputs_embeds[:, p:rem],
                cache=prompt_cache,
                n_to_process=rem - p,
                **extra,
            )
            engine.eval_cache(prompt_cache)

    cur = input_arr[:, -1:]
    prefill_sec = time.perf_counter() - total_start

    decode_start = time.perf_counter()
    out = []

    first_logits = engine.forward_one(lm, cur, prompt_cache)
    tok = engine.argmax_token(first_logits)
    
    first_id = int(tok.item())
    if first_id not in stop_ids:
        out.append(first_id)

    next_logits = engine.forward_one(lm, tok[:, None], prompt_cache)

    while len(out) < max_tokens:
        tok = engine.argmax_token(next_logits)
        tid = int(tok.item())

        if tid in stop_ids:
            break

        out.append(tid)
        next_logits = engine.forward_one(lm, tok[:, None], prompt_cache)

    decode_sec = time.perf_counter() - decode_start

    return Result(
        engine.decode_text(tokenizer, out),
        out,
        time.perf_counter() - total_start,
        prefill_sec,
        decode_sec,
    )


def run_reuse_greedy(target_model, tokenizer, suffix_ids, prompt_cache, stop_ids, max_tokens):
    lm = engine.get_lm(target_model)
    total_start = time.perf_counter()

    if len(suffix_ids) > 0:
        input_arr = mx.array([suffix_ids])
        emb = target_model.get_input_embeddings(input_arr, None, mask=None)
        inputs_embeds = emb.inputs_embeds
        extra = {
            k: v for k, v in emb.to_dict().items() if k != "inputs_embeds" and v is not None
        }

        if input_arr.shape[1] > 1:
            n = input_arr.shape[1] - 1
            lm(
                input_arr[:, :n],
                inputs_embeds=inputs_embeds[:, :n],
                cache=prompt_cache,
                n_to_process=n,
                **extra,
            )
            engine.eval_cache(prompt_cache)

        cur = input_arr[:, -1:]
    else:
        raise RuntimeError("suffix_ids must not be empty")

    prefill_sec = time.perf_counter() - total_start

    decode_start = time.perf_counter()
    out = []

    first_logits = engine.forward_one(lm, cur, prompt_cache)
    tok = engine.argmax_token(first_logits)
    
    first_id = int(tok.item())
    if first_id not in stop_ids:
        out.append(first_id)

    next_logits = engine.forward_one(lm, tok[:, None], prompt_cache)

    while len(out) < max_tokens:
        tok = engine.argmax_token(next_logits)
        tid = int(tok.item())

        if tid in stop_ids:
            break

        out.append(tid)
        next_logits = engine.forward_one(lm, tok[:, None], prompt_cache)

    decode_sec = time.perf_counter() - decode_start

    return Result(
        engine.decode_text(tokenizer, out),
        out,
        time.perf_counter() - total_start,
        prefill_sec,
        decode_sec,
    )


def run_reuse_template_draft(
    target_model,
    tokenizer,
    suffix_ids,
    prompt_cache,
    user_prompt,
    stop_ids,
    max_tokens,
    block_size,
    template_min_tokens,
    trace_template=False,
):
    target_lm = engine.get_lm(target_model)
    total_start = time.perf_counter()

    if len(suffix_ids) > 0:
        input_arr = mx.array([suffix_ids])
        emb = target_model.get_input_embeddings(input_arr, None, mask=None)
        inputs_embeds = emb.inputs_embeds
        extra = {
            k: v for k, v in emb.to_dict().items() if k != "inputs_embeds" and v is not None
        }

        if input_arr.shape[1] > 1:
            n = input_arr.shape[1] - 1
            target_lm(
                input_arr[:, :n],
                inputs_embeds=inputs_embeds[:, :n],
                cache=prompt_cache,
                n_to_process=n,
                **extra,
            )
            engine.eval_cache(prompt_cache)

        cur = input_arr[:, -1:]
    else:
        raise RuntimeError("suffix_ids must not be empty")

    prefill_sec = time.perf_counter() - total_start

    decode_start = time.perf_counter()

    first_logits = engine.forward_one(target_lm, cur, prompt_cache)
    tok = engine.argmax_token(first_logits)
    first_id = int(tok.item())

    out = []
    
    if first_id in stop_ids:
        decode_sec = time.perf_counter() - decode_start
        return Result(
            "",
            [],
            time.perf_counter() - total_start,
            prefill_sec,
            decode_sec,
        )

    out.append(first_id)
    target_next_logits = engine.forward_one(target_lm, tok[:, None], prompt_cache)

    accepted = 0
    drafted = 0
    rejected = 0

    candidates = engine.draft_candidates(user_prompt)
    candidate = engine.select_candidate(user_prompt, candidates, tokenizer, template_min_tokens, trace_template)
    candidate_ids = engine.encode_candidate(tokenizer, candidate) if candidate else []

    if candidate_ids and candidate_ids[0] == first_id:
        candidate_ids = candidate_ids[1:]

    if len(candidate_ids) < template_min_tokens:
        candidate_ids = []

    cursor = 0
    template_disabled = False

    while len(out) < max_tokens:
        remaining = max_tokens - len(out)
        if remaining <= 0:
            break

        proposal_ids = []
        if not template_disabled and cursor < len(candidate_ids):
            proposal_ids = candidate_ids[cursor : cursor + min(block_size, remaining)]

        if not proposal_ids:
            tok = engine.argmax_token(target_next_logits)
            tid = int(tok.item())

            if tid in stop_ids:
                break

            out.append(tid)
            target_next_logits = engine.forward_one(target_lm, tok[:, None], prompt_cache)
            continue

        drafted += len(proposal_ids)

        if any(p in stop_ids for p in proposal_ids):
            if trace_template:
                print("trace: reject due to stop token in proposal")
            template_disabled = True
            candidate_ids = []
            cursor = 0
            continue

        target_tok = engine.argmax_token(target_next_logits)
        target_id = int(target_tok.item())

        if target_id != proposal_ids[0]:
            if trace_template:
                print(f"trace: reject at block start. target={target_id}, proposed={proposal_ids[0]}")
            rejected += 1
            template_disabled = True
            candidate_ids = []
            cursor = 0
            continue

        snap = engine.full_snapshot(prompt_cache)

        verify_logits = engine.forward_many(
            target_lm,
            mx.array([proposal_ids], dtype=mx.int32),
            prompt_cache,
        )

        block_matches = True
        for i in range(1, len(proposal_ids)):
            target_tok = engine.argmax_token(verify_logits[:, i - 1, :])
            target_id = int(target_tok.item())
            if target_id != proposal_ids[i]:
                if trace_template:
                    print(f"trace: reject inside block at {i}. target={target_id}, proposed={proposal_ids[i]}")
                block_matches = False
                break

        if not block_matches:
            engine.restore_full(prompt_cache, snap)
            rejected += 1
            template_disabled = True
            candidate_ids = []
            cursor = 0
            continue

        out.extend(proposal_ids)
        accepted += len(proposal_ids)
        cursor += len(proposal_ids)
        target_next_logits = verify_logits[:, -1, :]

    decode_sec = time.perf_counter() - decode_start

    return Result(
        engine.decode_text(tokenizer, out),
        out,
        time.perf_counter() - total_start,
        prefill_sec,
        decode_sec,
        accepted,
        drafted,
        rejected,
    )


def prefill_prefix(target_model, prefix_ids, max_kv_size):
    lm = engine.get_lm(target_model)
    start = time.perf_counter()

    prompt_cache = engine.make_cache(lm, max_kv_size)
    input_arr = mx.array([prefix_ids])
    emb = target_model.get_input_embeddings(input_arr, None, mask=None)
    inputs_embeds = emb.inputs_embeds
    extra = {
        k: v for k, v in emb.to_dict().items() if k != "inputs_embeds" and v is not None
    }

    lm(
        input_arr,
        inputs_embeds=inputs_embeds,
        cache=prompt_cache,
        n_to_process=input_arr.shape[1],
        **extra,
    )
    engine.eval_cache(prompt_cache)

    prefill_sec = time.perf_counter() - start
    return prompt_cache, prefill_sec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--repeat-lines", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--draft-block-size", type=int, default=8)
    parser.add_argument("--template-min-tokens", type=int, default=1)
    parser.add_argument("--max-kv-size", type=int, default=None)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    model_path = args.model or engine.DEFAULT_TARGET_MODEL_PATH
    print("loading target:", model_path)
    target_model, processor = d.load(model_path)
    tokenizer = getattr(processor, "tokenizer", processor)
    stop_ids = engine.build_stop_ids(tokenizer)

    prefix_str = generate_prefix(args.repeat_lines)
    
    formatted_prompts = [engine.format_prompt(processor, prefix_str + "\n" + c["text"]) for c in SUFFIX_CASES]
    user_prompts = [prefix_str + "\n" + c["text"] for c in SUFFIX_CASES]
    tokenized_prompts = [tokenizer.encode(p) for p in formatted_prompts]

    common_prefix_ids = tokenized_prompts[0]
    for t in tokenized_prompts[1:]:
        idx = 0
        while idx < len(common_prefix_ids) and idx < len(t) and common_prefix_ids[idx] == t[idx]:
            idx += 1
        common_prefix_ids = common_prefix_ids[:idx]

    prefix_ids = common_prefix_ids
    suffix_ids_list = [t[len(common_prefix_ids):] for t in tokenized_prompts]

    print(f"repeat_lines: {args.repeat_lines}")
    print(f"prefix_tokens: {len(prefix_ids)}")
    for i, s in enumerate(suffix_ids_list):
        print(f"case {i} ({SUFFIX_CASES[i]['name']}) suffix_tokens: {len(s)}")

    for run_idx in range(args.runs):
        print(f"\n========== run {run_idx+1}/{args.runs} ==========")
        
        # A. baseline_full_greedy
        baseline_results = []
        print("--- A. Baseline Full Greedy ---")
        for i, t in enumerate(tokenized_prompts):
            print(f"  case {i} ({SUFFIX_CASES[i]['name']}):", end=" ", flush=True)
            res = run_baseline_greedy(target_model, tokenizer, t, stop_ids, args.max_tokens, args.max_kv_size, len(prefix_ids))
            baseline_results.append(res)
            print(f"elapsed={res.elapsed_sec:.3f}s prefill={res.prefill_sec:.3f}s decode={res.decode_sec:.3f}s")
            mx.clear_cache()
            
        # B. prefix_reuse_greedy
        reuse_greedy_results = []
        print("--- B. Prefix Reuse Greedy ---")
        prefix_cache_B, prefix_prefill_sec_B = prefill_prefix(target_model, prefix_ids, args.max_kv_size)
        print(f"  shared_prefix_prefill_sec: {prefix_prefill_sec_B:.3f}s")
        
        snapshot_B = engine.full_snapshot(prefix_cache_B)
        
        for i, suffix_ids in enumerate(suffix_ids_list):
            print(f"  case {i} ({SUFFIX_CASES[i]['name']}):", end=" ", flush=True)
            engine.restore_full(prefix_cache_B, snapshot_B)
            
            res = run_reuse_greedy(target_model, tokenizer, suffix_ids, prefix_cache_B, stop_ids, args.max_tokens)
            reuse_greedy_results.append(res)
            print(f"elapsed(exc_prefix)={res.elapsed_sec:.3f}s suffix_prefill={res.prefill_sec:.3f}s decode={res.decode_sec:.3f}s")
            
            if res.token_ids != baseline_results[i].token_ids:
                print("MISMATCH in B")
                print("Baseline text:", repr(baseline_results[i].text))
                print("Reuse    text:", repr(res.text))
                print("Baseline ids:", baseline_results[i].token_ids)
                print("Reuse    ids:", res.token_ids)
                raise SystemExit(2)
            mx.clear_cache()

        # C. prefix_reuse_template_draft
        reuse_draft_results = []
        print("--- C. Prefix Reuse Template Draft ---")
        prefix_cache_C, prefix_prefill_sec_C = prefill_prefix(target_model, prefix_ids, args.max_kv_size)
        print(f"  shared_prefix_prefill_sec: {prefix_prefill_sec_C:.3f}s")
        
        snapshot_C = engine.full_snapshot(prefix_cache_C)
        
        for i, suffix_ids in enumerate(suffix_ids_list):
            print(f"  case {i} ({SUFFIX_CASES[i]['name']}):", end=" ", flush=True)
            engine.restore_full(prefix_cache_C, snapshot_C)
            
            res = run_reuse_template_draft(
                target_model, 
                tokenizer, 
                suffix_ids, 
                prefix_cache_C, 
                user_prompts[i],
                stop_ids, 
                args.max_tokens,
                args.draft_block_size,
                args.template_min_tokens,
                args.trace
            )
            reuse_draft_results.append(res)
            print(f"elapsed(exc_prefix)={res.elapsed_sec:.3f}s suffix_prefill={res.prefill_sec:.3f}s decode={res.decode_sec:.3f}s accept={res.accepted}/{res.drafted}")
            
            if res.token_ids != baseline_results[i].token_ids:
                print("MISMATCH in C")
                raise SystemExit(2)
            mx.clear_cache()

        print("\n--- Speedups ---")
        
        total_A_elapsed = sum(r.elapsed_sec for r in baseline_results)
        total_B_elapsed = prefix_prefill_sec_B + sum(r.elapsed_sec for r in reuse_greedy_results)
        total_C_elapsed = prefix_prefill_sec_C + sum(r.elapsed_sec for r in reuse_draft_results)
        
        print(f"A baseline total elapsed: {total_A_elapsed:.3f}s")
        print(f"B amortized total elapsed: {total_B_elapsed:.3f}s")
        print(f"C amortized total elapsed: {total_C_elapsed:.3f}s")
        
        print(f"B vs A amortized elapsed speedup: {total_A_elapsed / total_B_elapsed:.3f}x")
        print(f"C vs A amortized elapsed speedup: {total_A_elapsed / total_C_elapsed:.3f}x")
        print(f"C vs B amortized elapsed speedup: {total_B_elapsed / total_C_elapsed:.3f}x")
        
        # specific exact_pytest_plan speedup
        exact_idx = 0
        decode_speedup_exact = reuse_greedy_results[exact_idx].decode_sec / reuse_draft_results[exact_idx].decode_sec if reuse_draft_results[exact_idx].decode_sec > 0 else 0
        print(f"C vs B decode speedup for exact_pytest_plan: {decode_speedup_exact:.3f}x")

    print("\nOK: prefix reuse template draft probe completed with token match")

if __name__ == "__main__":
    main()
