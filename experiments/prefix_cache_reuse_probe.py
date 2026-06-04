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
        lines.append(f"[INFO] module=agent step={i:06d} status=ok message=\"synthetic long context line for prefix cache benchmark\"")
    return "\n".join(lines)

SUFFIXES = [
    """次の確認手順をbashブロックだけで出してください。
前提:
- pytestの失敗内容を短く確認したい
- gitの差分も確認したい
- 出力は次の3行だけ
- 説明文は不要
- コマンドはこの順番にする:
  1. git status --short
  2. pytest --tb=short
  3. git diff
""",
    """次に実行すべき確認コマンドを1つだけ出してください。
出力は `pytest --tb=short` だけ。
説明文は不要。
""",
    """作業ツリーの状態を確認するため、次に実行すべきコマンドを1つだけ出してください。
説明文は不要。
"""
]

@dataclass
class Result:
    text: str
    token_ids: list[int]
    elapsed_sec: float
    prefill_sec: float
    decode_sec: float

def run_baseline_greedy(target_model, tokenizer, input_ids, stop_ids, max_tokens, max_kv_size):
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
    parser.add_argument("--max-tokens", type=int, default=64)
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
    
    formatted_prompts = [engine.format_prompt(processor, prefix_str + "\n" + s) for s in SUFFIXES]
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
        print(f"case {i} suffix_tokens: {len(s)}")

    for run_idx in range(args.runs):
        print(f"\n========== run {run_idx+1}/{args.runs} ==========")
        
        baseline_results = []
        print("--- Baseline Full Prompt ---")
        for i, t in enumerate(tokenized_prompts):
            print(f"  case {i}:", end=" ", flush=True)
            res = run_baseline_greedy(target_model, tokenizer, t, stop_ids, args.max_tokens, args.max_kv_size)
            baseline_results.append(res)
            print(f"prefill={res.prefill_sec:.3f}s decode={res.decode_sec:.3f}s elapsed={res.elapsed_sec:.3f}s")
            mx.metal.clear_cache()
            
        reuse_results = []
        print("--- Prefix Cache Reuse ---")
        prefix_cache, prefix_prefill_sec = prefill_prefix(target_model, prefix_ids, args.max_kv_size)
        print(f"  prefix_prefill_sec: {prefix_prefill_sec:.3f}s")
        
        snapshot = engine.full_snapshot(prefix_cache)
        
        for i, suffix_ids in enumerate(suffix_ids_list):
            print(f"  case {i}:", end=" ", flush=True)
            engine.restore_full(prefix_cache, snapshot)
            
            res = run_reuse_greedy(target_model, tokenizer, suffix_ids, prefix_cache, stop_ids, args.max_tokens)
            reuse_results.append(res)
            print(f"suffix_prefill={res.prefill_sec:.3f}s decode={res.decode_sec:.3f}s elapsed={res.elapsed_sec:.3f}s")
            
            if res.token_ids != baseline_results[i].token_ids:
                print("MISMATCH")
                print("baseline ids:", baseline_results[i].token_ids)
                print("reuse ids:", res.token_ids)
                raise SystemExit(2)
            mx.metal.clear_cache()

        print("\n--- Speedups ---")
        total_baseline_elapsed = sum(r.elapsed_sec for r in baseline_results)
        
        total_reuse_elapsed_excluding_prefix = sum(r.elapsed_sec for r in reuse_results)
        total_reuse_elapsed_including_prefix = prefix_prefill_sec + total_reuse_elapsed_excluding_prefix
        
        print(f"baseline total elapsed: {total_baseline_elapsed:.3f}s")
        print(f"reuse amortized total elapsed: {total_reuse_elapsed_including_prefix:.3f}s")
        
        amortized_speedup = total_baseline_elapsed / total_reuse_elapsed_including_prefix
        print(f"amortized elapsed speedup: {amortized_speedup:.3f}x")
        
        for i in range(len(SUFFIXES)):
            speedup = baseline_results[i].elapsed_sec / reuse_results[i].elapsed_sec
            print(f"  case {i} elapsed speedup (excluding shared prefix): {speedup:.3f}x")

    print("\nOK: prefix cache reuse probe completed with token match")

if __name__ == "__main__":
    main()
