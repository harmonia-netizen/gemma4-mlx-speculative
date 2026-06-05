import argparse
import time
import importlib

import mlx.core as mx
import template_draft_engine as engine
from template_draft_runtime import TemplateDraftRuntime

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
- repo=local-speculative-runtime
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
    
    runtime = TemplateDraftRuntime(target_model, tokenizer, processor, args.max_kv_size)

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
        
        # A. baseline_chunked_greedy
        baseline_results = []
        print("--- A. Baseline Chunked Greedy ---")
        for i, t in enumerate(tokenized_prompts):
            print(f"  case {i} ({SUFFIX_CASES[i]['name']}):", end=" ", flush=True)
            res = runtime.baseline_chunked_greedy(prefix_ids, suffix_ids_list[i], args.max_tokens)
            baseline_results.append(res)
            print(f"elapsed={res.elapsed_sec:.3f}s prefill={res.prefill_sec:.3f}s decode={res.decode_sec:.3f}s")
            mx.clear_cache()
            
        # B. prefix_reuse_greedy
        reuse_greedy_results = []
        print("--- B. Prefix Reuse Greedy ---")
        for i, suffix_ids in enumerate(suffix_ids_list):
            print(f"  case {i} ({SUFFIX_CASES[i]['name']}):", end=" ", flush=True)
            res = runtime.prefix_reuse_greedy(prefix_str, prefix_ids, suffix_ids, args.max_tokens)
            reuse_greedy_results.append(res)
            print(f"elapsed(exc_prefix)={res.elapsed_sec:.3f}s suffix_prefill={res.prefill_sec:.3f}s decode={res.decode_sec:.3f}s")
            
            if res.token_ids != baseline_results[i].token_ids:
                print("MISMATCH in B")
                raise SystemExit(2)
            mx.clear_cache()

        # C. prefix_reuse_template_draft
        reuse_draft_results = []
        print("--- C. Prefix Reuse Template Draft ---")
        for i, suffix_ids in enumerate(suffix_ids_list):
            print(f"  case {i} ({SUFFIX_CASES[i]['name']}):", end=" ", flush=True)
            res = runtime.prefix_reuse_template_draft(
                prefix_str,
                prefix_ids,
                suffix_ids,
                user_prompts[i],
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
        
        # We need to manually add prefix prefill sec for B and C for amortized calculation
        # Prefix cache was created during the first call in B.
        prefix_entry = runtime.prefix_manager.get_or_create(prefix_str, prefix_ids, target_model, args.max_kv_size)
        prefix_prefill_sec = prefix_entry.prefill_sec
        
        total_A_elapsed = sum(r.elapsed_sec for r in baseline_results)
        total_B_elapsed = prefix_prefill_sec + sum(r.elapsed_sec for r in reuse_greedy_results)
        total_C_elapsed = prefix_prefill_sec + sum(r.elapsed_sec for r in reuse_draft_results)
        
        print(f"A baseline total elapsed: {total_A_elapsed:.3f}s")
        print(f"B amortized total elapsed: {total_B_elapsed:.3f}s")
        print(f"C amortized total elapsed: {total_C_elapsed:.3f}s")
        
        print(f"B vs A amortized elapsed speedup: {total_A_elapsed / total_B_elapsed:.3f}x")
        print(f"C vs A amortized elapsed speedup: {total_A_elapsed / total_C_elapsed:.3f}x")
        print(f"C vs B amortized elapsed speedup: {total_B_elapsed / total_C_elapsed:.3f}x")
        
        exact_idx = 0
        decode_speedup_exact = reuse_greedy_results[exact_idx].decode_sec / reuse_draft_results[exact_idx].decode_sec if reuse_draft_results[exact_idx].decode_sec > 0 else 0
        print(f"C vs B decode speedup for exact_pytest_plan: {decode_speedup_exact:.3f}x")

    print("\nOK: runtime benchmark completed with token match")

if __name__ == "__main__":
    main()
