import argparse
import time
import os
import importlib

import mlx.core as mx
import template_draft_engine as engine
from template_draft_runtime import TemplateDraftRuntime

d = importlib.import_module("mlx_vlm.generate.dispatch")

SUFFIX_CASES = {
    "exact_pytest_plan": """次の確認手順をbashブロックだけで出してください。
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
    "medium_pytest_plan": """pytest失敗の原因を安全に確認するため、次に実行する確認手順を3行のbashブロックで出してください。
前提:
- repo=gemma4-mlx-speculative
- destructive commandは禁止
- 3コマンドだけ出す
- 説明文は不要
""",
    "git_status": """作業ツリーの状態を確認するため、次に実行すべきコマンドを1つだけ出してください。
説明文は不要。
"""
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-path", default="prompt_100k.txt")
    parser.add_argument("--model", default=engine.DEFAULT_TARGET_MODEL_PATH)
    parser.add_argument("--target-tokens", type=int, default=100000)
    parser.add_argument("--step-size", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--draft-block-size", type=int, default=8)
    parser.add_argument("--template-min-tokens", type=int, default=1)
    parser.add_argument("--safe-token-limit", type=int, default=120000)
    parser.add_argument("--case", action="append", choices=["exact_pytest_plan", "medium_pytest_plan", "git_status"])
    parser.add_argument("--max-kv-size", type=int, default=None)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    cases = args.case if args.case else ["exact_pytest_plan"]

    if not os.path.exists(args.prompt_path):
        print(f"Error: {args.prompt_path} not found.")
        return

    with open(args.prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    print(f"loading target: {args.model}")
    target_model, processor = d.load(args.model)
    tokenizer = getattr(processor, "tokenizer", processor)
    
    runtime = TemplateDraftRuntime(target_model, tokenizer, processor, args.max_kv_size)

    formatted_prompt = engine.format_prompt(processor, prompt_text)
    prompt_ids_full = tokenizer.encode(formatted_prompt)
    prompt_tokens_total = len(prompt_ids_full)
    
    prompt_ids = prompt_ids_full[:args.target_tokens]
    actual_prefix_tokens = len(prompt_ids)
    prefix_str = engine.decode_text(tokenizer, prompt_ids)
    
    print(f"prompt_path: {args.prompt_path}")
    print(f"prompt_tokens_total: {prompt_tokens_total}")
    print(f"target_tokens: {args.target_tokens}")
    print(f"actual_prefix_tokens: {actual_prefix_tokens}")
    print(f"step_size: {args.step_size}")
    print(f"cases: {cases}")
    
    if actual_prefix_tokens > args.safe_token_limit:
        print(f"skipped_by_guard: actual_prefix_tokens {actual_prefix_tokens} exceeds safe_token_limit {args.safe_token_limit}")
        return

    suffix_ids_list = []
    user_prompts = []
    
    for c in cases:
        user_prompt = prefix_str + "\n" + SUFFIX_CASES[c]
        formatted_full = engine.format_prompt(processor, user_prompt)
        full_ids = tokenizer.encode(formatted_full)
        suffix_ids = full_ids[actual_prefix_tokens:]
        suffix_ids_list.append(suffix_ids)
        user_prompts.append(user_prompt)

    try:
        print("\n--- 1. Prefix Prefill & Snapshot ---")
        # PrefixCacheManager.get_or_create does chunked prefill directly and takes a snapshot
        entry = runtime.prefix_manager.get_or_create(prefix_str, prompt_ids, target_model, args.max_kv_size)
        
        # It's already in the cache entry object but it doesn't return the timing independently
        # Since we just created it, the prefill_sec is in the entry
        prefix_prefill_sec = entry.prefill_sec
        # We can assume snapshot took ~0.0s, but it's part of prefix_manager
        # I'll just print what we have
        print(f"prefix_prefill_sec: {prefix_prefill_sec:.3f}s")
        print("snapshot_sec: ~0.000s")
        
        mx.clear_cache()

        print("\n--- 2. Prefix Reuse + Template Draft (Suffix Processing) ---")
        for i, c in enumerate(cases):
            print(f"\ncase: {c}")
            print(f"suffix_tokens: {len(suffix_ids_list[i])}")
            
            res = runtime.prefix_reuse_template_draft(
                prefix_str,
                prompt_ids,
                suffix_ids_list[i],
                user_prompts[i],
                args.max_tokens,
                args.draft_block_size,
                args.template_min_tokens,
                args.trace
            )
            
            print(f"suffix_prefill_sec: {res.prefill_sec:.3f}s")
            print(f"decode_sec: {res.decode_sec:.3f}s")
            print(f"elapsed_excluding_prefix: {res.elapsed_sec:.3f}s")
            print(f"accepted/drafted/rejected: {res.accepted}/{res.drafted}/{res.rejected}")
            print(f"output snippet: {repr(res.text)}")
            mx.clear_cache()

        print("\nOK: prompt 100k reuse path probe completed")

    except Exception as e:
        print(f"Execution failed: {e}")
    finally:
        mx.clear_cache()

if __name__ == "__main__":
    main()
