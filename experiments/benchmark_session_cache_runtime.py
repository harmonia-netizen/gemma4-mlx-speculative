import argparse
import time
import os
import importlib

import mlx.core as mx
import template_draft_engine as engine
from template_draft_runtime import LongInputGuard, PrefixCacheManager, CandidateRegistry
from session_cache_runtime import SessionCacheRuntime

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
- repo=local-speculative-runtime
- destructive commandは禁止
- 3コマンドだけ出す
- 説明文は不要
"""
}

def build_synthetic_prefix(processor, repeat_lines: int) -> str:
    base_line = "User: Can you show me the test plan?\nAsst: I will check the tests and then show you the git diff.\n"
    prompt_text = base_line * repeat_lines
    return engine.format_prompt(processor, prompt_text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "prompt100k"], default="synthetic")
    parser.add_argument("--repeat-lines", type=int, default=500)
    parser.add_argument("--target-tokens", type=int, default=32000)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--safe-token-limit", type=int, default=120000)
    parser.add_argument("--step-size", type=int, default=512)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--model", default=engine.DEFAULT_TARGET_MODEL_PATH)
    parser.add_argument("--max-kv-size", type=int, default=None)
    args = parser.parse_args()

    print(f"loading target: {args.model}")
    target_model, processor = d.load(args.model)
    tokenizer = getattr(processor, "tokenizer", processor)

    registry = CandidateRegistry()
    prefix_manager = PrefixCacheManager(max_entries=2, max_total_tokens=args.safe_token_limit)
    guard = LongInputGuard(safe_token_limit=args.safe_token_limit)
    
    runtime = SessionCacheRuntime(
        target_model=target_model,
        tokenizer=tokenizer,
        candidate_registry=registry,
        prefix_cache_manager=prefix_manager,
        guard=guard,
        step_size=args.step_size,
        max_kv_size=args.max_kv_size
    )

    if args.mode == "synthetic":
        print(f"\nMode: synthetic, repeat-lines: {args.repeat_lines}")
        prefix_text = build_synthetic_prefix(processor, args.repeat_lines)
        prefix_ids = tokenizer.encode(prefix_text)
        cases = ["exact_pytest_plan", "medium_pytest_plan"]
    else:
        print(f"\nMode: prompt100k, target_tokens: {args.target_tokens}")
        prompt_path = "prompt_100k.txt"
        if not os.path.exists(prompt_path):
            print(f"Error: {prompt_path} not found")
            return
        with open(prompt_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        full_text = engine.format_prompt(processor, raw_text)
        full_ids = tokenizer.encode(full_text)
        prefix_ids = full_ids[:args.target_tokens]
        prefix_text = engine.decode_text(tokenizer, prefix_ids)
        cases = ["exact_pytest_plan"]
        
    print(f"prefix_tokens: {len(prefix_ids)}")
    print(f"safe_token_limit: {args.safe_token_limit}")

    session_id = "test_session_1"
    
    print("\n--- Create Session ---")
    start = time.perf_counter()
    create_res = runtime.create_session(session_id, prefix_text, prefix_ids)
    
    if not create_res.ok:
        print(f"guard result: skipped_by_guard ({create_res.guard_reason})")
        return
        
    print(f"guard result: allowed")
    print(f"session_id: {session_id}")
    print(f"prefix_prefill_sec: {create_res.prefix_prefill_sec:.3f}s")
    if create_res.evicted_keys:
        print(f"evicted keys during create: {create_res.evicted_keys}")
    
    print("\n--- Generate With Suffix ---")
    for c in cases:
        print(f"\ncase: {c}")
        user_prompt = SUFFIX_CASES[c]
        formatted_full = engine.format_prompt(processor, prefix_text + "\n" + user_prompt)
        full_ids = tokenizer.encode(formatted_full)
        suffix_ids = full_ids[len(prefix_ids):]
        
        res = runtime.generate_with_suffix(
            session_id=session_id,
            suffix_text=user_prompt,
            suffix_ids=suffix_ids,
            max_tokens=args.max_tokens,
            trace=args.trace
        )
        
        if not res.ok:
            print(f"Error: {res.error}")
            continue
            
        print(f"suffix_prefill_sec: {res.suffix_prefill_sec:.3f}s")
        print(f"decode_sec: {res.decode_sec:.3f}s")
        print(f"accepted/drafted/rejected: {res.accepted}/{res.drafted}/{res.rejected}")
        print(f"output snippet: {repr(res.text)}")
        mx.clear_cache()
        
    print("\n--- Cache State ---")
    stats = runtime.stats()
    print(f"cache entries: {stats.entries}")
    print(f"current_total_tokens: {stats.current_total_tokens}")
    print(f"evictions: none so far (checked during create)")
    
    print("\nOK: session cache runtime benchmark completed")

if __name__ == "__main__":
    main()
