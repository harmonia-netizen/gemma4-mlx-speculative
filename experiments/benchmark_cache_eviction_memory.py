import argparse
import time
import os
import importlib

import mlx.core as mx
import template_draft_engine as engine
from session_cache_core import SessionState, CacheStats
from template_draft_runtime import PrefixCacheManager, PrefixCacheEntry

def run_mock(args):
    print("\n--- Running Mock Mode ---")
    print(f"sessions: {args.sessions}")
    print(f"max-entries: {args.max_entries}")
    print(f"max-total-tokens: {args.max_total_tokens}")

    manager = PrefixCacheManager(max_entries=args.max_entries, max_total_tokens=args.max_total_tokens)
    
    for i in range(args.sessions):
        print(f"\n[Session {i}]")
        prefix_ids = [0] * args.target_tokens
        prefix_text = f"mock_prefix_{i}"
        
        try:
            if len(prefix_ids) > manager.max_total_tokens:
                raise ValueError(f"prefix length {len(prefix_ids)} exceeds max_total_tokens {manager.max_total_tokens}")
            
            evicted = manager.evict_if_needed(len(prefix_ids))
            print(f"Evicted keys: {evicted}")
            
            entry = PrefixCacheEntry(
                text_hash=manager._hash(prefix_text),
                token_ids=prefix_ids,
                snapshot=None,
                cache=None,
                prefill_sec=0.0,
                created_at=time.time(),
                last_used_at=time.time(),
                hit_count=0,
                evicted_keys=evicted
            )
            manager.entries[entry.text_hash] = entry
            manager.current_total_tokens += len(prefix_ids)
            print(f"Added cache entry for {entry.text_hash[:8]}...")
        except Exception as e:
            print(f"Failed to add: {e}")
            
        print(f"Current Cache: {len(manager.entries)} entries, {manager.current_total_tokens} tokens")

def run_real32k(args):
    print("\n--- Running Real 32K Mode ---")
    d = importlib.import_module("mlx_vlm.generate.dispatch")
    from session_cache_runtime import SessionCacheRuntime
    from template_draft_runtime import CandidateRegistry, LongInputGuard
    
    prompt_path = "prompt_100k.txt"
    if not os.path.exists(prompt_path):
        print(f"Error: {prompt_path} not found")
        return
        
    print(f"loading target: {engine.DEFAULT_TARGET_MODEL_PATH}")
    target_model, processor = d.load(engine.DEFAULT_TARGET_MODEL_PATH)
    tokenizer = getattr(processor, "tokenizer", processor)

    registry = CandidateRegistry()
    prefix_manager = PrefixCacheManager(max_entries=args.max_entries, max_total_tokens=args.max_total_tokens)
    guard = LongInputGuard(safe_token_limit=args.safe_token_limit)
    
    runtime = SessionCacheRuntime(
        target_model=target_model,
        tokenizer=tokenizer,
        candidate_registry=registry,
        prefix_cache_manager=prefix_manager,
        guard=guard,
        step_size=args.step_size,
    )

    with open(prompt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
        
    full_text = engine.format_prompt(processor, raw_text)
    full_ids = tokenizer.encode(full_text)
    
    session_ids = []
    
    for i in range(args.sessions):
        print(f"\n[Create Session {i}]")
        prefix_ids = full_ids[i * 10 : args.target_tokens + i * 10]
        prefix_text = engine.decode_text(tokenizer, prefix_ids)
        session_id = f"session_{i}"
        
        res = runtime.create_session(session_id, prefix_text, prefix_ids)
        print(f"ok: {res.ok}")
        if not res.ok:
            print(f"reason: {res.guard_reason}")
            continue
            
        print(f"evicted_keys: {res.evicted_keys}")
        print(f"prefix_prefill_sec: {res.prefix_prefill_sec:.3f}s")
        session_ids.append(session_id)
        
        stats = runtime.stats()
        print(f"Cache stats: entries={stats.entries}, tokens={stats.current_total_tokens}")

    print("\n[Generate Test on Evicted Session]")
    if session_ids:
        oldest_session = session_ids[0]
        print(f"Generating on oldest session {oldest_session}")
        gen_res = runtime.generate_with_suffix(
            session_id=oldest_session,
            suffix_text="次の確認手順をbashブロックだけで出してください",
            suffix_ids=tokenizer.encode("次の確認手順をbashブロックだけで出してください")
        )
        print(f"ok: {gen_res.ok}")
        print(f"error: {gen_res.error}")

    print("\nOK: benchmark cache eviction memory completed")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mock", "real32k"], default="mock")
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--target-tokens", type=int, default=32000)
    parser.add_argument("--max-entries", type=int, default=2)
    parser.add_argument("--max-total-tokens", type=int, default=64000)
    parser.add_argument("--safe-token-limit", type=int, default=120000)
    parser.add_argument("--step-size", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    if args.mode == "mock":
        run_mock(args)
    else:
        run_real32k(args)

if __name__ == "__main__":
    main()
