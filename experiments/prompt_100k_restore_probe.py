import argparse
import time
import os
import importlib

import mlx.core as mx
import template_draft_engine as engine

d = importlib.import_module("mlx_vlm.generate.dispatch")

def chunked_prefill(target_model, prompt_ids, max_kv_size, step_size=512):
    lm = engine.get_lm(target_model)
    prompt_cache = engine.make_cache(lm, max_kv_size)
    
    input_arr = mx.array([prompt_ids])
    emb = target_model.get_input_embeddings(input_arr, None, mask=None)
    inputs_embeds = emb.inputs_embeds
    extra = {k: v for k, v in emb.to_dict().items() if k != "inputs_embeds" and v is not None}
    
    start = time.perf_counter()
    if input_arr.shape[1] > 1:
        n = input_arr.shape[1] - 1
        for i in range(0, n, step_size):
            chunk_len = min(step_size, n - i)
            lm(
                input_arr[:, i:i+chunk_len],
                inputs_embeds=inputs_embeds[:, i:i+chunk_len] if inputs_embeds is not None else None,
                cache=prompt_cache,
                n_to_process=chunk_len,
                **extra,
            )
            engine.eval_cache(prompt_cache)
            
    cur = input_arr[:, -1:]
    prefill_sec = time.perf_counter() - start
    
    first_logits = engine.forward_one(lm, cur, prompt_cache)
    return lm, prompt_cache, first_logits, prefill_sec

def decode_greedy(lm, first_logits, prompt_cache, stop_ids, max_tokens):
    start = time.perf_counter()
    out = []
    
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
            
    decode_sec = time.perf_counter() - start
    return out, decode_sec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-path", default="prompt_100k.txt")
    parser.add_argument("--model", default=engine.DEFAULT_TARGET_MODEL_PATH)
    parser.add_argument("--target-tokens", type=int, default=32000)
    parser.add_argument("--step-size", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--advance-tokens", type=int, default=8)
    parser.add_argument("--safe-token-limit", type=int, default=120000)
    parser.add_argument("--max-kv-size", type=int, default=None)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.prompt_path):
        print(f"Error: {args.prompt_path} not found.")
        return

    with open(args.prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    print(f"loading target: {args.model}")
    target_model, processor = d.load(args.model)
    tokenizer = getattr(processor, "tokenizer", processor)
    stop_ids = engine.build_stop_ids(tokenizer)
    
    formatted_prompt = engine.format_prompt(processor, prompt_text)
    prompt_ids_full = tokenizer.encode(formatted_prompt)
    prompt_tokens_total = len(prompt_ids_full)
    
    prompt_ids = prompt_ids_full[:args.target_tokens]
    actual_prefix_tokens = len(prompt_ids)
    
    print(f"prompt_path: {args.prompt_path}")
    print(f"prompt_tokens_total: {prompt_tokens_total}")
    print(f"target_tokens: {args.target_tokens}")
    print(f"actual_prefix_tokens: {actual_prefix_tokens}")
    print(f"step_size: {args.step_size}")
    
    if actual_prefix_tokens > args.safe_token_limit:
        print(f"skipped_by_guard: actual_prefix_tokens {actual_prefix_tokens} exceeds safe_token_limit {args.safe_token_limit}")
        return

    try:
        # 1. Baseline chunked prefill and decode
        print("\n--- Baseline ---")
        lm_b, cache_b, logits_b, baseline_prefill_sec = chunked_prefill(target_model, prompt_ids, args.max_kv_size, args.step_size)
        print(f"baseline_prefill_sec: {baseline_prefill_sec:.3f}s")
        
        baseline_ids, baseline_decode_sec = decode_greedy(lm_b, logits_b, cache_b, stop_ids, args.max_tokens)
        print(f"baseline_decode_sec: {baseline_decode_sec:.3f}s")
        if args.trace:
            print(f"baseline_ids: {baseline_ids}")
        mx.clear_cache()
        
        # 2. Snapshot, advance, restore, decode
        print("\n--- Restore Test ---")
        lm_s, cache_s, logits_s, snapshot_prefill_sec = chunked_prefill(target_model, prompt_ids, args.max_kv_size, args.step_size)
        print(f"snapshot_prefill_sec: {snapshot_prefill_sec:.3f}s")
        
        start = time.perf_counter()
        snap = engine.full_snapshot(cache_s)
        snapshot_create_sec = time.perf_counter() - start
        print(f"snapshot_create_sec: {snapshot_create_sec:.3f}s")
        
        start = time.perf_counter()
        tok = engine.argmax_token(logits_s)
        if int(tok.item()) not in stop_ids:
            next_logits = engine.forward_one(lm_s, tok[:, None], cache_s)
            for _ in range(args.advance_tokens - 1):
                tok = engine.argmax_token(next_logits)
                if int(tok.item()) in stop_ids:
                    break
                next_logits = engine.forward_one(lm_s, tok[:, None], cache_s)
        advance_sec = time.perf_counter() - start
        print(f"advance_sec: {advance_sec:.3f}s")
        
        start = time.perf_counter()
        engine.restore_full(cache_s, snap)
        restore_sec = time.perf_counter() - start
        print(f"restore_sec: {restore_sec:.3f}s")
        
        restored_ids, restored_decode_sec = decode_greedy(lm_s, logits_s, cache_s, stop_ids, args.max_tokens)
        print(f"restored_decode_sec: {restored_decode_sec:.3f}s")
        if args.trace:
            print(f"restored_ids: {restored_ids}")
            
        print(f"\nbaseline_ids: {baseline_ids}")
        print(f"restored_ids: {restored_ids}")
        
        if baseline_ids == restored_ids:
            print("\nOK: 100k restore probe completed with token match")
        else:
            print("\nFAIL: token mismatch")
            
    except Exception as e:
        print(f"Execution failed: {e}")
    finally:
        mx.clear_cache()

if __name__ == "__main__":
    main()
