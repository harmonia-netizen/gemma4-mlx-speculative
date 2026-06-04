import argparse
import time
import os
import importlib

import mlx.core as mx
import template_draft_engine as engine

d = importlib.import_module("mlx_vlm.generate.dispatch")

def run_prefill(target_model, prompt_ids, max_kv_size):
    lm = engine.get_lm(target_model)
    prompt_cache = engine.make_cache(lm, max_kv_size)
    
    input_arr = mx.array([prompt_ids])
    emb = target_model.get_input_embeddings(input_arr, None, mask=None)
    inputs_embeds = emb.inputs_embeds
    extra = {k: v for k, v in emb.to_dict().items() if k != "inputs_embeds" and v is not None}
    
    print("Starting prefill evaluation (chunked)...")
    start = time.perf_counter()
    
    step_size = 512
    total_len = input_arr.shape[1]
    
    for i in range(0, total_len, step_size):
        chunk_len = min(step_size, total_len - i)
        lm(
            input_arr[:, i:i+chunk_len],
            inputs_embeds=inputs_embeds[:, i:i+chunk_len] if inputs_embeds is not None else None,
            cache=prompt_cache,
            n_to_process=chunk_len,
            **extra,
        )
        engine.eval_cache(prompt_cache)
        
    prefill_sec = time.perf_counter() - start
    print(f"Prefill complete in {prefill_sec:.3f}s")
    
    cache_len = len(prompt_cache)
    print(f"Cache summary: {cache_len} layers initialized")
    return prompt_cache, prefill_sec


def run_prefill_snapshot(target_model, prompt_ids, max_kv_size):
    prompt_cache, prefill_sec = run_prefill(target_model, prompt_ids, max_kv_size)
    print("Taking full snapshot of the prompt cache...")
    start = time.perf_counter()
    snap = engine.full_snapshot(prompt_cache)
    snap_sec = time.perf_counter() - start
    print(f"Snapshot complete in {snap_sec:.3f}s")
    return snap, snap_sec


def run_greedy_decode(target_model, tokenizer, prompt_ids, max_kv_size, stop_ids, max_tokens):
    lm = engine.get_lm(target_model)
    prompt_cache = engine.make_cache(lm, max_kv_size)
    
    input_arr = mx.array([prompt_ids])
    emb = target_model.get_input_embeddings(input_arr, None, mask=None)
    inputs_embeds = emb.inputs_embeds
    extra = {k: v for k, v in emb.to_dict().items() if k != "inputs_embeds" and v is not None}
    
    print(f"Starting prefill evaluation (chunked) for {input_arr.shape[1]} tokens...")
    start = time.perf_counter()
    if input_arr.shape[1] > 1:
        step_size = 512
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
    print(f"Prefill complete in {prefill_sec:.3f}s")
    
    print(f"Starting decode for max {max_tokens} tokens...")
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
    print(f"Decode complete in {decode_sec:.3f}s")
    
    text = engine.decode_text(tokenizer, out)
    print(f"Output snippet: {repr(text)}")
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-path", default="prompt_100k.txt")
    parser.add_argument("--model", default=engine.DEFAULT_TARGET_MODEL_PATH)
    parser.add_argument("--mode", choices=["count", "prefill", "prefill-snapshot", "greedy"], default="count")
    parser.add_argument("--safe-token-limit", type=int, default=120000)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-kv-size", type=int, default=None)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.prompt_path):
        print(f"Error: {args.prompt_path} not found.")
        return

    with open(args.prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    print(f"Loading target model: {args.model}")
    target_model, processor = d.load(args.model)
    tokenizer = getattr(processor, "tokenizer", processor)
    
    formatted_prompt = engine.format_prompt(processor, prompt_text)
    prompt_ids = tokenizer.encode(formatted_prompt)
    token_count = len(prompt_ids)
    
    print(f"Prompt formatting complete. Token count: {token_count}")
    
    if args.mode == "count":
        return
        
    if token_count > args.safe_token_limit:
        print(f"skipped_by_guard: token_count {token_count} exceeds safe_token_limit {args.safe_token_limit}")
        return

    print(f"\n--- Running mode: {args.mode} ---")
    try:
        if args.mode == "prefill":
            run_prefill(target_model, prompt_ids, args.max_kv_size)
        elif args.mode == "prefill-snapshot":
            run_prefill_snapshot(target_model, prompt_ids, args.max_kv_size)
        elif args.mode == "greedy":
            stop_ids = engine.build_stop_ids(tokenizer)
            run_greedy_decode(target_model, tokenizer, prompt_ids, args.max_kv_size, stop_ids, args.max_tokens)
    except Exception as e:
        print(f"Execution failed: {e}")
    finally:
        mx.clear_cache()
        print("Done. Cleared MX cache.")

if __name__ == "__main__":
    main()
