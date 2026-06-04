import argparse
import sys
import importlib
import mlx.core as mx

d = importlib.import_module("mlx_vlm.generate.dispatch")
from run_gemma4_template_draft_v10 import (
    DEFAULT_TARGET_MODEL_PATH,
    get_lm,
    format_prompt,
    bootstrap_after_first_token,
    argmax_token,
    forward_one,
    decode_text,
    commit_tokens_to_cache,
    build_stop_ids,
)

from cache_clone_probe import clone_cache_objects

PROMPT = """あなたはローカル常駐エージェントです。
pytest失敗の原因を安全に確認するため、次に実行する確認手順を3行のbashブロックで出してください。
前提:
- repo=gemma4-mlx-speculative
- destructive commandは禁止
- 3コマンドだけ出す
- 説明文は不要
"""

TEMPLATE = """```bash
git status --short
pytest --tb=short
git diff
```"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_TARGET_MODEL_PATH)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--max-kv-size", type=int, default=None)
    parser.add_argument("--trace", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()

    print("loading model:", args.model)
    target_model, processor = d.load(args.model)
    tokenizer = getattr(processor, "tokenizer", processor)
    lm = get_lm(target_model)

    formatted_prompt = format_prompt(processor, PROMPT)
    stop_ids = build_stop_ids(tokenizer)

    print("C. Baseline greedy run")
    first_base, next_logits_base, cache_base, _ = bootstrap_after_first_token(
        target_model, lm, tokenizer, formatted_prompt, args.max_kv_size
    )

    baseline_tokens = [int(first_base.item())]
    if baseline_tokens[-1] not in stop_ids:
        logits = next_logits_base
        for _ in range(args.tokens - 1):
            tok = argmax_token(logits)
            tid = int(tok.item())
            baseline_tokens.append(tid)
            if tid in stop_ids:
                break
            logits = forward_one(lm, tok[:, None], cache_base)
            
    print("baseline tokens:", baseline_tokens)
    print("baseline text:", repr(decode_text(tokenizer, baseline_tokens)))

    print("D. Second run")
    first, next_logits, live_cache, _ = bootstrap_after_first_token(
        target_model, lm, tokenizer, formatted_prompt, args.max_kv_size
    )

    actual_tokens = []
    first_id = int(first.item())
    
    if first_id in stop_ids:
        actual_tokens.append(first_id)
    else:
        actual_tokens.append(first_id)

        # E. Template candidate -> tokens
        try:
            candidate_ids = tokenizer.encode(TEMPLATE, add_special_tokens=False)
        except TypeError:
            candidate_ids = tokenizer.encode(TEMPLATE)

        # F. First token match check
        if candidate_ids and candidate_ids[0] == first_id:
            candidate_ids = candidate_ids[1:]

        # Create cloned cache for verification
        cloned_cache = clone_cache_objects(live_cache)
        verify_logits = next_logits

        emitted = []
        reject_idx = -1
        
        # G. Verify candidate on CLONED cache
        for i, proposed_id in enumerate(candidate_ids):
            target_tok = argmax_token(verify_logits)
            target_id = int(target_tok.item())
            
            if target_id != proposed_id:
                # H. Reject on mismatch
                reject_idx = i
                if args.trace:
                    print("--- trace ---")
                    print(f"reject index: {i}")
                    print(f"target_id: {target_id}")
                    print(f"proposed_id: {proposed_id}")
                    print(f"decoded target token: {repr(decode_text(tokenizer, [target_id]))}")
                    print(f"decoded proposed token: {repr(decode_text(tokenizer, [proposed_id]))}")
                    print("-------------")
                break
                
            emitted.append(proposed_id)
            
            verify_logits = forward_one(lm, mx.array([[proposed_id]], dtype=first.dtype), cloned_cache)

            if proposed_id in stop_ids:
                break
            
            if len(actual_tokens) + len(emitted) >= args.tokens:
                break
                
        # I. Continue with live cache greedy from rejection point
        if emitted:
            actual_tokens.extend(emitted)
            next_logits = commit_tokens_to_cache(lm, live_cache, emitted, first.dtype)
            if next_logits is not None:
                mx.eval(next_logits)

        while len(actual_tokens) < args.tokens and actual_tokens[-1] not in stop_ids:
            tok = argmax_token(next_logits)
            tid = int(tok.item())
            actual_tokens.append(tid)
            if tid in stop_ids:
                break
            next_logits = forward_one(lm, tok[:, None], live_cache)

    print("actual tokens:", actual_tokens)
    print("actual text:", repr(decode_text(tokenizer, actual_tokens)))

    # J. Compare
    if baseline_tokens != actual_tokens:
        print("MISMATCH")
        print("baseline ids:", baseline_tokens)
        print("actual ids:", actual_tokens)
        sys.exit(2)

    print("OK: verify fallback actual tokens match baseline")

if __name__ == "__main__":
    main()
