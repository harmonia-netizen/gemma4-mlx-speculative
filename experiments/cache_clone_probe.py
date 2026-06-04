import argparse
import sys
import importlib
import mlx.core as mx

d = importlib.import_module("mlx_vlm.generate.dispatch")
from run_gemma4_template_draft_v10 import (
    DEFAULT_TARGET_MODEL_PATH,
    DEFAULT_PROMPT,
    get_lm,
    format_prompt,
    build_stop_ids,
    bootstrap_after_first_token,
    argmax_token,
    forward_one,
    decode_text,
    full_snapshot,
    restore_full,
)

MX_ARRAY_TYPE = type(mx.array([0]))

def clone_value(v):
    if isinstance(v, MX_ARRAY_TYPE):
        x = mx.array(v)
        mx.eval(x)
        return x
    if isinstance(v, tuple):
        return tuple(clone_value(x) for x in v)
    if isinstance(v, list):
        return [clone_value(x) for x in v]
    if isinstance(v, dict):
        return {k: clone_value(x) for k, x in v.items()}
    return v

def iter_slot_names(cls):
    names = []
    for base in reversed(cls.__mro__):
        slots = getattr(base, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for s in slots:
            if s not in ("__dict__", "__weakref__"):
                names.append(s)
    return names

def clone_cache_object(c):
    nc = object.__new__(type(c))
    
    if hasattr(c, "__dict__"):
        for k, v in c.__dict__.items():
            setattr(nc, k, clone_value(v))
            
    for slot in iter_slot_names(type(c)):
        if hasattr(c, slot):
            setattr(nc, slot, clone_value(getattr(c, slot)))
            
    return nc

def clone_cache_objects(prompt_cache):
    return [clone_cache_object(c) for c in prompt_cache]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_TARGET_MODEL_PATH)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--clone-advance-tokens", type=int, default=8)
    parser.add_argument("--max-kv-size", type=int, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("loading model:", args.model)
    target_model, processor = d.load(args.model)
    tokenizer = getattr(processor, "tokenizer", processor)
    lm = get_lm(target_model)
    
    formatted_prompt = format_prompt(processor, args.prompt)
    try:
        prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    except TypeError:
        prompt_ids = tokenizer.encode(formatted_prompt)
    print("prompt_tokens:", len(prompt_ids))
    
    first, target_next_logits, live_cache, prefill_sec = bootstrap_after_first_token(
        target_model,
        lm,
        tokenizer,
        formatted_prompt,
        args.max_kv_size,
    )
    
    snap = full_snapshot(live_cache)
    
    # D: baseline tokens
    baseline_tokens = []
    baseline_logits = target_next_logits
    
    for _ in range(args.tokens):
        tok = argmax_token(baseline_logits)
        baseline_tokens.append(int(tok.item()))
        baseline_logits = forward_one(lm, tok[:, None], live_cache)
        
    baseline_text = decode_text(tokenizer, baseline_tokens)
    print("baseline tokens:", baseline_tokens)
    print("baseline text:", repr(baseline_text))
    
    # E: return to initial state
    restore_full(live_cache, snap)
    
    # F: create cloned cache
    cloned_cache = clone_cache_objects(live_cache)
    
    # Ensure no src is dst
    for i, (c, nc) in enumerate(zip(live_cache, cloned_cache)):
        if c is nc:
            print("ERROR: src is dst for cache index", i)
            sys.exit(1)
            
    print("clone_advance_tokens:", args.clone_advance_tokens)
            
    # G: advance cloned_cache
    clone_next_logits = target_next_logits
    for _ in range(args.clone_advance_tokens):
        tok = argmax_token(clone_next_logits)
        clone_next_logits = forward_one(lm, tok[:, None], cloned_cache)
        
    # H: verify live_cache still produces baseline
    actual_tokens = []
    actual_logits = target_next_logits
    
    for _ in range(args.tokens):
        tok = argmax_token(actual_logits)
        actual_tokens.append(int(tok.item()))
        actual_logits = forward_one(lm, tok[:, None], live_cache)
        
    actual_text = decode_text(tokenizer, actual_tokens)
    print("after clone-forward live tokens:", actual_tokens)
    print("after clone-forward live text:", repr(actual_text))
    
    if baseline_tokens != actual_tokens:
        print("MISMATCH")
        print("baseline ids:", baseline_tokens)
        print("actual ids:", actual_tokens)
        print("baseline text:", repr(baseline_text))
        print("actual text:", repr(actual_text))
        sys.exit(2)
        
    print("OK: cloned cache did not affect live cache")

if __name__ == "__main__":
    main()
