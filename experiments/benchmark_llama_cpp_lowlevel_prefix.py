import argparse
import sys
import time
import json
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model-label", type=str, default="gguf")
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    original_stdout_fd = None
    if args.json:
        sys.stdout.flush()
        original_stdout_fd = os.dup(1)
        os.dup2(2, 1)

    try:
        import llama_cpp
    except ImportError:
        if args.json:
            sys.stdout.flush()
            os.dup2(original_stdout_fd, 1)
            os.close(original_stdout_fd)
            print(json.dumps({
                "ok": False, 
                "backend": "llama_cpp",
                "model_path": args.model,
                "error": "llama-cpp-python not found"
            }))
        else:
            print("SKIP: llama-cpp-python not found")
        return

    if not args.json:
        print("--- 1. Initialize ---")
        
    llm = llama_cpp.Llama(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        verbose=args.trace
    )

    prefix_text = "You are a concise assistant. " * 50 + "\n"
    suffixA_text = "Say exactly: APPLE."
    suffixB_text = "Say exactly: BANANA."

    # Baseline
    if not args.json:
        print("\n--- 2. Baseline ---")
        
    start = time.perf_counter()
    resA_base = llm.create_completion(prompt=prefix_text + suffixA_text, max_tokens=args.max_tokens, temperature=0.0)
    timeA_base = time.perf_counter() - start
    
    start = time.perf_counter()
    resB_base = llm.create_completion(prompt=prefix_text + suffixB_text, max_tokens=args.max_tokens, temperature=0.0)
    timeB_base = time.perf_counter() - start
    
    baseline_total_sec = timeA_base + timeB_base

    # Lowlevel
    if not args.json:
        print("\n--- 3. Lowlevel Prefix Evaluation ---")
        
    llm.reset()
    tokens_prefix = llm.tokenize(prefix_text.encode("utf-8"), add_bos=True)
    
    start = time.perf_counter()
    llm.eval(tokens_prefix)
    lowlevel_prefix_eval_sec = time.perf_counter() - start
    
    state = llm.save_state()
    
    # generate A
    start_ls = time.perf_counter()
    resA_low = None
    resB_low = None
    lowlevel_suffix_a_sec = 0.0
    lowlevel_suffix_b_sec = 0.0
    lowlevel_total_excluding_prefix_sec = 0.0
    failure_reason = ""
    reuse_effective = False
    speedup_vs_baseline = 0.0

    debug_info = {}
    try:
        debug_info["n_tokens_before_load_A"] = getattr(llm, "n_tokens", None)
        llm.load_state(state)
        debug_info["n_tokens_after_load_A"] = getattr(llm, "n_tokens", None)
        
        if not args.json:
            print("load_state time:", time.perf_counter() - start_ls)
            
        start = time.perf_counter()
        resA_low = llm.create_completion(prompt=prefix_text + suffixA_text, max_tokens=args.max_tokens, temperature=0.0)
        lowlevel_suffix_a_sec = time.perf_counter() - start
        
        # generate B
        start_ls = time.perf_counter()
        debug_info["n_tokens_before_load_B"] = getattr(llm, "n_tokens", None)
        llm.load_state(state)
        debug_info["n_tokens_after_load_B"] = getattr(llm, "n_tokens", None)
        
        if not args.json:
            print("load_state time:", time.perf_counter() - start_ls)
            
        start = time.perf_counter()
        resB_low = llm.create_completion(prompt=prefix_text + suffixB_text, max_tokens=args.max_tokens, temperature=0.0)
        lowlevel_suffix_b_sec = time.perf_counter() - start
        
        lowlevel_total_excluding_prefix_sec = lowlevel_suffix_a_sec + lowlevel_suffix_b_sec
        
        speedup_vs_baseline = baseline_total_sec / (lowlevel_prefix_eval_sec + lowlevel_total_excluding_prefix_sec) if (lowlevel_prefix_eval_sec + lowlevel_total_excluding_prefix_sec) > 0 else 0
        reuse_effective = (speedup_vs_baseline > 1.1)

    except Exception as e:
        failure_reason = str(e)
        import traceback
        debug_info["traceback"] = traceback.format_exc()

    if not failure_reason and not reuse_effective:
        if getattr(llm, "_is_hybrid", False) or getattr(llm, "_is_recurrent", False):
            failure_reason = "Model is hybrid/recurrent, llama.cpp forces full state reset on branch (even via load_state + manual eval)."
        else:
            failure_reason = "Low-level API did not yield expected speedup."

    if not args.json:
        print("\n--- Summary ---")
        print(f"baseline_total_sec: {baseline_total_sec:.3f}s (A: {timeA_base:.3f}s, B: {timeB_base:.3f}s)")
        print(f"lowlevel_prefix_eval_sec: {lowlevel_prefix_eval_sec:.3f}s")
        print(f"lowlevel_suffix_a_sec: {lowlevel_suffix_a_sec:.3f}s")
        print(f"lowlevel_suffix_b_sec: {lowlevel_suffix_b_sec:.3f}s")
        print(f"lowlevel_total_excluding_prefix_sec: {lowlevel_total_excluding_prefix_sec:.3f}s")
        print(f"speedup_vs_baseline: {speedup_vs_baseline:.3f}x")
        print(f"reuse_effective: {reuse_effective}")
        if failure_reason:
            print(f"failure_reason: {failure_reason}")
    else:
        sys.stdout.flush()
        os.dup2(original_stdout_fd, 1)
        os.close(original_stdout_fd)
            
        output = {
            "ok": not bool(failure_reason) and reuse_effective,
            "backend": "llama_cpp",
            "model_path": args.model,
            "baseline_total_sec": baseline_total_sec,
            "lowlevel_prefix_eval_sec": lowlevel_prefix_eval_sec,
            "lowlevel_suffix_a_sec": lowlevel_suffix_a_sec,
            "lowlevel_suffix_b_sec": lowlevel_suffix_b_sec,
            "lowlevel_total_excluding_prefix_sec": lowlevel_total_excluding_prefix_sec,
            "speedup_vs_baseline": speedup_vs_baseline,
            "reuse_effective": reuse_effective,
            "failure_reason": failure_reason if failure_reason else None,
            "debug": debug_info,
            "outputs": {
                "base_A": resA_base["choices"][0]["text"].strip() if resA_base else "",
                "base_B": resB_base["choices"][0]["text"].strip() if resB_base else "",
                "low_A": resA_low["choices"][0]["text"].strip() if resA_low else "",
                "low_B": resB_low["choices"][0]["text"].strip() if resB_low else ""
            }
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
