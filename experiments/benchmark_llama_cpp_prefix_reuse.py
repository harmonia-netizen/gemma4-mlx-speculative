import argparse
import sys
import os
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    print(f"--- 1. Init API (backend=llama_cpp) ---")
    
    try:
        import llama_cpp
    except ImportError:
        print("SKIP: llama-cpp-python is not installed")
        sys.exit(0)

    from local_speculative_runtime.llama_cpp_backend import LlamaCppBackend
    backend = LlamaCppBackend(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        verbose=args.trace
    )

    print("\n--- 2. Prefix Reuse Probe ---")
    prefix = "You are a concise assistant. " * 50
    suffixA = "Say exactly: APPLE."
    suffixB = "Say exactly: BANANA."
    suffixC = "Different context. " * 50 + "Say exactly: CHERRY."

    backend.create_session("prefix-test", "")

    # A1: same prefix + suffix A
    import time
    start = time.perf_counter()
    resA1 = backend.generate(
        session_id="prefix-test",
        prompt_or_suffix=prefix + suffixA,
        max_tokens=args.max_tokens,
        temperature=0.0
    )
    timeA1 = time.perf_counter() - start
    print(f"Turn A1 (prefix + A): {timeA1:.3f}s")
    if not resA1.ok:
        print("Failed to generate A1")
        sys.exit(1)

    # A2: same prefix + suffix B
    start = time.perf_counter()
    resA2 = backend.generate(
        session_id="prefix-test",
        prompt_or_suffix=prefix + suffixB,
        max_tokens=args.max_tokens,
        temperature=0.0
    )
    timeA2 = time.perf_counter() - start
    print(f"Turn A2 (prefix + B): {timeA2:.3f}s")

    # B1: different prefix + suffix C
    start = time.perf_counter()
    resB1 = backend.generate(
        session_id="prefix-test",
        prompt_or_suffix=suffixC,
        max_tokens=args.max_tokens,
        temperature=0.0
    )
    timeB1 = time.perf_counter() - start
    print(f"Turn B1 (diff prefix + C): {timeB1:.3f}s")

    prefix_reuse_status = "unknown"
    if timeA2 < timeA1 * 0.5:
        prefix_reuse_status = "observed"
    elif timeA2 > timeA1 * 0.8:
        prefix_reuse_status = "not_observed"

    print(f"\nprefix_reuse_status: {prefix_reuse_status}")
    if prefix_reuse_status == "not_observed":
        print("Result: llama-cpp-python full state reset occurred, prefix reuse was NOT observed.")
    else:
        print("Result: Prefix reuse observed or unknown.")
        
    print("\nOK: llama.cpp prefix reuse benchmark completed")
if __name__ == "__main__":
    main()
