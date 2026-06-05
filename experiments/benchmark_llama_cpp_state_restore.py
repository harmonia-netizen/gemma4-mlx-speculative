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

    from gemma4_mlx_runtime.llama_cpp_backend import LlamaCppBackend
    backend = LlamaCppBackend(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        verbose=args.trace
    )

    print("\n--- 2. Exact KV Snapshot Restore Probe ---")
    prefix = "You are a concise assistant. " * 50
    suffix = "Say exactly: OK."
    
    backend.create_session("restore-test", "")
    
    # Generate 1
    import time
    start = time.perf_counter()
    res1 = backend.generate(
        session_id="restore-test",
        prompt_or_suffix=prefix + suffix,
        max_tokens=args.max_tokens,
        temperature=0.0
    )
    time1 = time.perf_counter() - start
    print(f"Turn 1: {time1:.3f}s")
    if not res1.ok:
        print("Failed to generate Turn 1:", res1.error)
        sys.exit(1)
        
    text1 = res1.text.strip()

    # Generate 2 with EXACT SAME PROMPT (which will use the same prefix + suffix)
    # But first, we can try to save state manually if we had access to it.
    # We will simulate a branch and see if it resets by checking the timings.
    # Actually, we don't have direct access to save_state in the api.
    # But we can access the underlying Llama instance:
    llm = backend.llm
    if not llm:
        print("Failed to get Llama instance")
        sys.exit(1)
        
    has_save_state = hasattr(llm, "save_state")
    has_load_state = hasattr(llm, "load_state")
    model_is_recurrent = getattr(llm, "_is_recurrent", False)
    model_is_hybrid = getattr(llm, "_is_hybrid", False)
    
    print(f"has_save_state: {has_save_state}")
    print(f"has_load_state: {has_load_state}")
    print(f"model_is_recurrent: {model_is_recurrent}")
    print(f"model_is_hybrid: {model_is_hybrid}")
    
    state_restore_passed = False
    state_restore_status = "unknown"
    failure_reason = ""
    
    if has_save_state and has_load_state:
        # evaluate prefix
        tokens = llm.tokenize(prefix.encode("utf-8"), add_bos=True)
        llm.eval(tokens)
        
        # save state
        state = llm.save_state()
        
        # generate 1
        resA = backend.generate(
            session_id="restore-test",
            prompt_or_suffix=prefix + "Say exactly: APPLE.",
            max_tokens=args.max_tokens,
            temperature=0.0
        )
        timeA = resA.elapsed_sec
        
        # load state
        try:
            llm.load_state(state)
        except Exception as e:
            state_restore_passed = False
            failure_reason = f"load_state exception: {e}"
            
        if not failure_reason:
            # generate 2
            resB = backend.generate(
                session_id="restore-test",
                prompt_or_suffix=prefix + "Say exactly: BANANA.",
                max_tokens=args.max_tokens,
                temperature=0.0
            )
            timeB = resB.elapsed_sec
            
            # check times
            if timeB > timeA * 0.8:
                state_restore_passed = False
                failure_reason = "Full state reset occurred during generation"
                state_restore_status = "failed_for_tested_model"
            else:
                state_restore_passed = True
                state_restore_status = "passed"
    else:
        state_restore_status = "not_enabled"
        failure_reason = "Missing save_state/load_state APIs"
        
    print(f"state_restore_passed: {state_restore_passed}")
    print(f"state_restore_status: {state_restore_status}")
    print(f"failure_reason: {failure_reason}")
    print("\nOK: llama.cpp state restore benchmark completed")

if __name__ == "__main__":
    main()
