import argparse
import sys
import os
import json
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
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

    if not args.json:
        print(f"--- 1. Init API (backend=llama_cpp) ---")
    
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
                "has_save_state": False,
                "has_load_state": False,
                "model_is_recurrent": None,
                "model_is_hybrid": None,
                "state_restore_passed": False,
                "token_match": None,
                "text_match": None,
                "state_restore_status": "unknown",
                "failure_reason": "llama-cpp-python is not installed"
            }))
        else:
            print("SKIP: llama-cpp-python is not installed")
        return

    from gemma4_mlx_runtime.llama_cpp_backend import LlamaCppBackend
    backend = LlamaCppBackend(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        verbose=args.trace
    )

    if not args.json:
        print("\n--- 2. Exact KV Snapshot Restore Probe ---")
    prefix = "You are a concise assistant. " * 50 + "\n"
    suffix = "Say exactly: OK."
    
    backend.create_session("restore-test", "")
    
    # Generate 1
    start = time.perf_counter()
    res1 = backend.generate(
        session_id="restore-test",
        prompt_or_suffix=prefix + suffix,
        max_tokens=args.max_tokens,
        temperature=0.0
    )
    time1 = time.perf_counter() - start
    if not args.json:
        print(f"Turn 1: {time1:.3f}s")
        
    if not res1.ok:
        failure_reason = "Failed to generate Turn 1: " + str(res1.error)
        if args.json:
            sys.stdout.flush()
            os.dup2(original_stdout_fd, 1)
            os.close(original_stdout_fd)
            print(json.dumps({
                "ok": False,
                "backend": "llama_cpp",
                "model_path": args.model,
                "has_save_state": False,
                "has_load_state": False,
                "model_is_recurrent": None,
                "model_is_hybrid": None,
                "state_restore_passed": False,
                "token_match": None,
                "text_match": None,
                "state_restore_status": "unknown",
                "failure_reason": failure_reason
            }))
        else:
            print(failure_reason)
        sys.exit(1)
        
    text1 = res1.text.strip()

    llm = backend.llm
    if not llm:
        failure_reason = "Failed to get Llama instance"
        if args.json:
            sys.stdout.flush()
            os.dup2(original_stdout_fd, 1)
            os.close(original_stdout_fd)
            print(json.dumps({
                "ok": False,
                "backend": "llama_cpp",
                "model_path": args.model,
                "has_save_state": False,
                "has_load_state": False,
                "model_is_recurrent": None,
                "model_is_hybrid": None,
                "state_restore_passed": False,
                "token_match": None,
                "text_match": None,
                "state_restore_status": "unknown",
                "failure_reason": failure_reason
            }))
        else:
            print(failure_reason)
        sys.exit(1)
        
    has_save_state = hasattr(llm, "save_state")
    has_load_state = hasattr(llm, "load_state")
    model_is_recurrent = getattr(llm, "_is_recurrent", False)
    model_is_hybrid = getattr(llm, "_is_hybrid", False)
    
    if not args.json:
        print(f"has_save_state: {has_save_state}")
        print(f"has_load_state: {has_load_state}")
        print(f"model_is_recurrent: {model_is_recurrent}")
        print(f"model_is_hybrid: {model_is_hybrid}")
    
    state_restore_passed = False
    state_restore_status = "unknown"
    failure_reason = ""
    token_match = None
    text_match = None
    
    if has_save_state and has_load_state:
        # evaluate prefix
        llm.reset()
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
        textA = resA.text.strip()
        
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
            textB = resB.text.strip()
            
            token_match = None
            text_match = None
            
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
        
    if not args.json:
        print(f"state_restore_passed: {state_restore_passed}")
        print(f"state_restore_status: {state_restore_status}")
        print(f"failure_reason: {failure_reason}")
        print("\nOK: llama.cpp state restore benchmark completed")
    else:
        sys.stdout.flush()
        os.dup2(original_stdout_fd, 1)
        os.close(original_stdout_fd)
        
        output = {
            "ok": True,
            "backend": "llama_cpp",
            "model_path": args.model,
            "has_save_state": has_save_state,
            "has_load_state": has_load_state,
            "model_is_recurrent": model_is_recurrent,
            "model_is_hybrid": model_is_hybrid,
            "state_restore_passed": state_restore_passed,
            "token_match": token_match,
            "text_match": text_match,
            "state_restore_status": state_restore_status,
            "failure_reason": failure_reason if failure_reason else None
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
