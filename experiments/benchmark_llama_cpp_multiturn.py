import argparse
import sys
import os
import time
from gemma4_mlx_runtime import SessionCacheAPI

def main():
    parser = argparse.ArgumentParser(description="Llama.cpp backend multi-turn benchmark")
    parser.add_argument("--model", type=str, required=True, help="Path to GGUF model")
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found at {args.model}")
        sys.exit(1)

    print(f"--- 1. Init API (backend=llama_cpp) ---")
    api = SessionCacheAPI.load(
        model_path=args.model,
        backend="llama_cpp",
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers
    )
    
    backend_stats = api.stats()
    if not backend_stats.get("available", False):
        print("SKIP: llama-cpp-python is not installed")
        sys.exit(0)

    print(f"model path: {args.model}")

    session_id = "gguf-multiturn-session"
    prefix_text = "You are a concise local runtime assistant. Answer briefly."

    print("\n--- 2. Create Session ---")
    create_res = api.create_session(session_id, prefix_text)
    print(f"create result: {create_res}")
    if not create_res["ok"]:
        print("Failed to create session.")
        sys.exit(1)

    tasks = [
        "Return exactly: OK",
        "Say the word READY and nothing else.",
        "List two safe shell inspection commands.",
        "Summarize this runtime backend in one sentence."
    ]

    print("\n--- 3. Generate Tasks ---")
    for i, task in enumerate(tasks, 1):
        print(f"\nTask {i}: {task}")
        gen_res = api.generate(session_id, task, max_tokens=args.max_tokens, trace=args.trace, temperature=args.temperature)
        print(f"ok: {gen_res['ok']}")
        if not gen_res['ok']:
            print(f"Failed to generate: {gen_res.get('error')}")
            sys.exit(1)
            
        text = gen_res.get("text", "").strip()
        print(f"text snippet: {repr(text)}")
        if not text:
            print("ERROR: Empty text returned.")
            sys.exit(1)
            
        print(f"prompt_tokens: {gen_res.get('prompt_tokens')} | completion_tokens: {gen_res.get('completion_tokens')}")
        print(f"elapsed_sec: {gen_res.get('elapsed_sec', 0.0):.3f}s")
        if args.trace:
            print(f"metadata: {gen_res.get('metadata')}")

    print("\n--- 4. Stats before clear ---")
    print(f"stats: {api.stats()}")

    print("\n--- 5. Clear Session ---")
    clear_res = api.clear_session(session_id)
    print(f"clear result: {clear_res}")

    print("\n--- 6. Final Stats ---")
    final_stats = api.stats()
    print(f"stats: {final_stats}")
    if final_stats.get("sessions", -1) != 0:
        print("ERROR: sessions count is not 0 after clear.")
        sys.exit(1)
    
    try:
        from experiments.session_cache_memory import get_memory_stats, format_memory_stats
        print(format_memory_stats(get_memory_stats()))
    except ImportError:
        pass

    print("\nOK: llama.cpp multiturn benchmark completed")

if __name__ == "__main__":
    main()
