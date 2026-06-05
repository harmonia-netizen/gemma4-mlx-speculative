import argparse
import sys
import os
import time
from gemma4_mlx_runtime import SessionCacheAPI

def main():
    parser = argparse.ArgumentParser(description="Llama.cpp backend smoke benchmark")
    parser.add_argument("--model", type=str, required=True, help="Path to GGUF model")
    parser.add_argument("--prompt", type=str, default="Return exactly: OK")
    parser.add_argument("--prefix", type=str, default="You are a concise assistant.")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--session-id", type=str, default="gguf-session")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found at {args.model}")
        sys.exit(1)

    print(f"--- 1. Init API (backend=llama_cpp) ---")
    api = SessionCacheAPI.load(
        model_path=args.model,
        backend="llama_cpp",
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        verbose=args.verbose
    )
    
    backend_stats = api.stats()
    if not backend_stats.get("available", False):
        print("SKIP: llama-cpp-python is not installed")
        sys.exit(0)

    print(f"model path: {args.model}")
    print(f"capabilities: {backend_stats.get('capabilities', {})}")

    print("\n--- 2. Create Session ---")
    create_res = api.create_session(args.session_id, args.prefix)
    print(f"create result: {create_res}")
    if not create_res["ok"]:
        print("Failed to create session.")
        sys.exit(1)

    print("\n--- 3. Generate ---")
    gen_res = api.generate(args.session_id, args.prompt, max_tokens=args.max_tokens, trace=args.trace, temperature=args.temperature)
    print(f"ok: {gen_res['ok']}")
    print(f"text snippet: {repr(gen_res.get('text', ''))}")
    print(f"prompt_tokens: {gen_res.get('prompt_tokens')}")
    print(f"completion_tokens: {gen_res.get('completion_tokens')}")
    print(f"elapsed_sec: {gen_res.get('elapsed_sec', 0.0):.3f}s")
    print(f"metadata: {gen_res.get('metadata')}")
    
    print("\n--- 4. Stats before clear ---")
    print(f"stats: {api.stats()}")

    print("\n--- 5. Clear Session ---")
    clear_res = api.clear_session(args.session_id)
    print(f"clear result: {clear_res}")

    print("\n--- 6. Final Stats ---")
    print(f"stats: {api.stats()}")
    
    try:
        from experiments.session_cache_memory import get_memory_stats, format_memory_stats
        print(format_memory_stats(get_memory_stats()))
    except ImportError:
        pass

    print("\nOK: llama.cpp backend benchmark completed")

if __name__ == "__main__":
    main()
