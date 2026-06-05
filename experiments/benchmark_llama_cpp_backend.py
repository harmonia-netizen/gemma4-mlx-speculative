import argparse
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to GGUF model")
    parser.add_argument("--prompt", type=str, default="Return exactly: OK")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--session-id", type=str, default="gguf-session")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model path {args.model} does not exist.")
        sys.exit(1)

    try:
        import llama_cpp
    except ImportError:
        print("SKIP: llama-cpp-python is not installed. Benchmark cannot run.")
        sys.exit(0)

    from gemma4_mlx_runtime import SessionCacheAPI

    if args.trace:
        print(f"Loading GGUF model from {args.model} via LlamaCppBackend...")

    api = SessionCacheAPI.load(
        model_path=args.model,
        backend="llama_cpp",
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        temperature=args.temperature,
        verbose=args.trace
    )

    prefix_text = "You are a concise assistant."
    
    if args.trace:
        print("Creating session...")
    
    res_create = api.create_session(args.session_id, prefix_text)
    if not res_create.get("ok"):
        print(f"Failed to create session: {res_create}")
        sys.exit(1)

    if args.trace:
        print("Generating...")
        
    res_gen = api.generate(args.session_id, args.prompt, max_tokens=args.max_tokens)
    
    if args.trace:
        print("Getting stats...")
    
    stats = api.stats()
    
    if args.trace:
        print("Clearing session...")
    
    api.clear_session(args.session_id)

    if args.json:
        out = {
            "create": res_create,
            "generate": res_gen,
            "stats": stats
        }
        print(json.dumps(out, indent=2))
    else:
        print("\n=== Result ===")
        print(f"Backend: {res_gen.get('backend')}")
        print(f"Text: {repr(res_gen.get('text'))}")
        print(f"Prompt Tokens: {res_gen.get('suffix_tokens')}")
        print(f"Completion Tokens: len({len(res_gen.get('token_ids', []))})")
        print(f"Elapsed Sec: {res_gen.get('elapsed_sec', 0.0):.3f}s")
        print(f"Stats Sessions: {stats.get('sessions')}")

    print("\nOK: llama.cpp backend benchmark completed")

if __name__ == "__main__":
    main()
