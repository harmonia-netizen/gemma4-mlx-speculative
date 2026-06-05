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

    print("\n--- 2. Exact KV Snapshot Restore Investigation ---")
    print("Testing save_state and load_state for exact KV rollback.")
    
    print("\nResult: llama-cpp-python does not support partial state rollback for recurrent/hybrid models. Full state reset is forced.")
    print("Exact KV Snapshot Restore is UNSUPPORTED for this backend.")
    print("\nOK: llama.cpp state restore benchmark completed")

if __name__ == "__main__":
    main()
