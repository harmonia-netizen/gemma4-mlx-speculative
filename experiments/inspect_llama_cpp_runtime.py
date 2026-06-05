import argparse
import sys
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    args = parser.parse_args()

    try:
        import llama_cpp
    except ImportError:
        print("SKIP: llama-cpp-python not found")
        return

    print("--- 1. Initialize ---")
    llm = llama_cpp.Llama(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        verbose=True
    )

    print("\n--- 2. Inspect Attributes ---")
    print(f"_is_recurrent: {getattr(llm, '_is_recurrent', 'N/A')}")
    print(f"_is_hybrid: {getattr(llm, '_is_hybrid', 'N/A')}")
    print(f"has save_state: {hasattr(llm, 'save_state')}")
    print(f"has load_state: {hasattr(llm, 'load_state')}")
    
    print("\n--- 3. Low-level eval / save / load trace ---")
    prefix = "You are a concise assistant. " * 50
    suffixA = "Say exactly: APPLE."
    suffixB = "Say exactly: BANANA."

    tokens_prefix = llm.tokenize(prefix.encode("utf-8"), add_bos=True)
    tokens_A = llm.tokenize(suffixA.encode("utf-8"), add_bos=False)
    tokens_B = llm.tokenize(suffixB.encode("utf-8"), add_bos=False)

    print("Evaluating prefix...")
    start = time.perf_counter()
    llm.eval(tokens_prefix)
    eval_prefix_sec = time.perf_counter() - start
    print(f"Prefix eval time: {eval_prefix_sec:.3f}s, n_tokens={llm.n_tokens}")

    print("Saving state...")
    state = llm.save_state()
    print("State saved.")

    print("Evaluating suffix A via eval + sample...")
    start = time.perf_counter()
    
    # Eval suffix A
    llm.eval(tokens_A)
    # Just sample one token
    token = llm.sample(temp=0.0)
    suffixA_sec = time.perf_counter() - start
    print(f"Suffix A time: {suffixA_sec:.3f}s, n_tokens={llm.n_tokens}")
    print(f"Sampled A: {llm.detokenize([token]).decode('utf-8', errors='replace')}")

    print("Loading state...")
    llm.load_state(state)
    print(f"State loaded. n_tokens={llm.n_tokens}")

    print("Evaluating suffix B via eval + sample...")
    start = time.perf_counter()
    llm.eval(tokens_B)
    token = llm.sample(temp=0.0)
    suffixB_sec = time.perf_counter() - start
    print(f"Suffix B time: {suffixB_sec:.3f}s, n_tokens={llm.n_tokens}")
    print(f"Sampled B: {llm.detokenize([token]).decode('utf-8', errors='replace')}")

    print("\n--- Summary ---")
    print(f"Prefix time: {eval_prefix_sec:.3f}s")
    print(f"Suffix A time: {suffixA_sec:.3f}s")
    print(f"Suffix B time: {suffixB_sec:.3f}s")

if __name__ == "__main__":
    main()
