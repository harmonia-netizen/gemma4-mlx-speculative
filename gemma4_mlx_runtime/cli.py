import argparse
import json
import sys
import os
import contextlib

from gemma4_mlx_runtime.session_cache import SessionCacheAPI

GGUF_CANDIDATE_PRESETS = {
    "qwen": "experiments/template_candidates_gguf_qwen.json"
}

def resolve_candidate_json(backend: str, model_type: str, candidate_json: str) -> str:
    """Resolves and validates the candidate JSON path based on the backend."""
    if backend == "mlx":
        if model_type is not None:
            raise ValueError("Error: --model-type cannot be specified with backend='mlx'")
        if candidate_json is not None:
            raise ValueError("Error: --candidate-json cannot be specified with backend='mlx'")
        return "experiments/template_candidates.json"
    
    elif backend in ["llama_cpp", "gguf"]:
        if model_type is None and candidate_json is None:
            raise ValueError("Error: Either --model-type or --candidate-json must be specified for GGUF backend")
        if model_type is not None and candidate_json is not None:
            raise ValueError("Error: Cannot specify both --model-type and --candidate-json")
            
        if model_type is not None:
            if model_type not in GGUF_CANDIDATE_PRESETS:
                raise ValueError(f"Error: Unknown model-type '{model_type}'. Available: {list(GGUF_CANDIDATE_PRESETS.keys())}")
            return GGUF_CANDIDATE_PRESETS[model_type]
            
        return candidate_json
    
    raise ValueError(f"Unknown backend: {backend}")

@contextlib.contextmanager
def redirect_stdout_to_stderr():
    """Temporarily redirect stdout to stderr. Useful for hiding C++ logs when outputting JSON."""
    old_stdout = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.close(old_stdout)

def main():
    parser = argparse.ArgumentParser(description="Gemma4 MLX / GGUF Speculative Runtime CLI")
    parser.add_argument("--backend", type=str, required=True, choices=["mlx", "llama_cpp", "gguf"], help="Backend to use")
    parser.add_argument("--model", type=str, required=True, help="Path or repo id for the model")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text. If not provided, reads from stdin.")
    parser.add_argument("--prefix-file", type=str, default=None, help="Optional text file to use as prefix for create_session.")
    parser.add_argument("--session-id", type=str, default="cli-session", help="Session ID (default: cli-session)")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum tokens to generate (default: 128)")
    parser.add_argument("--draft-block-size", type=int, default=8, help="Draft block size (default: 8)")
    parser.add_argument("--template-min-tokens", type=int, default=1, help="Template minimum tokens (default: 1)")
    parser.add_argument("--json", action="store_true", help="Output only JSON to stdout")
    parser.add_argument("--trace", action="store_true", help="Enable generation trace")
    
    # GGUF Specific
    parser.add_argument("--model-type", type=str, default=None, help="Preset model type for GGUF candidates (e.g. 'qwen')")
    parser.add_argument("--candidate-json", type=str, default=None, help="Explicit path to candidate JSON file")
    
    args = parser.parse_args()
    
    try:
        candidate_json_path = resolve_candidate_json(args.backend, args.model_type, args.candidate_json)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(2)
        
    prompt = args.prompt
    if prompt is None:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        else:
            print("Error: No prompt provided and stdin is empty", file=sys.stderr)
            sys.exit(1)
            
    prefix_text = ""
    if args.prefix_file:
        try:
            with open(args.prefix_file, "r") as f:
                prefix_text = f.read()
        except Exception as e:
            print(f"Error reading prefix-file: {e}", file=sys.stderr)
            sys.exit(1)

    result_json = {
        "ok": False,
        "backend": args.backend,
        "model": args.model,
        "session_id": args.session_id,
        "text": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "elapsed_sec": 0,
        "metadata": {},
        "create_session": {},
        "clear_session": {},
        "error": None
    }
    
    try:
        # Load API
        with contextlib.ExitStack() as stack:
            if args.json:
                stack.enter_context(redirect_stdout_to_stderr())
            
            if args.backend == "mlx":
                api = SessionCacheAPI.load(model_path=args.model, backend="mlx")
            else:
                api = SessionCacheAPI.load(model_path=args.model, backend="llama_cpp", candidate_json_path=candidate_json_path)
            
            # Create Session
            create_res = api.create_session(args.session_id, prefix_text)
            result_json["create_session"] = create_res
            
            # Generate
            gen_res = api.generate(
                session_id=args.session_id,
                suffix_text=prompt,
                max_tokens=args.max_tokens,
                draft_block_size=args.draft_block_size,
                template_min_tokens=args.template_min_tokens,
                trace=args.trace
            )
            
            result_json["ok"] = gen_res.get("ok", False)
            result_json["text"] = gen_res.get("text")
            result_json["metadata"] = gen_res.get("metadata", {})
            result_json["prompt_tokens"] = gen_res.get("prompt_tokens", 0)
            result_json["completion_tokens"] = gen_res.get("completion_tokens", 0)
            result_json["elapsed_sec"] = gen_res.get("elapsed_sec", 0)
            result_json["error"] = gen_res.get("error")
            
            # Clear Session
            clear_res = api.clear_session(args.session_id)
            result_json["clear_session"] = clear_res
            
    except Exception as e:
        result_json["error"] = str(e)
        if not args.json:
            print(f"Runtime error: {e}", file=sys.stderr)

    if args.json:
        print(json.dumps(result_json, ensure_ascii=False))
    else:
        if result_json["ok"]:
            print("=" * 40)
            print("Completion:")
            print("-" * 40)
            print(result_json["text"])
            print("=" * 40)
            print("Metadata:")
            print(f"Backend: {result_json['backend']}")
            print(f"Elapsed: {result_json['elapsed_sec']:.3f}s")
            print(f"Tokens: Prompt {result_json['prompt_tokens']} | Completion {result_json['completion_tokens']}")
            md = result_json["metadata"]
            if "C_vs_B_decode_speedup" in md:
                print(f"Draft Speedup: {md['C_vs_B_decode_speedup']:.2f}x")
        else:
            print(f"Failed: {result_json['error']}")

if __name__ == "__main__":
    main()
