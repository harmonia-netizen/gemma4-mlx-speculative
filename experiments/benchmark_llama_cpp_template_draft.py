import argparse
import time
import json
import os
import sys

from gemma4_mlx_runtime.llama_cpp_backend import LlamaCppBackend

def generate_prefix(repeat_lines: int) -> str:
    lines = []
    for i in range(repeat_lines):
        lines.append(f"[INFO] module=agent step={i:06d} status=ok message=\"synthetic long context line for prefix reuse template draft benchmark\"")
    return "\n".join(lines)

SUFFIX_CASES = [
    {
        "name": "exact_pytest_plan",
        "text": """次の確認手順をbashブロックだけで出してください。
前提:
- pytestの失敗内容を短く確認したい
- gitの差分も確認したい
- 出力は次の3行だけ
- 説明文は不要
- コマンドはこの順番にする:
  1. git status --short
  2. pytest --tb=short
  3. git diff
"""
    },
    {
        "name": "medium_pytest_plan",
        "text": """pytest失敗の原因を安全に確認するため、次に実行する確認手順を3行のbashブロックで出してください。
前提:
- repo=gemma4-mlx-speculative
- destructive commandは禁止
- 3コマンドだけ出す
- 説明文は不要
"""
    },
    {
        "name": "git_status",
        "text": """作業ツリーの状態を確認するため、次に実行すべきコマンドを1つだけ出してください。
説明文は不要。
"""
    }
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--repeat-lines", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--draft-block-size", type=int, default=8)
    parser.add_argument("--template-min-tokens", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--candidate-json", type=str, default="experiments/template_candidates_gguf_qwen.json")
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
            print(json.dumps({"ok": False, "error": "llama-cpp-python is not installed"}))
        else:
            print("SKIP: llama-cpp-python is not installed")
        return

    backend = LlamaCppBackend(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        verbose=False
    )
    # Reinitialize to override logits_all
    backend.llm = llama_cpp.Llama(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        verbose=False,
        logits_all=True
    )

    prefix_str = generate_prefix(args.repeat_lines)
    formatted_prefix = "<|im_start|>user\n" + prefix_str
    
    if not args.json:
        print(f"repeat_lines: {args.repeat_lines}")
        dummy_prompt = formatted_prefix + "\n" + "<|im_end|>\n<|im_start|>assistant\n"
        print(f"prefix_tokens: {len(backend.tokenize(dummy_prompt))}")

    all_results = []
    
    try:
        for run_idx in range(args.runs):
            if not args.json:
                print(f"\n========== run {run_idx+1}/{args.runs} ==========")
            
            # A. baseline
            if not args.json:
                print("--- A. Baseline Greedy (no prefix cache) ---")
            baseline_results = []
            for i, c in enumerate(SUFFIX_CASES):
                if not args.json:
                    print(f"  case {i} ({c['name']}):", end=" ", flush=True)
                
                # Without session
                full_prompt = formatted_prefix + "\n" + c["text"] + "<|im_end|>\n<|im_start|>assistant\n"
                res = backend.generate(None, full_prompt, max_tokens=args.max_tokens, template_min_tokens=0, draft_block_size=0)
                baseline_results.append(res)
                if not args.json:
                    print(f"elapsed={res.elapsed_sec:.3f}s decode={res.metadata.get('decode_sec', 0.0):.3f}s tokens={res.completion_tokens}")
                    print(f"TEXT: {repr(res.text)}")
                    
            # B. prefix reuse greedy
            if not args.json:
                print("--- B. Prefix Reuse Greedy ---")
            backend.create_session("sess_b", formatted_prefix)
            reuse_greedy_results = []
            for i, c in enumerate(SUFFIX_CASES):
                if not args.json:
                    print(f"  case {i} ({c['name']}):", end=" ", flush=True)
                
                suffix_text = "\n" + c["text"] + "<|im_end|>\n<|im_start|>assistant\n"
                res = backend.generate("sess_b", suffix_text, max_tokens=args.max_tokens, template_min_tokens=0, draft_block_size=0)
                reuse_greedy_results.append(res)
                if not args.json:
                    print(f"elapsed(exc_prefix)={res.elapsed_sec:.3f}s suffix_prefill={res.metadata.get('suffix_prefill_sec',0):.3f}s decode={res.metadata.get('decode_sec',0):.3f}s")
                if res.token_ids != baseline_results[i].token_ids:
                    if not args.json:
                        print("MISMATCH in B")
                    raise SystemExit(2)
                    
            # C. prefix reuse template draft
            if not args.json:
                print("--- C. Prefix Reuse Template Draft ---")
            backend.create_session("sess_c", formatted_prefix)
            reuse_draft_results = []
            mismatch_token_match = True
            for i, c in enumerate(SUFFIX_CASES):
                if not args.json:
                    print(f"  case {i} ({c['name']}):", end=" ", flush=True)
                
                suffix_text = "\n" + c["text"] + "<|im_end|>\n<|im_start|>assistant\n"
                res = backend.generate(
                    "sess_c", 
                    suffix_text, 
                    max_tokens=args.max_tokens, 
                    template_min_tokens=args.template_min_tokens, 
                    draft_block_size=args.draft_block_size,
                    trace=args.trace
                )
                reuse_draft_results.append(res)
                if not args.json:
                    print(f"elapsed(exc_prefix)={res.elapsed_sec:.3f}s suffix_prefill={res.metadata.get('suffix_prefill_sec',0):.3f}s decode={res.metadata.get('decode_sec',0):.3f}s accept={res.metadata.get('accepted')}/{res.metadata.get('drafted')}")
                if res.token_ids != baseline_results[i].token_ids:
                    mismatch_token_match = False
                    if not args.json:
                        print("MISMATCH in C")
                    raise SystemExit(2)
                    
            all_results.append({
                "run": run_idx + 1,
                "token_match": True,
                "mismatch_token_match": mismatch_token_match,
                "A_elapsed": sum(r.elapsed_sec for r in baseline_results),
                "B_elapsed": sum(r.elapsed_sec for r in reuse_greedy_results),
                "C_elapsed": sum(r.elapsed_sec for r in reuse_draft_results),
                "C_drafted": sum(r.metadata.get("drafted", 0) for r in reuse_draft_results),
                "C_accepted": sum(r.metadata.get("accepted", 0) for r in reuse_draft_results),
                "C_rejected": sum(r.metadata.get("rejected", 0) for r in reuse_draft_results),
                "B_exact_decode": reuse_greedy_results[0].metadata.get("decode_sec", 0.0),
                "C_exact_decode": reuse_draft_results[0].metadata.get("decode_sec", 0.0),
            })
            
    except SystemExit:
        if args.json:
            sys.stdout.flush()
            os.dup2(original_stdout_fd, 1)
            os.close(original_stdout_fd)
            print(json.dumps({"ok": False, "error": "Token mismatch detected"}))
        return

    if args.json:
        sys.stdout.flush()
        os.dup2(original_stdout_fd, 1)
        os.close(original_stdout_fd)
        
        output = {
            "ok": True,
            "token_match": True,
            "template_draft_enabled": True,
            "runs": all_results
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print("\n--- Speedups ---")
        res = all_results[-1]
        print(f"A baseline total elapsed: {res['A_elapsed']:.3f}s")
        print(f"B amortized total elapsed: {res['B_elapsed']:.3f}s")
        print(f"C amortized total elapsed: {res['C_elapsed']:.3f}s")
        
        print(f"C vs B decode speedup for exact_pytest_plan: {res['B_exact_decode'] / res['C_exact_decode'] if res['C_exact_decode'] > 0 else 0:.3f}x")
        print(f"Total accepted / drafted: {res['C_accepted']} / {res['C_drafted']}")
        print("\nOK: runtime benchmark completed with token match")

if __name__ == "__main__":
    main()
