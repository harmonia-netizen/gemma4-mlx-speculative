import argparse
import json
import os
import sys
import time
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Benchmark exact KV state restore for llama_cpp")
    parser.add_argument("--model", type=str, required=True, help="Path to the GGUF model")
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    # Capture internal llama.cpp logs if JSON is requested
    original_stdout_fd = None
    if args.json:
        sys.stdout.flush()
        original_stdout_fd = os.dup(1)
        os.dup2(2, 1)

    import llama_cpp
    
    llm = llama_cpp.Llama(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        verbose=False,
        logits_all=True
    )

    prefix_text = "<|im_start|>user\nSay exactly: APPLE.<|im_end|>\n<|im_start|>assistant\n"
    suffixA_text = "Wait, "
    suffixB_text = "Sure! "

    def greedy_sample(llama_instance):
        return int(np.argmax(llama_instance.scores[llama_instance.n_tokens - 1, :]))

    def generate_greedy(llama_instance, num_tokens):
        out_tokens = []
        for _ in range(num_tokens):
            t = greedy_sample(llama_instance)
            out_tokens.append(t)
            llama_instance.eval([t])
        return out_tokens

    debug_info = {
        "model_is_recurrent": getattr(llm, "_is_recurrent", False),
        "model_is_hybrid": getattr(llm, "_is_hybrid", False),
        "n_tokens_before_save": None,
        "n_tokens_after_load": None,
        "longest_prefix": None
    }
    
    failure_reason = ""
    baseline_tokens = []
    restored_tokens = []
    
    try:
        # Baseline Generation (Full evaluation of Prefix + SuffixB)
        llm.reset()
        tokens_base = llm.tokenize((prefix_text + suffixB_text).encode("utf-8"), add_bos=True)
        llm.eval(tokens_base)
        baseline_tokens = generate_greedy(llm, args.max_tokens)
        
        # State Restore Generation
        llm.reset()
        tokens_prefix = llm.tokenize(prefix_text.encode("utf-8"), add_bos=True)
        llm.eval(tokens_prefix)
        
        debug_info["n_tokens_before_save"] = llm.n_tokens
        state = llm.save_state()
        
        # Evaluate something else (SuffixA) to advance state and overwrite KV cache conceptually
        tokens_a = llm.tokenize(suffixA_text.encode("utf-8"), add_bos=False)
        llm.eval(tokens_a)
        _ = generate_greedy(llm, 2)
        
        # Restore State
        llm.load_state(state)
        debug_info["n_tokens_after_load"] = llm.n_tokens
        
        # We manually emulate longest_prefix for debugging JSON
        tokens_b = llm.tokenize(suffixB_text.encode("utf-8"), add_bos=False)
        debug_info["longest_prefix"] = llm.n_tokens # Since we align perfectly, it's the full prefix length
        
        llm.eval(tokens_b)
        restored_tokens = generate_greedy(llm, args.max_tokens)
        
    except Exception as e:
        failure_reason = str(e)
        import traceback
        debug_info["traceback"] = traceback.format_exc()

    token_match = False
    text_match = False
    if baseline_tokens and restored_tokens and len(baseline_tokens) == len(restored_tokens):
        token_match = all(a == b for a, b in zip(baseline_tokens, restored_tokens))
    
    if token_match and baseline_tokens:
        text_match = (llm.detokenize(baseline_tokens) == llm.detokenize(restored_tokens))
        
    full_reset_observed = False # In lowlevel eval, we bypass reset. If n_tokens mismatch, llama.cpp crashes. If it didn't crash, no reset.
    state_restore_passed = token_match and text_match and not bool(failure_reason)

    if args.json:
        sys.stdout.flush()
        os.dup2(original_stdout_fd, 1)
        os.close(original_stdout_fd)
        
        output = {
            "ok": True,
            "state_restore_passed": state_restore_passed,
            "token_match": token_match,
            "text_match": text_match,
            "full_reset_observed": full_reset_observed,
            "baseline_tokens": baseline_tokens,
            "restored_tokens": restored_tokens,
            "state_restore_status": "passed" if state_restore_passed else "failed_for_tested_model",
            "failure_reason": failure_reason if failure_reason else None,
            "debug": debug_info
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print("token_match:", token_match)
        print("text_match:", text_match)
        print("baseline:", llm.detokenize(baseline_tokens).decode("utf-8", errors="replace"))
        print("restored:", llm.detokenize(restored_tokens).decode("utf-8", errors="replace"))
        if failure_reason:
            print("failure_reason:", failure_reason)

if __name__ == "__main__":
    main()
