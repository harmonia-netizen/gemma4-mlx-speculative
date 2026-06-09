import argparse
import time
import json

def check_quality(text, expected_type):
    if not text or not text.strip():
        return False, "Empty output"
    
    if "own own own" in text:
        return False, "Found 'own own own'"
    if "//////////" in text:
        return False, "Found '//////////'"
    if "<|channel>thought" in text:
        return False, "Found channel tokens"
        
    text_lower = text.lower()
    if expected_type == "json":
        if "{" not in text or "}" not in text:
            return False, "Not a valid JSON structure"
        if "\"ok\"" not in text.replace("'", "\"") and "ok" not in text:
            return False, "Missing 'ok' key"
        if "hello" not in text_lower:
            return False, "Missing 'hello'"
    elif expected_type == "code":
        if "def " not in text and "list" not in text_lower:
            return False, "Not python code"
    elif expected_type == "commands":
        if "git status" not in text or "pytest" not in text or "git diff" not in text:
            return False, "Missing required bash commands"
            
    return True, ""

def build_long_prefix(base_sentence, target_tokens, backend):
    prefix = "System: The following is a long background text.\n"
    current_tokens = len(backend.tokenize(prefix))
    
    base_tokens = len(backend.tokenize(base_sentence))
    
    needed_tokens = target_tokens - current_tokens
    if needed_tokens > 0 and base_tokens > 0:
        repeat_count = int(needed_tokens / base_tokens)
        prefix += base_sentence * repeat_count
        
    return prefix

def format_table(results):
    if not results:
        return ""
    keys = list(results[0].keys())
    widths = {k: len(str(k)) for k in keys}
    for row in results:
        for k in keys:
            widths[k] = max(widths[k], len(str(row.get(k, ""))))
            
    header = " | ".join(str(k).ljust(widths[k]) for k in keys)
    separator = "-+-".join("-" * widths[k] for k in keys)
    
    lines = [header, separator]
    for row in results:
        lines.append(" | ".join(str(row.get(k, "")).ljust(widths[k]) for k in keys))
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--candidate-json", type=str, default="experiments/template_candidates_gguf_gemma4.json")
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--draft-block-size", type=int, default=12)
    parser.add_argument("--template-min-tokens", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = []
    
    cases = [
        {"name": "commands", "type": "commands", "prompt": "pytestが失敗しました。失敗内容を短く確認し、git差分も見るためのbashコマンドだけを3行で出してください。"},
        {"name": "code", "type": "code", "prompt": "Pythonでリストの重複を順序を保って削除する関数を書いてください。説明は短くしてください。"},
        {"name": "json", "type": "json", "prompt": "次の条件を守ってください。説明なし。出力は JSON だけ。キーは ok と message。message は hello。"},
        {"name": "japanese_summary", "type": "text", "prompt": "次の仕様変更を3点で要約してください。入力検証を追加し、失敗時は安全に停止し、ログには原因を短く残す。"}
    ]
    
    long_cases = [
        {"name": "long_input_short", "target_percent": 0.20},
        {"name": "long_input_medium", "target_percent": 0.50},
        {"name": "long_input_near_limit", "target_percent": 0.90},
        {"name": "long_input_over_limit", "target_percent": 1.10}
    ]
    
    try:
        import llama_cpp
    except ImportError:
        print("llama-cpp-python is required")
        return

    # A. direct_llama_cpp
    if not args.json:
        print("Loading direct llama_cpp model...")
    llm = llama_cpp.Llama(model_path=args.model, n_ctx=args.n_ctx, n_gpu_layers=-1, verbose=False)
    
    for case in cases:
        prompt = f"User: {case['prompt']}\n\nAssistant:"
        t0 = time.time()
        try:
            res = llm(prompt, max_tokens=args.max_tokens, temperature=0.0, echo=False)
            text = res["choices"][0]["text"]
            t1 = time.time()
            usage = res["usage"]
            ok = True
            error = None
        except Exception as e:
            text = ""
            t1 = time.time()
            usage = {"prompt_tokens": 0, "completion_tokens": 0}
            ok = False
            error = str(e)
            
        q_pass, q_reason = check_quality(text, case["type"])
        
        notes = "Baseline. LSR formatting/filtering absence may fail quality check."
        
        results.append({
            "case_name": case["name"],
            "mode": "direct_llama_cpp",
            "ok": ok,
            "error": error,
            "prefix_tokens": 0,
            "suffix_tokens": usage.get("prompt_tokens", 0),
            "total_input_tokens": usage.get("prompt_tokens", 0),
            "max_tokens": args.max_tokens,
            "n_ctx": args.n_ctx,
            "output_tokens": usage.get("completion_tokens", 0),
            "elapsed_sec": t1 - t0,
            "tokens_per_sec": usage.get("completion_tokens", 0) / (t1 - t0) if (t1 - t0) > 0 and ok else 0.0,
            "drafted": 0,
            "accepted": 0,
            "rejected": 0,
            "template_draft_enabled": False,
            "quality_pass": q_pass,
            "quality_fail_reasons": q_reason,
            "forbidden_marker_found": not q_pass and ("own" in q_reason or "channel" in q_reason or "///" in q_reason),
            "context_limit_status": "ok",
            "preflight_status": "not_implemented",
            "comparison_notes": notes
        })

    del llm

    from local_speculative_runtime.llama_cpp_backend import LlamaCppBackend
    
    if not args.json:
        print("Loading local_speculative_runtime backend...")
        
    backend = LlamaCppBackend(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=-1,
        verbose=False,
        candidate_json_path=args.candidate_json
    )
    
    modes = [
        {"name": "lsr_low_level_no_draft", "bs": 0, "mt": 0},
        {"name": "lsr_low_level_template_draft", "bs": args.draft_block_size, "mt": args.template_min_tokens}
    ]
    
    for mode in modes:
        for case in cases:
            prompt = case['prompt']
            
            session_id = f"{mode['name']}_{case['name']}"
            t0 = time.time()
            try:
                c_res = backend.create_session(session_id, prefix_text="")
                if not c_res.get("ok"):
                    raise Exception(c_res.get("error", "create_session failed"))
                    
                prefix_tokens = c_res.get("prefix_tokens", 0)
                    
                g_res = backend.generate(
                    session_id=session_id,
                    prompt_or_suffix=prompt,
                    max_tokens=args.max_tokens,
                    draft_block_size=mode["bs"],
                    template_min_tokens=mode["mt"],
                    temperature=0.0
                )
                t1 = time.time()
                ok = g_res.ok
                text = g_res.text
                error = g_res.error
                prompt_tokens = g_res.prompt_tokens
                completion_tokens = g_res.completion_tokens
                md = g_res.metadata
            except Exception as e:
                t1 = time.time()
                ok = False
                text = ""
                error = str(e)
                prefix_tokens = 0
                prompt_tokens = 0
                completion_tokens = 0
                md = {}
                
            total_input = (prefix_tokens or 0) + (prompt_tokens or 0)
            q_pass, q_reason = check_quality(text, case["type"])
            
            results.append({
                "case_name": case["name"],
                "mode": mode["name"],
                "ok": ok,
                "error": error,
                "prefix_tokens": prefix_tokens or 0,
                "suffix_tokens": prompt_tokens or 0,
                "total_input_tokens": total_input,
                "max_tokens": args.max_tokens,
                "n_ctx": args.n_ctx,
                "output_tokens": completion_tokens or 0,
                "elapsed_sec": t1 - t0,
                "tokens_per_sec": completion_tokens / (t1 - t0) if (t1 - t0) > 0 and ok else 0.0,
                "drafted": md.get("drafted", 0),
                "accepted": md.get("accepted", 0),
                "rejected": md.get("rejected", 0),
                "template_draft_enabled": md.get("template_verify_enabled", False),
                "quality_pass": q_pass,
                "quality_fail_reasons": q_reason,
                "forbidden_marker_found": not q_pass and ("own" in q_reason or "channel" in q_reason or "///" in q_reason),
                "context_limit_status": "ok",
                "preflight_status": "not_implemented",
                "comparison_notes": "LSR managed format."
            })
            backend.clear_session(session_id)
            
    # Long Input Tests
    base_sentence = "The quick brown fox jumps over the lazy dog. "
    for l_case in long_cases:
        target_tokens = int(args.n_ctx * l_case["target_percent"])
        prefix = build_long_prefix(base_sentence, target_tokens, backend)
        
        prompt = "User: What is the main subject?\n\nAssistant:"
        
        session_id = f"lsr_long_{l_case['name']}"
        t0 = time.time()
        
        prefix_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        ok = False
        error = None
        text = ""
        
        try:
            c_res = backend.create_session(session_id, prefix_text=prefix)
            if not c_res.get("ok"):
                error = c_res.get("error", "create_session failed")
            else:
                prefix_tokens = c_res.get("prefix_tokens", 0)
                g_res = backend.generate(
                    session_id=session_id,
                    prompt_or_suffix=prompt,
                    max_tokens=args.max_tokens,
                    draft_block_size=args.draft_block_size,
                    template_min_tokens=args.template_min_tokens,
                    temperature=0.0
                )
                ok = g_res.ok
                text = g_res.text
                error = g_res.error
                prompt_tokens = g_res.prompt_tokens
                completion_tokens = g_res.completion_tokens
        except Exception as e:
            error = str(e)
            
        t1 = time.time()
        
        total_input = (prefix_tokens or 0) + (prompt_tokens or 0)
        if total_input == 0 and not ok:
            total_input = target_tokens
        
        cl_status = "ok"
        if not ok and error and ("ValueError" in error or "broadcast" in error or "eval" in error or "decode" in error):
            cl_status = "decode_error_after_start"
            
        results.append({
            "case_name": l_case["name"],
            "mode": "lsr_low_level_template_draft",
            "ok": ok,
            "error": error,
            "prefix_tokens": prefix_tokens or 0,
            "suffix_tokens": prompt_tokens or 0,
            "total_input_tokens": total_input,
            "max_tokens": args.max_tokens,
            "n_ctx": args.n_ctx,
            "output_tokens": completion_tokens or 0,
            "elapsed_sec": t1 - t0,
            "tokens_per_sec": completion_tokens / (t1 - t0) if (t1 - t0) > 0 and ok else 0.0,
            "drafted": 0,
            "accepted": 0,
            "rejected": 0,
            "template_draft_enabled": True,
            "quality_pass": len(text) > 0 if ok else False,
            "quality_fail_reasons": "Failed" if not ok else "",
            "forbidden_marker_found": False,
            "context_limit_status": cl_status,
            "preflight_status": "not_implemented",
            "comparison_notes": f"Target {int(l_case['target_percent']*100)}% of n_ctx"
        })
        backend.clear_session(session_id)
        
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print("\n=== Validation Results ===")
        print(format_table(results))

if __name__ == "__main__":
    main()
