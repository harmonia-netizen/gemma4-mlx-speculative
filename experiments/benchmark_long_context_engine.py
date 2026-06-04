import argparse
import statistics
import time
from pathlib import Path
import importlib

import template_draft_engine as engine

d = importlib.import_module("mlx_vlm.generate.dispatch")

def generate_long_prompt(repeat_lines: int) -> str:
    lines = []
    for i in range(repeat_lines):
        lines.append(f"[INFO] module=agent step={i:06d} status=ok message=\"synthetic long context line for prefill benchmark\"")
    
    dummy_log = "\n".join(lines)
    
    instructions = """
あなたはローカル常駐エージェントです。
次の確認手順をbashブロックだけで出してください。
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
    return dummy_log + "\n" + instructions


def median(values):
    return statistics.median(values) if values else float("nan")


def summarize(name, results):
    decode_secs = [r.decode_sec for r in results]
    tok_s = [r.tok_s for r in results]
    elapsed_secs = [r.elapsed_sec for r in results]
    prefill_secs = [r.prefill_sec for r in results]

    print(f"========== summary: {name} ==========")
    print("runs:", len(results))
    print("tokens:", [len(r.token_ids) for r in results])
    
    med_elapsed = median(elapsed_secs)
    med_prefill = median(prefill_secs)
    med_decode = median(decode_secs)
    
    print("median_elapsed_sec:", f"{med_elapsed:.6f}")
    print("median_prefill_sec:", f"{med_prefill:.6f}")
    print("median_decode_sec:", f"{med_decode:.6f}")
    print("median_decode_tok_s:", f"{median(tok_s):.3f}")

    drafted = sum(getattr(r, "drafted", 0) for r in results)
    accepted = sum(getattr(r, "accepted", 0) for r in results)
    rejected = sum(getattr(r, "rejected", 0) for r in results)

    if drafted:
        print("accepted:", accepted, "/", drafted, f"({accepted / drafted:.1%})")
        print("rejected:", rejected)

    return {
        "median_elapsed_sec": med_elapsed,
        "median_prefill_sec": med_prefill,
        "median_decode_sec": med_decode,
        "accepted": accepted,
        "drafted": drafted,
    }


def run_once(target_model, tokenizer, target_prompt, user_prompt, stop_ids, args):
    greedy = engine.run_target_greedy(
        target_model,
        tokenizer,
        target_prompt,
        stop_ids,
        args.max_tokens,
        args.max_kv_size,
    )

    template = engine.run_template_draft(
        target_model,
        tokenizer,
        target_prompt,
        user_prompt,
        stop_ids,
        args.max_tokens,
        args.draft_block_size,
        args.template_min_tokens,
        args.max_kv_size,
        trace_template=args.trace_template,
    )

    if template.token_ids != greedy.token_ids:
        print("MISMATCH")
        print("greedy ids:", greedy.token_ids)
        print("template ids:", template.token_ids)
        print("greedy text:", greedy.text)
        print("template text:", template.text)
        raise SystemExit(2)

    return greedy, template


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--repeat-lines", type=int, default=2000)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--draft-block-size", type=int, default=8)
    parser.add_argument("--template-min-tokens", type=int, default=1)
    parser.add_argument("--max-kv-size", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--trace-template", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.draft_block_size <= 0:
        raise ValueError("--draft-block-size must be positive")
    if args.template_min_tokens < 0:
        raise ValueError("--template-min-tokens must be >= 0")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be >= 0")
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if args.max_kv_size is not None and args.max_kv_size <= 0:
        raise ValueError("--max-kv-size must be positive")

    model_path = args.model or engine.DEFAULT_TARGET_MODEL_PATH

    print("loading target:", model_path)
    target_model, processor = d.load(model_path)

    tokenizer = getattr(processor, "tokenizer", processor)
    stop_ids = engine.build_stop_ids(tokenizer)

    user_prompt = generate_long_prompt(args.repeat_lines)
    target_prompt = engine.format_prompt(processor, user_prompt)
    
    prompt_tokens = len(tokenizer.encode(target_prompt))
    print(f"repeat_lines: {args.repeat_lines}")
    print(f"prompt_tokens: {prompt_tokens}")

    for i in range(args.warmup_runs):
        print(f"warmup {i + 1}/{args.warmup_runs}")
        run_once(target_model, tokenizer, target_prompt, user_prompt, stop_ids, args)

    greedy_results = []
    template_results = []

    for i in range(args.runs):
        print(f"run {i + 1}/{args.runs}")
        greedy, template = run_once(target_model, tokenizer, target_prompt, user_prompt, stop_ids, args)
        greedy_results.append(greedy)
        template_results.append(template)

        print(
            "  greedy elapsed=", f"{greedy.elapsed_sec:.3f}s",
            "(prefill=", f"{greedy.prefill_sec:.3f}s",
            "decode=", f"{greedy.decode_sec:.3f}s)",
            "| template elapsed=", f"{template.elapsed_sec:.3f}s",
            "(prefill=", f"{template.prefill_sec:.3f}s",
            "decode=", f"{template.decode_sec:.3f}s)",
            "accept=", f"{template.accepted}/{template.drafted}"
        )

    print("")
    g_sum = summarize("target greedy", greedy_results)
    t_sum = summarize("template draft engine", template_results)

    decode_speedup = (
        g_sum["median_decode_sec"] / t_sum["median_decode_sec"]
        if t_sum["median_decode_sec"] > 0
        else float("inf")
    )
    
    elapsed_speedup = (
        g_sum["median_elapsed_sec"] / t_sum["median_elapsed_sec"]
        if t_sum["median_elapsed_sec"] > 0
        else float("inf")
    )

    g_prefill_share = (g_sum["median_prefill_sec"] / g_sum["median_elapsed_sec"]) * 100 if g_sum["median_elapsed_sec"] > 0 else 0
    t_prefill_share = (t_sum["median_prefill_sec"] / t_sum["median_elapsed_sec"]) * 100 if t_sum["median_elapsed_sec"] > 0 else 0

    print("========== final speedup ==========")
    print(f"decode_sec_speedup: {decode_speedup:.3f}x")
    print(f"elapsed_sec_speedup: {elapsed_speedup:.3f}x")
    print(f"greedy prefill_share: {g_prefill_share:.1f}%")
    print(f"template prefill_share: {t_prefill_share:.1f}%")

    print("\nOK: long-context benchmark completed with token match")


if __name__ == "__main__":
    main()
