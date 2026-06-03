import argparse
import importlib.util
import statistics
from pathlib import Path

V8_PATH = Path(__file__).with_name("run_gemma4_template_draft_v8.py")


def load_v8_module():
    spec = importlib.util.spec_from_file_location("template_draft_v8", V8_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    print("median_elapsed_sec:", f"{median(elapsed_secs):.6f}")
    print("median_prefill_sec:", f"{median(prefill_secs):.6f}")
    print("median_decode_sec:", f"{median(decode_secs):.6f}")
    print("median_decode_tok_s:", f"{median(tok_s):.3f}")

    drafted = sum(getattr(r, "drafted", 0) for r in results)
    accepted = sum(getattr(r, "accepted", 0) for r in results)
    rejected = sum(getattr(r, "rejected", 0) for r in results)

    if drafted:
        print("accepted:", accepted, "/", drafted, f"({accepted / drafted:.1%})")
        print("rejected:", rejected)


def run_once(v8, target_model, tokenizer, target_prompt, user_prompt, stop_ids, args):
    greedy = v8.run_target_greedy(
        target_model,
        tokenizer,
        target_prompt,
        stop_ids,
        args.max_tokens,
        args.max_kv_size,
    )

    template = v8.run_speculative(
        target_model,
        tokenizer,
        target_prompt,
        user_prompt,
        stop_ids,
        args.max_tokens,
        args.draft_block_size,
        args.max_kv_size,
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
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--draft-block-size", type=int, default=4)
    parser.add_argument("--max-kv-size", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.draft_block_size <= 0:
        raise ValueError("--draft-block-size must be positive")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be >= 0")
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if args.max_kv_size is not None and args.max_kv_size <= 0:
        raise ValueError("--max-kv-size must be positive")

    v8 = load_v8_module()

    model_path = args.model or v8.DEFAULT_TARGET_MODEL_PATH
    user_prompt = args.prompt or v8.DEFAULT_PROMPT

    print("loading target:", model_path)
    target_model, processor = v8.d.load(model_path)

    tokenizer = getattr(processor, "tokenizer", processor)
    target_prompt = v8.format_prompt(processor, user_prompt)

    print("prompt_tokens:", len(tokenizer.encode(target_prompt)))

    stop_ids = v8.build_stop_ids(tokenizer)
    print("stop_ids:", sorted(stop_ids))

    for i in range(args.warmup_runs):
        print(f"warmup {i + 1}/{args.warmup_runs}")
        run_once(v8, target_model, tokenizer, target_prompt, user_prompt, stop_ids, args)

    greedy_results = []
    template_results = []

    for i in range(args.runs):
        print(f"run {i + 1}/{args.runs}")
        greedy, template = run_once(
            v8,
            target_model,
            tokenizer,
            target_prompt,
            user_prompt,
            stop_ids,
            args,
        )
        greedy_results.append(greedy)
        template_results.append(template)

        print(
            "  greedy_decode_sec=",
            f"{greedy.decode_sec:.6f}",
            "template_decode_sec=",
            f"{template.decode_sec:.6f}",
            "template_accept=",
            f"{template.accepted}/{template.drafted}",
        )

    summarize("target greedy", greedy_results)
    summarize("template draft", template_results)

    greedy_decode = median([r.decode_sec for r in greedy_results])
    template_decode = median([r.decode_sec for r in template_results])

    if template_decode > 0:
        print("decode_sec_speedup:", f"{greedy_decode / template_decode:.3f}x")

    print("OK: benchmark completed with token match")


if __name__ == "__main__":
    main()
